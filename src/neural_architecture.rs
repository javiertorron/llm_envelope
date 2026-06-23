use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Module, VarBuilder};
use std::sync::{Arc, Mutex};

/// Helper de multiplicación de matrices que soporta BF16 en CPU
pub fn matmul_bf16(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.dtype() == DType::BF16 && a.device().is_cpu() {
        let a_f32 = a.to_dtype(DType::F32)?;
        let b_f32 = b.to_dtype(DType::F32)?;
        let res = a_f32.matmul(&b_f32)?;
        res.to_dtype(DType::BF16)
    } else {
        a.matmul(b)
    }
}

/// Envoltorio sobre candle_nn::Linear que usa matmul_bf16
#[derive(Debug)]
pub struct CpuLinear {
    inner: candle_nn::Linear,
}

impl CpuLinear {
    pub fn new(inner: candle_nn::Linear) -> Self {
        Self { inner }
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.inner.weight();
        let mut out = matmul_bf16(x, &w.t()?)?;
        if let Some(bias) = self.inner.bias() {
            out = out.broadcast_add(bias)?;
        }
        Ok(out)
    }
}

/// Normalización Root Mean Square (RMSNorm) exclusiva para la arquitectura Gemma.
/// A diferencia del estándar, Gemma suma 1.0 a los pesos aprendidos de forma matemática.
#[derive(Debug)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        // En los safetensors de Gemma, estos pesos suelen llamarse "weight"
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        // Para mayor estabilidad numérica, elevamos la precisión a F32 para calcular la varianza.
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        
        let x_f32 = x.to_dtype(internal_dtype)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let x_normed = x_normed.to_dtype(x_dtype)?;
        
        // Peculiaridad de Gemma: (1.0 + weight)
        let weight_plus_one = (&self.weight + 1.0)?;
        x_normed.broadcast_mul(&weight_plus_one)
    }
}

/// Caché Key-Value para generación autorregresiva rápida.
/// En vez de recalcular todo el contexto en cada token, guardamos los valores K y V de atención pasada.
#[derive(Debug, Clone)]
pub struct KVCache {
    // Almacena el tensor de Keys y Values de la capa
    // Dimensiones típicas: (batch_size, num_kv_heads, seq_len, head_dim)
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl KVCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    /// Limpia la caché para iniciar una nueva generación independiente
    pub fn clear(&mut self) {
        self.k = None;
        self.v = None;
    }

    /// Actualiza la caché adjuntando (concatenando) los nuevos tensores K y V generados.
    pub fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor)> {
        match (&self.k, &self.v) {
            (Some(k_past), Some(v_past)) => {
                // Dimensión 2 suele ser la longitud de secuencia (seq_len) en la arquitectura de atención
                let k = Tensor::cat(&[k_past, new_k], 2)?;
                let v = Tensor::cat(&[v_past, new_v], 2)?;
                self.k = Some(k.clone());
                self.v = Some(v.clone());
                Ok((k, v))
            }
            _ => {
                // Es el primer token, por tanto la caché se inicializa con los tensores entrantes.
                self.k = Some(new_k.clone());
                self.v = Some(new_v.clone());
                Ok((new_k.clone(), new_v.clone()))
            }
        }
    }
}
use candle_nn::{linear_no_bias, Linear};
use crate::llm::{TextConfig, RopeAttentionConfig};

/// RoPE (Rotary Position Embeddings)
#[derive(Debug)]
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dtype: DType, device: &Device, dim: usize, max_seq_len: usize, config: &RopeAttentionConfig) -> Result<Self> {
        let base = config.rope_theta;
        let mut inv_freq: Vec<f32> = Vec::new();
        for i in (0..dim).step_by(2) {
            inv_freq.push(1.0 / base.powf(i as f64 / dim as f64) as f32);
        }
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (inv_freq_len,), device)?;
        let t: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let t = Tensor::from_vec(t, (max_seq_len,), device)?;
        let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let freqs = Tensor::cat(&[&freqs, &freqs], 1)?; // (seq_len, dim)
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        Ok(Self { cos, sin })
    }

    pub fn forward(&self, x: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_sz, _n_head, seq_len, n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, seq_len, dim)
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
        
        let x1 = x.narrow(D::Minus1, 0, n_embd / 2)?;
        let x2 = x.narrow(D::Minus1, n_embd / 2, n_embd / 2)?;
        let x2_neg = x2.neg()?;
        let rotate_half = Tensor::cat(&[&x2_neg, &x1], D::Minus1)?;
        
        let res = (x.broadcast_mul(&cos)? + rotate_half.broadcast_mul(&sin)?)?;
        Ok(res)
    }
}

/// Multi-Layer Perceptron (GemmaMLP)
#[derive(Debug)]
pub struct GemmaMlp {
    gate_proj: CpuLinear,
    up_proj: CpuLinear,
    down_proj: CpuLinear,
}

impl GemmaMlp {
    pub fn load(vb: VarBuilder, config: &TextConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        
        let gate_proj = CpuLinear::new(candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?);
        let up_proj = CpuLinear::new(candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?);
        let down_proj = CpuLinear::new(candle_nn::linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?);
        Ok(Self { gate_proj, up_proj, down_proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        // Usamos la aproximación estándar de GELU (gelu_pytorch_tanh equivalente aprox en candle)
        let gate = candle_nn::Activation::NewGelu.forward(&gate)?;
        let up = self.up_proj.forward(x)?;
        let intermediate = (gate * up)?;
        self.down_proj.forward(&intermediate)
    }
}

/// Helper para expandir las Key y Values en el algoritmo Grouped-Query Attention (GQA)
fn repeat_kv(x: Tensor, num_key_value_groups: usize) -> Result<Tensor> {
    if num_key_value_groups == 1 {
        return Ok(x);
    }
    let (b_sz, num_kv_heads, seq_len, head_dim) = x.dims4()?;
    let x = x
        .unsqueeze(2)?
        .expand((b_sz, num_kv_heads, num_key_value_groups, seq_len, head_dim))?
        .reshape((b_sz, num_kv_heads * num_key_value_groups, seq_len, head_dim))?;
    Ok(x)
}

/// Capa de Atención (GemmaAttention) con soporte GQA, caché y ventana deslizante
#[derive(Debug)]
pub struct GemmaAttention {
    q_proj: CpuLinear,
    k_proj: CpuLinear,
    v_proj: Option<CpuLinear>,
    o_proj: CpuLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    is_sliding: bool,
    sliding_window: usize,
    kv_cache: Mutex<KVCache>,
}

impl GemmaAttention {
    pub fn load(
        vb: VarBuilder,
        config: &TextConfig,
        is_sliding: bool,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        
        let (num_kv_heads, head_dim) = if is_sliding {
            (config.num_key_value_heads, config.head_dim)
        } else {
            (config.num_global_key_value_heads, config.global_head_dim)
        };
        let num_kv_groups = num_heads / num_kv_heads;

        let q_proj = CpuLinear::new(linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?);
        let k_proj = CpuLinear::new(linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?);
        
        // Si is_sliding, existe v_proj. Si no, K=V (attention_k_eq_v) y no hay v_proj.
        let v_proj = if is_sliding {
            Some(CpuLinear::new(linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?))
        } else {
            None
        };
        
        let o_proj = CpuLinear::new(linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?);
        
        let q_norm = RmsNorm::load(head_dim, config.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::load(head_dim, config.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            is_sliding,
            sliding_window: config.sliding_window,
            kv_cache: Mutex::new(KVCache::new()),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        rotary_emb: &RotaryEmbedding,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _hidden_size) = x.dims3()?;

        // 1. Proyecciones iniciales (Q, K, V)
        let query_states = self.q_proj.forward(x)?;
        let key_states = self.k_proj.forward(x)?;
        
        // K = V si v_proj no existe (attention_k_eq_v para full_attention)
        let value_states = match &self.v_proj {
            Some(vp) => vp.forward(x)?,
            None => key_states.clone(),
        };

        // 2. Reformatear para atención y normalizar por cabezal (b_sz, seq_len, num_heads, head_dim)
        let query_states = query_states.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?;
        let query_states = self.q_norm.forward(&query_states)?.transpose(1, 2)?;
        
        let key_states = key_states.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;
        let key_states = self.k_norm.forward(&key_states)?.transpose(1, 2)?;
        
        let value_states = value_states
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // 3. Aplicar los Rotational Embeddings (RoPE)
        let query_states = rotary_emb.forward(&query_states, seqlen_offset)?;
        let key_states = rotary_emb.forward(&key_states, seqlen_offset)?;

        // 4. Actualizar y recuperar Caché
        let (key_states, value_states) = {
            let mut cache = self.kv_cache.lock().unwrap();
            cache.append(&key_states, &value_states)?
        };

        // 5. Expandir K y V si es necesario (Grouped Query Attention)
        let key_states = repeat_kv(key_states, self.num_kv_groups)?;
        let value_states = repeat_kv(value_states, self.num_kv_groups)?;

        // 6. Dot-product attention pura
        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let attn_weights = (crate::neural_architecture::matmul_bf16(&query_states, &key_states.transpose(2, 3)?)? * scale)?;

        // 7. Enmascaramiento condicional (sólo para prompt processing o ventanas grandes)
        let attn_weights = if seq_len > 1 {
            let mask = self.get_causal_mask(seq_len, seqlen_offset, key_states.dim(2)?, x.dtype(), x.device())?;
            let attn_weights = attn_weights.broadcast_add(&mask)?;
            candle_nn::ops::softmax(&attn_weights, D::Minus1)?
        } else {
            candle_nn::ops::softmax(&attn_weights, D::Minus1)?
        };

        // 8. Multiplicar por V y ensamblar
        let attn_output = crate::neural_architecture::matmul_bf16(&attn_weights, &value_states)?;
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;

        // 9. Proyección de salida
        self.o_proj.forward(&attn_output)
    }

    fn get_causal_mask(&self, seq_len: usize, seqlen_offset: usize, kv_len: usize, dtype: DType, device: &Device) -> Result<Tensor> {
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..kv_len).map(move |j| {
                    if j > i + seqlen_offset {
                        f32::NEG_INFINITY
                    } else if self.is_sliding && (i + seqlen_offset).saturating_sub(j) >= self.sliding_window {
                        f32::NEG_INFINITY
                    } else {
                        0f32
                    }
                })
            })
            .collect();
        let mask = Tensor::from_vec(mask, (seq_len, kv_len), device)?;
        mask.unsqueeze(0)?.unsqueeze(0)?.to_dtype(dtype)
    }

    pub fn clear_kv_cache(&self) {
        let mut cache = self.kv_cache.lock().unwrap();
        cache.clear();
    }
}

/// Capa Completa del Decodificador (Atención + MLP + RMSNorms)
#[derive(Debug)]
pub struct DecoderLayer {
    self_attn: GemmaAttention,
    mlp: GemmaMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    layer_scalar: Option<Tensor>,
}

impl DecoderLayer {
    pub fn load(vb: VarBuilder, config: &TextConfig, is_sliding: bool) -> Result<Self> {
        let self_attn = GemmaAttention::load(vb.pp("self_attn"), config, is_sliding)?;
        let mlp = GemmaMlp::load(vb.pp("mlp"), config)?;
        let input_layernorm = RmsNorm::load(config.hidden_size, config.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::load(config.hidden_size, config.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        
        let pre_feedforward_layernorm = RmsNorm::load(config.hidden_size, config.rms_norm_eps, vb.pp("pre_feedforward_layernorm"))?;
        let post_feedforward_layernorm = RmsNorm::load(config.hidden_size, config.rms_norm_eps, vb.pp("post_feedforward_layernorm"))?;
        
        // layer_scalar se encuentra en algunas arquitecturas como gemma4/2
        let layer_scalar = vb.get(1, "layer_scalar").ok();
        
        Ok(Self { 
            self_attn, mlp, input_layernorm, post_attention_layernorm, 
            pre_feedforward_layernorm, post_feedforward_layernorm, layer_scalar 
        })
    }

    pub fn forward(&self, x: &Tensor, rotary_emb: &RotaryEmbedding, seqlen_offset: usize) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(&x)?;
        let x = self.self_attn.forward(&x, rotary_emb, seqlen_offset)?;
        let mut x = self.post_attention_layernorm.forward(&x)?;
        
        if let Some(scalar) = &self.layer_scalar {
            x = x.broadcast_mul(scalar)?;
        }
        let x = (x + residual)?;
        
        let residual = x.clone();
        let x = self.pre_feedforward_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let mut x = self.post_feedforward_layernorm.forward(&x)?;
        
        if let Some(scalar) = &self.layer_scalar {
            x = x.broadcast_mul(scalar)?;
        }
        x + residual
    }

    pub fn clear_kv_cache(&self) {
        self.self_attn.clear_kv_cache();
    }
}

/// El Modelo Gemma 4 Completo (End-to-End)
#[derive(Debug)]
pub struct Gemma4Model {
    pub embed_tokens: candle_nn::Embedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: RmsNorm,
    rotary_emb_full: RotaryEmbedding,
    rotary_emb_sliding: RotaryEmbedding,
}

impl Gemma4Model {
    pub fn load(vb: VarBuilder, config: &TextConfig, device: &Device) -> Result<Self> {
        let embed_tokens = candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
        
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            // El array layer_types intercala "sliding_attention" y "full_attention"
            let is_sliding = config.layer_types
                .get(layer_idx)
                .map(|s| s == "sliding_attention")
                .unwrap_or(false);
                
            let layer = DecoderLayer::load(vb.pp(&format!("layers.{}", layer_idx)), config, is_sliding)?;
            layers.push(layer);
        }
        
        let norm = RmsNorm::load(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        
        // Las instancias de RoPE se comparten entre capas del mismo tipo para ahorrar RAM
        let rotary_emb_full = RotaryEmbedding::new(
            DType::BF16, device, config.global_head_dim, config.max_position_embeddings, &config.rope_parameters.full_attention
        )?;
        let rotary_emb_sliding = RotaryEmbedding::new(
            DType::BF16, device, config.head_dim, config.max_position_embeddings, &config.rope_parameters.sliding_attention
        )?;
        
        Ok(Self { embed_tokens, layers, norm, rotary_emb_full, rotary_emb_sliding })
    }

    pub fn forward(&self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        let hidden_size = x.dim(D::Minus1)?;
        
        // Escalar los embeddings: x = x * sqrt(hidden_size)
        // Secreto matemático de Gemma para dar peso al embedding original frente a las capas profundas
        x = (x * (hidden_size as f64).sqrt())?;
        
        for layer in self.layers.iter() {
            let rotary_emb = if layer.self_attn.is_sliding {
                &self.rotary_emb_sliding
            } else {
                &self.rotary_emb_full
            };
            x = layer.forward(&x, rotary_emb, seqlen_offset)?;
        }
        
        self.norm.forward(&x)
    }

    /// Limpia el estado de la caché KV para preparar al modelo para un nuevo prompt
    pub fn clear_kv_cache(&self) {
        for layer in &self.layers {
            layer.clear_kv_cache();
        }
    }

    /// Proyecta el hidden state final en logits usando los pesos del embedding original (tie_word_embeddings = true)
    pub fn lm_head(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let embeddings = self.embed_tokens.embeddings();
        crate::neural_architecture::matmul_bf16(hidden_states, &embeddings.t()?)
    }
}
