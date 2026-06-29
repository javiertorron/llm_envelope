use candle_core::{DType, Device, Result, Tensor, D};
use candle_core::quantized::{QMatMul, QTensor};
use candle_nn::Module;
use std::collections::HashMap;
use std::sync::Mutex;
use crate::llm::{TextConfig, RopeAttentionConfig};

/// Helper para extraer y consumir QTensor del mapa GGUF
pub fn get_qtensor(tensors: &mut HashMap<String, QTensor>, name: &str) -> Result<QTensor> {
    tensors.remove(name).ok_or_else(|| candle_core::Error::Msg(format!("Missing tensor: {}", name)))
}

/// Normalización Root Mean Square (RMSNorm) exclusiva para la arquitectura Gemma.
#[derive(Debug)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn load(_size: usize, eps: f64, tensors: &mut HashMap<String, QTensor>, name: &str, device: &Device) -> Result<Self> {
        let weight_q = get_qtensor(tensors, name)?;
        let weight = weight_q.dequantize(device)?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        
        let x_f32 = x.to_dtype(internal_dtype)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let x_normed = x_normed.to_dtype(x_dtype)?;
        
        x_normed.broadcast_mul(&self.weight)
    }
}

/// Caché Key-Value
#[derive(Debug, Clone)]
pub struct KVCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl KVCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    pub fn clear(&mut self) {
        self.k = None;
        self.v = None;
    }

    pub fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor)> {
        match (&self.k, &self.v) {
            (Some(k_past), Some(v_past)) => {
                let k = Tensor::cat(&[k_past, new_k], 2)?;
                let v = Tensor::cat(&[v_past, new_v], 2)?;
                self.k = Some(k.clone());
                self.v = Some(v.clone());
                Ok((k, v))
            }
            _ => {
                self.k = Some(new_k.clone());
                self.v = Some(new_v.clone());
                Ok((new_k.clone(), new_v.clone()))
            }
        }
    }
}

/// RoPE
#[derive(Debug)]
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dtype: DType, device: &Device, dim: usize, max_seq_len: usize, config: &RopeAttentionConfig) -> Result<Self> {
        let base = config.rope_theta;
        let mut inv_freq: Vec<f32> = Vec::new();

        let partial_factor = config.partial_rotary_factor.unwrap_or(1.0);
        let rope_angles = (dim as f64 * partial_factor) as usize / 2;

        for i in (0..(rope_angles * 2)).step_by(2) {
            inv_freq.push(1.0 / base.powf(i as f64 / dim as f64) as f32);
        }

        let nope_angles = (dim / 2).saturating_sub(rope_angles);
        for _ in 0..nope_angles {
            inv_freq.push(0.0);
        }

        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (inv_freq_len,), device)?;
        let t: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let t = Tensor::from_vec(t, (max_seq_len,), device)?;
        let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let freqs = Tensor::cat(&[&freqs, &freqs], 1)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        Ok(Self { cos, sin })
    }

    pub fn forward(&self, x: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_sz, _n_head, seq_len, n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
        
        let x1 = x.narrow(D::Minus1, 0, n_embd / 2)?;
        let x2 = x.narrow(D::Minus1, n_embd / 2, n_embd / 2)?;
        let x2_neg = x2.neg()?;
        let rotate_half = Tensor::cat(&[&x2_neg, &x1], D::Minus1)?;
        
        let res = (x.broadcast_mul(&cos)? + rotate_half.broadcast_mul(&sin)?)?;
        Ok(res)
    }
}

#[derive(Debug)]
pub struct GemmaMlp {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

impl GemmaMlp {
    pub fn load(tensors: &mut HashMap<String, QTensor>, prefix: &str) -> Result<Self> {
        let gate_proj = QMatMul::from_qtensor(get_qtensor(tensors, &format!("{}.ffn_gate.weight", prefix))?)?;
        let up_proj = QMatMul::from_qtensor(get_qtensor(tensors, &format!("{}.ffn_up.weight", prefix))?)?;
        let down_proj = QMatMul::from_qtensor(get_qtensor(tensors, &format!("{}.ffn_down.weight", prefix))?)?;
        Ok(Self { gate_proj, up_proj, down_proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::Activation::NewGelu.forward(&gate)?;
        let up = self.up_proj.forward(x)?;
        let intermediate = (gate * up)?;
        self.down_proj.forward(&intermediate)
    }
}

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

#[derive(Debug)]
pub struct GemmaAttention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: Option<QMatMul>,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    is_sliding: bool,
    sliding_window: usize,
    attn_logit_softcapping: Option<f64>,
    kv_cache: Mutex<KVCache>,
}

impl GemmaAttention {
    pub fn load(
        tensors: &mut HashMap<String, QTensor>,
        prefix: &str,
        config: &TextConfig,
        is_sliding: bool,
        device: &Device,
    ) -> Result<Self> {
        let q_proj = QMatMul::from_qtensor(get_qtensor(tensors, &format!("{}.attn_q.weight", prefix))?)?;
        let k_proj = QMatMul::from_qtensor(get_qtensor(tensors, &format!("{}.attn_k.weight", prefix))?)?;
        
        let v_proj = match tensors.remove(&format!("{}.attn_v.weight", prefix)) {
            Some(qt) => Some(QMatMul::from_qtensor(qt)?),
            None => None,
        };
        
        let o_proj = QMatMul::from_qtensor(get_qtensor(tensors, &format!("{}.attn_output.weight", prefix))?)?;
        
        let head_dim = if !is_sliding && config.global_head_dim > 0 {
            config.global_head_dim
        } else {
            config.head_dim
        };

        let q_norm = RmsNorm::load(head_dim, config.rms_norm_eps, tensors, &format!("{}.attn_q_norm.weight", prefix), device)?;
        let k_norm = RmsNorm::load(head_dim, config.rms_norm_eps, tensors, &format!("{}.attn_k_norm.weight", prefix), device)?;

        let use_alternative_attention = config.attention_k_eq_v && !is_sliding;
        let num_kv_heads = if use_alternative_attention {
            config.num_global_key_value_heads
        } else {
            config.num_key_value_heads
        };

        let num_heads = config.num_attention_heads;
        let num_kv_groups = num_heads / num_kv_heads;

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
            attn_logit_softcapping: config.attn_logit_softcapping,
            kv_cache: Mutex::new(KVCache::new()),
        })
    }

    pub fn forward(&self, x: &Tensor, rotary_emb: &RotaryEmbedding, seqlen_offset: usize) -> Result<Tensor> {
        let (b_sz, seq_len, _hidden_size) = x.dims3()?;

        let query_states = self.q_proj.forward(x)?;
        let key_states = self.k_proj.forward(x)?;
        
        let value_states = match &self.v_proj {
            Some(vp) => vp.forward(x)?,
            None => key_states.clone(),
        };

        let query_states = query_states.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?;
        let query_states = self.q_norm.forward(&query_states)?.transpose(1, 2)?;
        
        let key_states = key_states.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;
        let key_states = self.k_norm.forward(&key_states)?.transpose(1, 2)?;
        
        let value_states = value_states
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let query_states = rotary_emb.forward(&query_states, seqlen_offset)?;
        let key_states = rotary_emb.forward(&key_states, seqlen_offset)?;

        let (key_states, value_states) = {
            let mut cache = self.kv_cache.lock().unwrap();
            cache.append(&key_states, &value_states)?
        };

        let key_states = repeat_kv(key_states, self.num_kv_groups)?;
        let value_states = repeat_kv(value_states, self.num_kv_groups)?;

        let mut attn_weights = query_states.matmul(&key_states.transpose(2, 3)?)?;

        // Attention Scaling
        attn_weights = (attn_weights / (self.head_dim as f64).sqrt())?;
        
        // Attention Softcapping (Gemma4 specific)
        if let Some(softcap) = self.attn_logit_softcapping {
            attn_weights = ((attn_weights / softcap)?.tanh()? * softcap)?;
        }

        let attn_weights = if seq_len > 1 {
            let mask = self.get_causal_mask(seq_len, seqlen_offset, key_states.dim(2)?, x.dtype(), x.device())?;
            let attn_weights = attn_weights.broadcast_add(&mask)?;
            candle_nn::ops::softmax(&attn_weights, D::Minus1)?
        } else {
            candle_nn::ops::softmax(&attn_weights, D::Minus1)?
        };

        let attn_output = attn_weights.matmul(&value_states)?;
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;

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
    pub fn load(tensors: &mut HashMap<String, QTensor>, prefix: &str, config: &TextConfig, is_sliding: bool, device: &Device) -> Result<Self> {
        let self_attn = GemmaAttention::load(tensors, prefix, config, is_sliding, device)?;
        let mlp = GemmaMlp::load(tensors, prefix)?;
        
        let input_layernorm = RmsNorm::load(config.hidden_size, config.rms_norm_eps, tensors, &format!("{}.attn_norm.weight", prefix), device)?;
        let post_attention_layernorm = RmsNorm::load(config.hidden_size, config.rms_norm_eps, tensors, &format!("{}.post_attention_norm.weight", prefix), device)?;
        
        let pre_feedforward_layernorm = RmsNorm::load(config.hidden_size, config.rms_norm_eps, tensors, &format!("{}.ffn_norm.weight", prefix), device)?;
        let post_feedforward_layernorm = RmsNorm::load(config.hidden_size, config.rms_norm_eps, tensors, &format!("{}.post_ffw_norm.weight", prefix), device)?;
        
        let layer_scalar = get_qtensor(tensors, &format!("{}.layer_output_scale.weight", prefix))
            .and_then(|qt| qt.dequantize(device)).ok();
        
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

pub struct Gemma4Model {
    pub embed_tokens: candle_nn::Embedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: RmsNorm,
    rotary_emb_full: RotaryEmbedding,
    rotary_emb_sliding: RotaryEmbedding,
}

impl Gemma4Model {
    pub fn load(tensors: &mut HashMap<String, QTensor>, config: &TextConfig, device: &Device) -> Result<Self> {
        let embed_q = get_qtensor(tensors, "token_embd.weight")?;
        let embed_t = embed_q.dequantize(device)?;
        let embed_tokens = candle_nn::Embedding::new(embed_t, config.hidden_size);
        
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let is_sliding = config.layer_types
                .get(layer_idx)
                .map(|s| s == "sliding_attention")
                .unwrap_or(false);
                
            let prefix = format!("blk.{}", layer_idx);
            let layer = DecoderLayer::load(tensors, &prefix, config, is_sliding, device)?;
            layers.push(layer);
        }
        
        let norm = RmsNorm::load(config.hidden_size, config.rms_norm_eps, tensors, "output_norm.weight", device)?;
        
        let rotary_emb_full = RotaryEmbedding::new(
            DType::F32, device, config.global_head_dim, config.max_position_embeddings, &config.rope_parameters.full_attention
        )?;
        let rotary_emb_sliding = RotaryEmbedding::new(
            DType::F32, device, config.head_dim, config.max_position_embeddings, &config.rope_parameters.sliding_attention
        )?;
        
        Ok(Self { embed_tokens, layers, norm, rotary_emb_full, rotary_emb_sliding })
    }

    pub fn forward(&self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        let hidden_size = x.dim(D::Minus1)?;
        
        x = (x * (hidden_size as f64).sqrt())?;
        
        for (i, layer) in self.layers.iter().enumerate() {
            println!("[DEBUG] Ejecutando capa {}", i);
            let rotary_emb = if layer.self_attn.is_sliding {
                &self.rotary_emb_sliding
            } else {
                &self.rotary_emb_full
            };
            x = layer.forward(&x, rotary_emb, seqlen_offset)?;
        }
        
        self.norm.forward(&x)
    }

    pub fn clear_kv_cache(&self) {
        for layer in &self.layers {
            layer.clear_kv_cache();
        }
    }

    pub fn lm_head(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let embeddings = self.embed_tokens.embeddings();
        // hidden_states can be [batch, seq_len, hidden_size] or [batch, hidden_size]
        
        let logits = if hidden_states.rank() == 3 {
            // [batch, seq_len, hidden_size]
            let hidden_states = hidden_states.narrow(1, hidden_states.dim(1)? - 1, 1)?; // [batch, 1, hidden_size]
            let w = embeddings.broadcast_left(hidden_states.dim(0)?)?; // [batch, vocab_size, hidden_size]
            hidden_states.matmul(&w.t()?)?.squeeze(1)? // [batch, vocab_size]
        } else {
            // [batch, hidden_size]
            hidden_states.matmul(&embeddings.t()?)? // [batch, vocab_size]
        };
        
        // Final Logit Softcapping
        let softcap = 30.0;
        let mut logits = ((logits / softcap)?.tanh()? * softcap)?;
        
        Ok(logits)
    }
}
