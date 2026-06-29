use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufReader;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    pub device_type: String,
    pub model: Option<String>,
    pub gguf: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            device_type: "cpu".to_string(),
            model: None,
            gguf: None,
        }
    }
}

pub fn load_server_config() -> ServerConfig {
    let config_path = std::path::Path::new("envelope_config.json");
    if config_path.exists() {
        if let Ok(config_data) = fs::read_to_string(&config_path) {
            return serde_json::from_str(&config_data).unwrap_or_default();
        }
    }
    ServerConfig::default()
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioConfig {
    pub _name_or_path: String,
    pub architectures: Option<Vec<String>>,
    pub audio_embed_dim: usize,
    pub audio_samples_per_token: usize,
    pub chunk_size_feed_forward: usize,
    pub dtype: Option<String>,
    pub hidden_size: usize,
    pub id2label: HashMap<String, String>,
    pub initializer_range: f64,
    pub is_encoder_decoder: bool,
    pub label2id: HashMap<String, usize>,
    pub model_type: String,
    pub output_attentions: bool,
    pub output_hidden_states: bool,
    pub output_proj_dims: usize,
    pub problem_type: Option<String>,
    pub return_dict: bool,
    pub rms_norm_eps: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RopeAttentionConfig {
    pub partial_rotary_factor: Option<f64>,
    pub rope_theta: f64,
    pub rope_type: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RopeParameters {
    pub full_attention: RopeAttentionConfig,
    pub sliding_attention: RopeAttentionConfig,
}

fn default_attn_logit_softcapping() -> Option<f64> {
    Some(50.0)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextConfig {
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub attention_k_eq_v: bool,
    pub bos_token_id: u32,
    pub enable_moe_block: bool,
    pub eos_token_id: u32,
    #[serde(default = "default_attn_logit_softcapping", alias = "attention_logit_cap")]
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: f64,
    pub global_head_dim: usize,
    pub head_dim: usize,
    pub hidden_activation: String,
    pub hidden_size: usize,
    pub hidden_size_per_layer_input: usize,
    pub initializer_range: f64,
    pub intermediate_size: usize,
    pub layer_types: Vec<String>,
    pub max_position_embeddings: usize,
    pub model_type: String,
    pub moe_intermediate_size: Option<usize>,
    pub num_attention_heads: usize,
    pub num_experts: Option<usize>,
    pub num_global_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub num_kv_shared_layers: usize,
    pub pad_token_id: u32,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,
    pub sliding_window: usize,
    pub tie_word_embeddings: bool,
    pub top_k_experts: Option<usize>,
    pub use_bidirectional_attention: String,
    pub use_cache: bool,
    pub use_double_wide_mlp: bool,
    pub vocab_size: usize,
    pub vocab_size_per_layer_input: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VisionConfig {
    pub _name_or_path: String,
    pub architectures: Option<Vec<String>>,
    pub chunk_size_feed_forward: usize,
    pub dtype: Option<String>,
    pub id2label: HashMap<String, String>,
    pub initializer_range: f64,
    pub is_encoder_decoder: bool,
    pub label2id: HashMap<String, usize>,
    pub mm_embed_dim: usize,
    pub mm_posemb_size: usize,
    pub model_patch_size: usize,
    pub model_type: String,
    pub num_soft_tokens: usize,
    pub output_attentions: bool,
    pub output_hidden_states: bool,
    pub output_proj_dims: usize,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub problem_type: Option<String>,
    pub return_dict: bool,
    pub rms_norm_eps: f64,
}

/// Estructura de configuración estricta para el modelo.
/// Usamos `deny_unknown_fields` para garantizar que no se ignore NINGÚN campo.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub audio_config: AudioConfig,
    pub audio_token_id: u32,
    pub boa_token_id: u32,
    pub boi_token_id: u32,
    pub dtype: String,
    pub eoa_token_index: u32,
    pub eoi_token_id: u32,
    pub image_token_id: u32,
    pub initializer_range: f64,
    pub model_type: String,
    pub text_config: TextConfig,
    pub tie_word_embeddings: bool,
    pub transformers_version: String,
    pub video_token_id: u32,
    pub vision_config: VisionConfig,
}

/// Estructura para capturar sobreescrituras permitidas por el usuario.
/// Se lee desde el archivo opcional `config_custom.json`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustomConfig {
    pub max_position_embeddings: Option<usize>,
}

pub struct EnvelopeLlm {
    pub config: ModelConfig,
    pub model: crate::neural_architecture::Gemma4Model,
}

impl EnvelopeLlm {
    /// Inicializa y carga la configuración (y futuramente los pesos) a partir de la ruta base
    pub fn load(
        base_path: &std::path::Path,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config_path = base_path.join("config.json");

        let file = File::open(&config_path)?;
        let reader = BufReader::new(file);

        // Transformamos el JSON de forma estricta. Fallará si hay un campo no contemplado.
        let mut config: ModelConfig = serde_json::from_reader(reader)?;
        let original_max_pos = config.text_config.max_position_embeddings;

        // Sobreescritura opcional de parámetros (si existe config_custom.json)
        let custom_config_path = base_path.join("config_custom.json");
        if custom_config_path.exists() {
            let custom_file = File::open(&custom_config_path)?;
            let custom_reader = BufReader::new(custom_file);
            let custom_config: CustomConfig = serde_json::from_reader(custom_reader)?;

            if let Some(new_max_pos) = custom_config.max_position_embeddings {
                if new_max_pos > original_max_pos {
                    return Err(format!("max_position_embeddings ({}) no puede ser mayor que el límite físico del modelo ({})", new_max_pos, original_max_pos).into());
                }
                config.text_config.max_position_embeddings = new_max_pos;
                println!(
                    "🔧 Override aplicado: max_position_embeddings reducido a {}",
                    new_max_pos
                );
            }
        }

        println!("✅ Configuración estricta del LLM cargada correctamente.");

        // === FASE DE MEMORY MAPPING (mmap) ===

        // 2. Establecemos el dispositivo según el archivo envelope_config.json en el directorio de ejecución
        let config_path = std::path::Path::new("envelope_config.json");
        let server_config: ServerConfig = if config_path.exists() {
            let config_data = fs::read_to_string(&config_path)?;
            serde_json::from_str(&config_data).unwrap_or_default()
        } else {
            ServerConfig::default()
        };

        let device = match server_config.device_type.to_lowercase().as_str() {
            "cuda" | "gpu" => {
                println!("🚀 Intentando inicializar CUDA/GPU...");
                candle_core::Device::new_cuda(0).unwrap_or_else(|e| {
                    println!("⚠️ Falló la inicialización de CUDA: {:?}. Usando CPU como fallback.", e);
                    candle_core::Device::Cpu
                })
            },
            "metal" => {
                println!("🚀 Intentando inicializar Apple Metal...");
                candle_core::Device::new_metal(0).unwrap_or_else(|_| {
                    println!("⚠️ Falló la inicialización de Metal. Usando CPU como fallback.");
                    candle_core::Device::Cpu
                })
            },
            _ => {
                println!("🚀 Usando CPU por defecto.");
                candle_core::Device::Cpu
            }
        };

        let gguf_filename = server_config.gguf.unwrap_or_else(|| "model.gguf".to_string());
        let gguf_path = base_path.join(gguf_filename);

        println!("⏳ Mapeando archivo GGUF: {:?}...", gguf_path);
        let mut file = std::fs::File::open(&gguf_path).map_err(|e| format!("Error al abrir GGUF: {}", e))?;

        // Leemos la estructura del GGUF
        use candle_core::quantized::gguf_file;
        let content = gguf_file::Content::read(&mut file).map_err(|e| format!("Error parseando GGUF: {}", e))?;

        let mut q_tensors = HashMap::new();
        for (tensor_name, _info) in content.tensor_infos.iter() {
            if let Ok(qtensor) = content.tensor(&mut file, tensor_name, &device) {
                q_tensors.insert(tensor_name.clone(), qtensor);
            }
        }

        println!("🏗️ Instanciando red neuronal Gemma4 Cuantizada (QMatMul)...");
        let model = crate::neural_architecture::Gemma4Model::load(
            &mut q_tensors,
            &config.text_config,
            &device,
        )?;
        
        println!("✅ GGUF cargado exitosamente en dispositivo: {:?}", device);

        Ok(Self { config, model })
    }

    /// Genera la respuesta del modelo procesando los tokens de entrada y produciendo nuevos autorregresivamente.
    pub fn generate<F>(
        &self,
        prompt_tokens: &[u32],
        temperature: f64,
        max_tokens: usize,
        mut on_token: F,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    where
        F: FnMut(u32) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
    {
        use candle_core::IndexOp;
        use candle_transformers::generation::LogitsProcessor;

        // 1. Limpiar Caché KV de ejecuciones anteriores
        self.model.clear_kv_cache();

        let device = &candle_core::Device::Cpu;
        let temp = if temperature < 1e-7 { None } else { Some(temperature) };
        let mut logits_processor = LogitsProcessor::new(299792458, temp, None);

        let mut tokens = prompt_tokens.to_vec();

        for index in 0..max_tokens {
            let context_size = if index == 0 { tokens.len() } else { 1 };
            let start_pos = if index == 0 { 0 } else { tokens.len() - 1 };

            // Construir tensor de entrada: [1, seq_len]
            let input_slice = &tokens[start_pos..start_pos + context_size];
            let input_tensor = candle_core::Tensor::new(input_slice, device)?.unsqueeze(0)?;

            // Pase hacia adelante (Forward Pass)
            let hidden_states = self.model.forward(&input_tensor, start_pos)?;

            // Obtener el último estado oculto: (1, seq_len, hidden_size) -> (1, hidden_size)
            let last_hidden = hidden_states.i((0, context_size - 1))?.unsqueeze(0)?;

            // Calcular logits finales (LM Head)
            let mut final_logits = self.model.lm_head(&last_hidden)?.squeeze(0)?; // (vocab_size)

            // Convertir a f32 para Softcapping y Muestreo
            final_logits = final_logits.to_dtype(candle_core::DType::F32)?;

            // Softcapping (Gemma 4 usa 30.0 por defecto)
            let softcap = self.config.text_config.final_logit_softcapping as f64;
            if softcap > 0.0 {
                final_logits = ((final_logits / softcap)?.tanh()? * softcap)?;
            }

            // Muestreo probabilístico (Temperature / Top-K integrados en el LogitsProcessor)
            let next_token = logits_processor.sample(&final_logits)?;

            println!("[DEBUG] iter={}, next_token={}", index, next_token);

            tokens.push(next_token);
            on_token(next_token)?;

            // Condición de Parada
            if next_token == self.config.text_config.eos_token_id {
                println!("[DEBUG] EOS alcanzado.");
                break;
            }
        }

        Ok(())
    }
}
