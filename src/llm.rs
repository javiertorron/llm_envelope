use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextConfig {
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub attention_k_eq_v: bool,
    pub bos_token_id: u32,
    pub enable_moe_block: bool,
    pub eos_token_id: u32,
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
    pub fn load(base_path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
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
                println!("🔧 Override aplicado: max_position_embeddings reducido a {}", new_max_pos);
            }
        }
        
        println!("✅ Configuración estricta del LLM cargada correctamente.");

        // === FASE DE MEMORY MAPPING (mmap) ===
        use candle_core::{DType, Device};
        use candle_nn::VarBuilder;
        use std::path::PathBuf;

        // 1. Recolectar todos los archivos .safetensors en el directorio
        let mut safetensor_files: Vec<PathBuf> = Vec::new();
        for entry in std::fs::read_dir(base_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "safetensors" {
                        safetensor_files.push(path);
                    }
                }
            }
        }

        if safetensor_files.is_empty() {
            return Err("No se encontraron archivos .safetensors en el directorio base.".into());
        }
        // Ordenarlos alfabéticamente para asegurar que se mapeen en orden (00001, 00002...)
        safetensor_files.sort();

        // 2. Establecemos el dispositivo y la precisión
        // Por autonomía e independencia, forzamos CPU. Gemma usa BF16 de forma nativa.
        let device = Device::Cpu;
        println!("⏳ Mapeando {} archivos safetensors (mmap) en memoria virtual...", safetensor_files.len());

        // 3. Crear el VarBuilder con memory mapping (unsafe porque el SO asume que nadie borrará los archivos mientras corre)
        let _vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&safetensor_files, DType::BF16, &device)?
        };

        println!("🏗️ Instanciando red neuronal Gemma4Unified en RAM virtual...");
        // Pasamos _vb.pp("model.language_model") porque todos los tensores cuelgan de esa raíz.
        let model = crate::neural_architecture::Gemma4Model::load(_vb.pp("model").pp("language_model"), &config.text_config, &device)?;
        println!("✅ Memoria virtual mapeada y arquitectura ensamblada. El modelo está listo para inferir.");

        Ok(Self { config, model })
    }

    /// Genera la respuesta del modelo recibiendo los tokens del prompt
    pub fn generate(&self, _prompt_tokens: &[u32]) -> Result<Vec<u32>, Box<dyn std::error::Error + Send + Sync>> {
        // Lógica futura de inferencia paso a paso
        Ok(Vec::new())
    }
}


