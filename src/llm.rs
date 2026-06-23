use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;

/// Estructura para la configuración del modelo leída del `config.json`.
/// Definimos los campos mínimos esenciales que suelen estar en todos los modelos.
/// Serde ignorará automáticamente los cientos de campos restantes del json.
#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub model_type: Option<String>,
    pub vocab_size: Option<usize>,
    pub hidden_size: Option<usize>,
}

pub struct EnvelopeLlm {
    pub config: ModelConfig,
    // TODO: Añadir aquí los componentes internos (ej: candle_core::Tensor)
}

impl EnvelopeLlm {
    /// Inicializa y carga la configuración (y futuramente los pesos) a partir de la ruta base
    pub fn load(base_path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config_path = base_path.join("config.json");
        
        // 1. Abrimos el archivo
        let file = File::open(&config_path)?;
        // 2. Usamos BufReader para no colapsar la RAM leyendo poco a poco
        let reader = BufReader::new(file);
        // 3. Transformamos el JSON en nuestra estructura ModelConfig
        let config: ModelConfig = serde_json::from_reader(reader)?;
        
        println!("✅ Configuración del LLM cargada: {:?}", config);

        Ok(Self { config })
    }

    /// Genera la respuesta del modelo recibiendo los tokens del prompt
    pub fn generate(&self, _prompt_tokens: &[u32]) -> Result<Vec<u32>, Box<dyn std::error::Error + Send + Sync>> {
        // Lógica futura de inferencia paso a paso
        Ok(Vec::new())
    }
}

