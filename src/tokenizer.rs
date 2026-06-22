use std::error::Error;
use std::io;
use std::path::Path;
use tokenizers::Tokenizer;

pub struct EnvelopeTokenizer {
    pub inner: Tokenizer,
}

impl EnvelopeTokenizer {
    pub fn load(base_path: &Path) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let tokenizer_config_path = base_path.join("tokenizer.json");

        // Ya sabemos que existe por la validación inicial en main, pero es buena práctica de seguridad interna
        if !tokenizer_config_path.exists() {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "No se encuentra el archivo en la ruta: {:?}",
                    tokenizer_config_path
                ),
            )));
        }

        // El operador `?` propaga automáticamente el error si from_file falla
        let tokenizer = Tokenizer::from_file(&tokenizer_config_path)?;

        println!("✅ Tokenizer cargado correctamente.");
        Ok(Self { inner: tokenizer })
    }

    /// Convierte un texto (prompt) en una lista de IDs de tokens
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn Error + Send + Sync>> {
        // encode recibe el texto y un booleano indicando si debe añadir tokens especiales (ej. <s>, </s>)
        let encoding = self.inner.encode(text, true)?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Convierte una lista de IDs de tokens de vuelta a texto
    pub fn decode(&self, ids: &[u32]) -> Result<String, Box<dyn Error + Send + Sync>> {
        // decode recibe los IDs y un booleano indicando si debe omitir tokens especiales en el texto devuelto
        self.inner.decode(ids, true)
    }
}
