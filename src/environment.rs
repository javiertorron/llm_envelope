use std::error::Error;
use std::io;
use std::path::Path;

/// Valida que el entorno de ejecución (archivos del modelo) esté preparado.
pub fn validate_model_environment(base_path: &Path) -> Result<(), Box<dyn Error + Send + Sync>> {
    // 1. Verificamos que el path provisto sea un directorio válido
    if !base_path.is_dir() {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "La ruta del modelo especificada no es un directorio o no existe: {:?}",
                base_path
            ),
        )));
    }

    // 2. Archivos requeridos según la especificación funcional
    let required_files = [
        "tokenizer_config.json",
        "tokenizer.json",
        "processor_config.json",
        "config.json",
        "generation_config.json",
        "model.safetensors",
    ];

    // 3. Verificamos que no falte ninguno
    for file in &required_files {
        let file_path = base_path.join(file);
        if !file_path.exists() {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "Falta el archivo requerido '{}' en el directorio del modelo: {:?}",
                    file, base_path
                ),
            )));
        }
    }

    Ok(())
}
