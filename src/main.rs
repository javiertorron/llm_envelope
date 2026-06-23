use clap::Parser;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process;
mod environment;
mod llm;
mod tokenizer;
pub mod neural_architecture;

#[derive(Parser, Debug)]
#[command(author, version, about = "Candle LLM Inference Envelope")]
struct Args {
    /// Ruta absoluta al directorio del modelo (opcional)
    #[arg(long = "model")]
    model: Option<PathBuf>,
}

pub mod server;

#[tokio::main]
async fn main() {
    let args = Args::parse();
    
    // Si pasaron el parámetro --model lo usamos, sino usamos "./model"
    let base_path_buf = args.model.unwrap_or_else(|| PathBuf::from("./model"));
    let base_path = base_path_buf.as_path();

    // Validamos el entorno del modelo (directorio y archivos)
    if let Err(e) = environment::validate_model_environment(base_path) {
        eprintln!("❌ Error crítico de validación de entorno: {}", e);
        process::exit(1);
    }

    // Cargamos la configuración del LLM
    let llm = match llm::EnvelopeLlm::load(base_path) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("❌ Error crítico al cargar la configuración del modelo: {}", e);
            process::exit(1);
        }
    };

    let tokenizer = match tokenizer::EnvelopeTokenizer::load(base_path) {
        Ok(tok) => tok,
        Err(e) => {
            eprintln!("❌ Error crítico al cargar el tokenizador: {}", e);
            process::exit(1);
        }
    };

    // Preparamos el AppState
    let state = std::sync::Arc::new(server::AppState {
        llm: std::sync::Arc::new(llm),
        tokenizer: std::sync::Arc::new(std::sync::Mutex::new(tokenizer.inner)),
    });

    // Arrancamos el servidor
    server::start_server(state, 8080).await;
}
