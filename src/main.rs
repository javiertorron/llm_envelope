use clap::Parser;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process;

mod environment;
mod tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about = "Candle LLM Inference Envelope")]
struct Args {
    /// Ruta absoluta al directorio del modelo (opcional)
    #[arg(long = "model")]
    model: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();
    
    // Si pasaron el parámetro --model lo usamos, sino usamos "./model"
    let base_path_buf = args.model.unwrap_or_else(|| PathBuf::from("./model"));
    let base_path = base_path_buf.as_path();

    // Validamos el entorno del modelo (directorio y archivos)
    if let Err(e) = environment::validate_model_environment(base_path) {
        eprintln!("❌ Error crítico de validación de entorno: {}", e);
        process::exit(1);
    }

    let tokenizer = match tokenizer::EnvelopeTokenizer::load(base_path) {
        Ok(tok) => tok,
        Err(e) => {
            eprintln!("❌ Error crítico al cargar el tokenizador: {}", e);
            process::exit(1);
        }
    };

    // Pedimos el prompt al usuario
    print!("> Introduce tu prompt: ");
    io::stdout().flush().unwrap(); // Aseguramos que el mensaje se imprima antes de esperar la entrada

    let mut prompt = String::new();
    io::stdin().read_line(&mut prompt).expect("Error al leer la entrada del usuario");
    let prompt = prompt.trim(); // Limpiamos saltos de línea al final

    if prompt.is_empty() {
        println!("⚠️ No has introducido ningún prompt. Saliendo...");
        return;
    }

    // 1. Convertimos texto a tokens (Encode)
    match tokenizer.encode(prompt) {
        Ok(tokens) => {
            println!("📥 Prompt tokenizado (IDs): {:?}", tokens);

            // 2. Convertimos tokens de vuelta a texto (Decode)
            match tokenizer.decode(&tokens) {
                Ok(decoded_text) => {
                    println!("📤 Texto restaurado (Decode): {}", decoded_text);
                }
                Err(e) => eprintln!("❌ Error al decodificar los tokens: {}", e),
            }
        }
        Err(e) => eprintln!("❌ Error al codificar el prompt: {}", e),
    }
}

