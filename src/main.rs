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

    /// Arranca el servidor HTTP en lugar de usar la consola interactiva
    #[arg(long = "serve", default_value_t = false)]
    serve: bool,
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

    if args.serve {
        // Preparamos el AppState
        let state = std::sync::Arc::new(server::AppState {
            llm: std::sync::Arc::new(llm),
            tokenizer: std::sync::Arc::new(std::sync::Mutex::new(tokenizer.inner)),
        });

        // Arrancamos el servidor
        server::start_server(state, 8080).await;
    } else {
        // Modo Consola Interactiva
        
        loop {
            print!("\n🧑‍💻 Prompt: ");
            io::stdout().flush().unwrap();
            
            let mut prompt = String::new();
            if io::stdin().read_line(&mut prompt).is_err() || prompt.trim().is_empty() {
                continue;
            }
            
            let prompt = prompt.trim();
            if prompt == "exit" || prompt == "quit" {
                break;
            }
            
            // Plantilla obligatoria de Gemma para evitar alucinaciones
            let formatted_prompt = format!("<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n", prompt);
            
            println!("🤖 Respuesta:");
            let prompt_tokens = match tokenizer.encode(&formatted_prompt) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error al tokenizar: {}", e);
                    continue;
                }
            };
            
            // max_tokens = 512 por defecto en consola, con temperatura estándar
            let _ = llm.generate(&prompt_tokens, 0.7, 512, |token_id| {
                if let Ok(text) = tokenizer.decode(&[token_id]) {
                    print!("{}", text);
                    io::stdout().flush().unwrap();
                }
                Ok(())
            });
            
            println!();
            
            // Limpiar caché para el siguiente prompt
            llm.model.clear_kv_cache();
        }
    }
}
