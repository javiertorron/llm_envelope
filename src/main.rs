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

    // 1. Convertimos texto a tokens de entrada (Prompt)
    match tokenizer.encode(prompt) {
        Ok(prompt_tokens) => {
            let prompt_tokens_count = prompt_tokens.len();
            println!("📥 Prompt tokenizado (IDs): {:?}", prompt_tokens);
            
            // [AQUÍ IRÁ LA INFERENCIA DEL LLM EN EL FUTURO]
            // Simulamos que el LLM genera unos tokens de respuesta (ahora mismo, repetimos el prompt para probar el decoder)
            let generated_tokens = prompt_tokens.clone();
            let generated_tokens_count = generated_tokens.len();

            // 2. Convertimos tokens generados de vuelta a texto (Decode)
            match tokenizer.decode(&generated_tokens) {
                Ok(decoded_text) => {
                    println!("📤 Texto restaurado (Decode): {}", decoded_text);
                    
                    // 3. Contabilidad total de uso para facturación/tasas
                    let total_tokens_used = prompt_tokens_count + generated_tokens_count;
                    println!("\n📊 --- REPORTE DE USO ---");
                    println!("   Tokens de Entrada (Prompt): {}", prompt_tokens_count);
                    println!("   Tokens de Salida (Generados): {}", generated_tokens_count);
                    println!("   Total de Tokens a facturar: {}", total_tokens_used);
                }
                Err(e) => eprintln!("❌ Error al decodificar los tokens: {}", e),
            }
        }
        Err(e) => eprintln!("❌ Error al codificar el prompt: {}", e),
    }
}
