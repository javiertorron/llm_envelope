use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    routing::post,
    Json, Router,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, sync::Arc};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use crate::llm::EnvelopeLlm;
use tokenizers::Tokenizer;

#[derive(Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<usize>,
}

#[derive(Serialize)]
pub struct ChatCompletionResponseChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

pub struct AppState {
    pub llm: Arc<EnvelopeLlm>,
    pub tokenizer: Arc<std::sync::Mutex<Tokenizer>>,
}

pub async fn start_server(state: Arc<AppState>, port: u16) {
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    println!("🚀 Servidor HTTP nativo escuchando en http://{}", addr);
    
    axum::serve(listener, app).await.unwrap();
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::unbounded_channel();
    
    // Preparar el prompt usando la plantilla de chat de Gemma
    let prompt = req.messages.iter()
        .map(|m| format!("<start_of_turn>{}\n{}<end_of_turn>\n", m.role, m.content))
        .collect::<String>() + "<start_of_turn>model\n";
    
    let tokenizer = state.tokenizer.lock().unwrap().clone();
    let prompt_tokens = tokenizer.encode(prompt, true).unwrap().get_ids().to_vec();
    
    let temperature = req.temperature.unwrap_or(0.0);
    let max_tokens = req.max_tokens.unwrap_or(1024);
    
    let llm = state.llm.clone();
    let tokenizer_for_decode = tokenizer.clone();
    let model_name = req.model.clone();
    
    tokio::task::spawn_blocking(move || {
        let mut _tokenizer = tokenizer_for_decode;
        let res = llm.generate(&prompt_tokens, temperature, max_tokens, |token| {
            // Decodificar token a string
            if let Some(text) = _tokenizer.decode(&[token], false).ok() {
                // Evitamos mandar strings vacíos que puedan romper SSE parseo
                if text.is_empty() { return Ok(()); }
                
                let chunk = ChatCompletionResponseChunk {
                    id: "chatcmpl-123".to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created: 1677652288,
                    model: model_name.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta { content: Some(text) },
                        finish_reason: None,
                    }],
                };
                let json = serde_json::to_string(&chunk).unwrap();
                let _ = tx.send(json);
            }
            Ok(())
        });
        
        if let Err(e) = res {
            println!("❌ Error durante la generación: {}", e);
        }
        
        // Enviar token de cierre finalizando el stream
        let chunk = ChatCompletionResponseChunk {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1677652288,
            model: model_name.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta { content: None },
                finish_reason: Some("stop".to_string()),
            }],
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let _ = tx.send(json);
        let _ = tx.send("[DONE]".to_string());
    });

    let stream = UnboundedReceiverStream::new(rx).map(|data| {
        if data == "[DONE]" {
            Ok(Event::default().data("[DONE]"))
        } else {
            Ok(Event::default().data(data))
        }
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
}
