use serde::{Deserialize};

#[derive(Debug, Deserialize)]
struct TextConfig {
    pub global_head_dim: usize,
}

fn main() {
    let data = std::fs::read_to_string("/home/lordvermiis/llms/gemma-4-12b/config.json").unwrap();
    let v: serde_json::Value = serde_json::from_str(&data).unwrap();
    let text_config: TextConfig = serde_json::from_value(v["text_config"].clone()).unwrap();
    println!("global_head_dim = {}", text_config.global_head_dim);
}
