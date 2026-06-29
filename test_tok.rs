use std::path::Path;
use tokenizers::Tokenizer;

fn main() {
    let tokenizer = Tokenizer::from_file(Path::new("/home/lordvermiis/llms/gemma-4-12b/tokenizer.json")).unwrap();
    println!("Special tokens: {:?}", tokenizer.get_added_tokens_decoder().keys().collect::<Vec<_>>());
    println!("Encode '<start_of_turn>user\nte gustan los tests?<end_of_turn>\n<start_of_turn>model\n': {:?}", tokenizer.encode("<start_of_turn>user\nte gustan los tests?<end_of_turn>\n<start_of_turn>model\n", true).unwrap().get_ids());
}
