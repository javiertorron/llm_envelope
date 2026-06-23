import sys

with open("src/neural_architecture.rs", "r") as f:
    code = f.read()

# Replace types
code = code.replace("    gate_proj: candle_nn::Linear,", "    gate_proj: CpuLinear,")
code = code.replace("    up_proj: candle_nn::Linear,", "    up_proj: CpuLinear,")
code = code.replace("    down_proj: candle_nn::Linear,", "    down_proj: CpuLinear,")
code = code.replace("    pub q_proj: candle_nn::Linear,", "    pub q_proj: CpuLinear,")
code = code.replace("    pub k_proj: candle_nn::Linear,", "    pub k_proj: CpuLinear,")
code = code.replace("    pub v_proj: candle_nn::Linear,", "    pub v_proj: CpuLinear,")
code = code.replace("    pub o_proj: candle_nn::Linear,", "    pub o_proj: CpuLinear,")

# Replace instantiation
code = code.replace("let gate_proj = candle_nn::linear_no_bias", "let gate_proj = CpuLinear::new(candle_nn::linear_no_bias")
code = code.replace("let up_proj = candle_nn::linear_no_bias", "let up_proj = CpuLinear::new(candle_nn::linear_no_bias")
code = code.replace("let down_proj = candle_nn::linear_no_bias", "let down_proj = CpuLinear::new(candle_nn::linear_no_bias")

code = code.replace("let q_proj = candle_nn::linear_no_bias", "let q_proj = CpuLinear::new(candle_nn::linear_no_bias")
code = code.replace("let k_proj = candle_nn::linear_no_bias", "let k_proj = CpuLinear::new(candle_nn::linear_no_bias")
code = code.replace("let v_proj = candle_nn::linear_no_bias", "let v_proj = CpuLinear::new(candle_nn::linear_no_bias")
code = code.replace("let o_proj = candle_nn::linear_no_bias", "let o_proj = CpuLinear::new(candle_nn::linear_no_bias")

# Add the missing closing parenthesis for CpuLinear::new()
code = code.replace("gate_proj\"))?;", "gate_proj\"))?);")
code = code.replace("up_proj\"))?;", "up_proj\"))?);")
code = code.replace("down_proj\"))?;", "down_proj\"))?);")
code = code.replace("q_proj\"))?;", "q_proj\"))?);")
code = code.replace("k_proj\"))?;", "k_proj\"))?);")
code = code.replace("v_proj\"))?;", "v_proj\"))?);")
code = code.replace("o_proj\"))?;", "o_proj\"))?);")

# Replace matmuls
code = code.replace("let mut attention_scores = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;", "let mut attention_scores = (crate::neural_architecture::matmul_bf16(&query_states, &key_states.transpose(2, 3)?)? * scale)?;")
code = code.replace("let context_layer = attention_probs.matmul(&value_states)?;", "let context_layer = crate::neural_architecture::matmul_bf16(&attention_probs, &value_states)?;")
code = code.replace("hidden_states.matmul(&embeddings.t()?)", "crate::neural_architecture::matmul_bf16(hidden_states, &embeddings.t()?)")

with open("src/neural_architecture.rs", "w") as f:
    f.write(code)
