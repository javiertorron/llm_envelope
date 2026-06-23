# Especificación de Configuración: Gemma 4 12b

Este documento describe la estructura estricta del archivo `config.json` esperado por el envelope para el modelo **Gemma 4 12b**. Debido a que el envoltorio ha sido diseñado exclusivamente para este modelo, la presencia y el tipo de dato de todos y cada uno de estos campos es **obligatoria**. El sistema abortará la carga si se encuentra cualquier desviación o campo desconocido.

## 🧠 Estructura Principal (`ModelConfig`)

Define el esquema general del modelo multimodal unificado:

| Campo | Tipo | Ejemplo / Valor esperado | Descripción |
|---|---|---|---|
| `architectures` | `Vec<String>` | `["Gemma4UnifiedForConditionalGeneration"]` | Define la clase de modelo base de HuggingFace. |
| `model_type` | `String` | `"gemma4_unified"` | Identificador principal de la arquitectura unificada de Google. |
| `dtype` | `String` | `"bfloat16"` | Precisión de los tensores. |
| `tie_word_embeddings` | `bool` | `true` | Indica si los pesos de input embeddings y output logits están atados. |
| `transformers_version` | `String` | `"5.10.0.dev0"` | Versión mínima compatible. |
| `initializer_range` | `f64` | `0.02` | Rango usado para inicializar pesos faltantes. |

### 🔤 Identificadores Especiales (Tokens)
- `audio_token_id`: `258881`
- `boa_token_id`: `256000` (Beginning of Audio)
- `boi_token_id`: `255999` (Beginning of Image)
- `eoa_token_index`: `258883` (End of Audio)
- `eoi_token_id`: `258882` (End of Image)
- `image_token_id`: `258880`
- `video_token_id`: `258884`

---

## 📝 Configuración de Texto (`TextConfig`)

Corazón lógico del LLM para el procesamiento y generación de texto natural.

| Campo | Tipo | Valor en Gemma 4 | Notas |
|---|---|---|---|
| `model_type` | `String` | `"gemma4_unified_text"` | Tipo específico del decodificador de texto. |
| `vocab_size` | `usize` | `262144` | El tamaño colosal del vocabulario de Gemma 4. |
| `hidden_size` | `usize` | `3840` | Dimensión de las capas ocultas. |
| `num_hidden_layers` | `usize` | `48` | Cantidad total de bloques transformer. |
| `num_attention_heads` | `usize` | `16` | Cabezas de atención completas. |
| `num_key_value_heads` | `usize` | `8` | Cabezas de atención agrupadas (GQA). |
| `head_dim` | `usize` | `256` | Dimensión de cada cabeza. |
| `intermediate_size` | `usize` | `15360` | Dimensión interna del MLP feed-forward. |
| `hidden_activation` | `String` | `"gelu_pytorch_tanh"` | Función de activación. |
| `max_position_embeddings`| `usize` | `262144` | Ventana de contexto máxima soportada. |
| `sliding_window` | `usize` | `1024` | Tamaño de ventana deslizante para atención local. |
| `layer_types` | `Vec<String>` | `["sliding_attention", "full_attention", ...]` | Topología de atención híbrida en bloques intercalados. |

### 🌀 Parámetros de RoPE (`RopeParameters`)
Gemma usa `Rotary Position Embeddings` con configuraciones separadas según el tipo de atención (`full` vs `sliding`):
- `full_attention`: `rope_theta` de `1000000.0`, `rope_type` `"proportional"`, `partial_rotary_factor` `0.25`
- `sliding_attention`: `rope_theta` de `10000.0`, `rope_type` `"default"`

---

## 👁️ Configuración de Visión (`VisionConfig`)

Módulo que dota al modelo de capacidades de comprensión visual:

| Campo | Tipo | Valor |
|---|---|---|
| `model_type` | `String` | `"gemma4_unified_vision"` |
| `mm_embed_dim` | `usize` | `3840` |
| `patch_size` | `usize` | `16` |
| `model_patch_size` | `usize` | `48` |
| `mm_posemb_size` | `usize` | `1120` |
| `num_soft_tokens` | `usize` | `280` |

---

## 🎵 Configuración de Audio (`AudioConfig`)

Submódulo encargado del procesamiento acústico:

| Campo | Tipo | Valor |
|---|---|---|
| `model_type` | `String` | `"gemma4_unified_audio"` |
| `audio_embed_dim` | `usize` | `640` |
| `hidden_size` | `usize` | `640` |
| `audio_samples_per_token` | `usize` | `640` |

---

## 🛠️ Personalización e Inferencia (`config_custom.json`)

Dado que la estructura matemática del modelo principal (`config.json`) está totalmente bloqueada, el sobre soporta un archivo opcional secundario llamado `config_custom.json`. 

Este archivo se utiliza exclusivamente para sobreescribir y limitar el comportamiento del modelo durante la inferencia. Actualmente soporta las siguientes modificaciones válidas:

| Campo Permitido | Tipo | Validación Interna (Safety Check) | Descripción |
|---|---|---|---|
| `max_position_embeddings` | `usize` | `No puede ser > al config.json base` | Útil para recortar el contexto permitido (ej: de 262144 a 8192) y rechazar prompts gigantes para ahorrar RAM. |

**Ejemplo de uso de `config_custom.json`:**
```json
{
  "max_position_embeddings": 8192
}
```
Si el archivo existe y es válido, Rust aplicará la sobrescritura y avisará por terminal (`🔧 Override aplicado`).

---

*Nota Arquitectónica: Cualquier variable de tipo diccionario como `label2id` o `id2label` mapea estrictamente a los tipos llave/valor nativos de Rust (`HashMap<String, String>` o `HashMap<String, usize>`).*
