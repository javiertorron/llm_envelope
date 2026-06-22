# Candle LLM Inference Envelope 🦀🚀

Un motor y servidor de inferencia para Modelos de Lenguaje Grande (LLMs) empaquetado en un único ejecutable autónomo y portable, desarrollado 100% en Rust utilizando el framework **Candle** de Hugging Face.

## 📋 Descripción del Proyecto

Este proyecto redefine la distribución de modelos de Inteligencia Artificial mediante el patrón **Model-as-a-Service (MaaS)** en local. A diferencia de las soluciones tradicionales basadas en entornos pesados de Python, este sistema compila todo el ciclo de vida de la inferencia (Tokenización, Carga de Tensores, Procesamiento de Matrices y Servidor API HTTP) dentro de un único binario nativo, optimizado, portable y sin dependencias dinámicas.

El desarrollo se aborda de forma incremental siguiendo una metodología **Agile orientada a Épicas e Hitos Atómicos**, minimizando la parálisis por análisis y garantizando la robustez de cada capa antes de exponerla a la red.

## ✨ Características Principales

- **Runtime 100% Nativo en Rust:** Construido sobre el ecosistema matemático de `candle-core` y `candle-transformers`.
- **Carga de Memoria Eficiente:** Uso de `Memory Mapping (mmap)` a través de `VarBuilder` para inicializar modelos binarios masivos de forma segura sin saturar la memoria RAM.
- **Formato Safetensors:** Compatibilidad nativa con el estándar industrial seguro y veloz de Hugging Face (`.safetensors`).
- **Arquitectura Asíncrona Extrema:** Servidor HTTP ligero basado en `Axum` y el runtime `Tokio`, diseñado para procesar peticiones en microsegundos y mantener un consumo de memoria plano y predecible.
- **Generación en Streaming:** Implementación de Server-Sent Events (SSE) para la transmisión token-a-token (palabra por palabra) en tiempo real por red.

## ⚙️ Comportamiento y Ciclo de Vida

Al ejecutar el binario, el sistema realiza estrictamente las siguientes fases:

1. **Validación de Entorno:** Comprueba la existencia de `config.json`, `tokenizer.json` y `*.safetensors` en la ruta especificada.
2. **Carga Inicial (Warm-up):** Parsea la configuración e inicializa los pesos del modelo en memoria mediante `mmap` **antes** de abrir cualquier puerto de red.
3. **Inicialización de Red:** Levanta el servicio HTTP en el puerto designado (por defecto `8080`).
4. **Interfaz Silenciosa:** Reporta por `stdout` únicamente los hitos críticos (estado de la carga con timestamps, IP/puerto de escucha, y registro básico de peticiones procesadas).

## 🔌 Especificación de la API

El servidor expone un endpoint compatible con el estándar de OpenAI para facilitar la integración con herramientas de la industria.

**Endpoint:** `POST /v1/chat/completions`

**Estructura de la Petición:**

```json
{
  "model": "string",
  "messages": [
    {
      "role": "user",
      "content": "string"
    }
  ],
  "temperature": 0.0,
  "max_tokens": 50
}
```

**Respuesta en Streaming (SSE):**
Se transmite un flujo de eventos `text/event-stream` enviando tokens individuales en tiempo real, finalizando con la señal `[DONE]`:

```json
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"phi3","choices":[{"index":0,"delta":{"content":"palabra"},"finish_reason":null}]}
...
data: [DONE]
```

## 📦 Criterios de Portabilidad (El Binario Autónomo)

El producto final garantiza un despliegue sin fricciones en entornos productivos:

- **Cero Dependencias Dinámicas:** Compilación contra el target `x86_64-unknown-linux-musl` para incrustar la librería de C estáticamente en sistemas Linux.
- **Tamaño Optimizado:** Uso de Link-Time Optimization (`lto = true`) en el perfil de producción para eliminar código muerto y reducir el tamaño binario generado por las librerías matemáticas.
- **Autonomía:** Arranca en instalaciones Linux limpias requiriendo únicamente acceso de lectura a la carpeta de pesos del modelo.

## 🗺️ Mapa de Ruta de Ingeniería (Épicas)

1. **Épica 1: El Motor en Consola (Inferencia Local):** Construcción del núcleo matemático de Candle y el bucle autorregresivo. El modelo carga desde el disco duro y genera texto directamente en la terminal (CLI puro).
2. **Épica 2: El Servidor Rígido (API Síncrona):** Integración de la fachada de red con Axum. El modelo se aloja como estado compartido concurrente (`std::sync::Arc`) y responde peticiones HTTP POST.
3. **Épica 3: El Servidor Fluido (API en Streaming):** Integración de canales asíncronos (`tokio::sync::mpsc`) para emitir la respuesta en tiempo real a través de flujos SSE.
4. **Épica 4: El Binario Autónomo (Compilación Estática):** Aplicación de LTO y compilación `musl` para generar el ejecutable portable final.

## 🛠️ Requisitos Previos

- **Rust Toolchain:** Versión estable más reciente (`rustup update`).
- **Archivos del Modelo (Ej: Microsoft Phi-3 o equivalente < 3B Parámetros):**
  - Archivo de vocabulario: `tokenizer.json`
  - Archivo de hiperparámetros: `config.json`
  - Pesos del modelo: `model.safetensors`

## 🚀 Progreso Actual e Instrucciones (Fase 1)

**Hito Actual:**
Se ha configurado la suite completa de dependencias matemáticas en el `Cargo.toml` (`candle-core`, `candle-nn`, `candle-transformers`, `candle-datasets`, `hf-hub`, `tokenizers`) y se ha validado la ejecución básica del motor de tensores (soporte de fallback a CPU configurado).

Para probar el entorno de desarrollo:

```Bash
# Actualizar el compilador a su versión estable
rustup update

# Descargar dependencias, compilar y ejecutar prueba matemática básica
cargo run
```
