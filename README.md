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
- **Binario Autónomo (Envelope):** Preparado para compilación estática completa (vía target `musl`), eliminando la necesidad de runtimes, contenedores o librerías del sistema anfitrión.

## 🗺️ Mapa de Ruta de Ingeniería (Épicas)

El desarrollo del proyecto está estrictamente acotado en 4 grandes bloques secuenciales:

1. **Épica 1: El Motor en Consola (Inferencia Local):** Construcción del núcleo matemático de Candle y el bucle autorregresivo. El modelo carga desde el disco duro y genera texto directamente en la terminal (CLI puro).
2. **Épica 2: El Servidor Rígido (API Síncrona):** Integración de la fachada de red con Axum. El modelo se aloja como estado compartido concurrente (`std::sync::Arc`) y responde peticiones HTTP POST con estructuras JSON estructuradas.
3. **Épica 3: El Servidor Fluido (API en Streaming):** Optimización de la experiencia de usuario (UX) mediante canales asíncronos (`tokio::sync::mpsc`) para emitir la respuesta en tiempo real a través de flujos SSE.
4. **Épica 4: El Binario Autónomo (Compilación Estática):** Aplicación de Link-Time Optimization (LTO) y enlazado estático para generar el ejecutable final portable a cualquier máquina anfitriona.

## 🛠️ Requisitos Previos

- **Rust Toolchain:** Versión estable más reciente (`rustup update`).
- **Archivos del Modelo (Elegido: Microsoft Phi-3 o equivalente < 3B Parámetros):**
  - Archivo de vocabulario: `tokenizer.json`
  - Archivo de hiperparámetros: `config.json`
  - Pesos del modelo: `model.safetensors` (Se recomienda un único archivo de pesos para el MVP).

## 📂 Estructura de Archivos Planificada

```text
.
├── Cargo.toml            # Manifiesto de dependencias (Candle, Axum, Tokio, Serde)
├── README.md             # Documentación técnica del proyecto
├── model/                # Directorio local aislado para los artefactos de la IA (Ignorado en Git)
│   ├── config.json
│   ├── tokenizer.json
│   └── model.safetensors
└── src/
    └── main.rs           # Código fuente principal (Orquestador y API)
```

## 🚀 Instrucciones de Desarrollo Inicial (Fase 1)

Para validar el entorno de desarrollo y realizar la prueba de humo del ecosistema matemático en CPU:

1. Clonar o inicializar este repositorio localmente.
2. Asegurar la última versión estable del compilador:

```Bash
rustup update
```

Ejecutar el proyecto en modo desarrollo para compilar las dependencias de Candle:

```Bash
cargo run
```

(Nota: La primera compilación puede tomar unos minutos debido al enlazado de rutinas numéricas de bajo nivel nativas de Candle).
