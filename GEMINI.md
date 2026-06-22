# 📄 Especificación Funcional y Técnica: Candle LLM Inference Envelope

Este documento establece la especificación completa del sistema, los modelos de datos y los requisitos técnicos para la construcción del ejecutable autónomo de inferencia. Sirve como la "fuente de la verdad" para el desarrollo del proyecto.

---

## 🎯 1. Objetivo del Sistema (Visión General)

El propósito de este proyecto es construir un **Envelope (envoltorio)** en forma de un único archivo ejecutable binario y portable, desarrollado en Rust. Este binario debe ser capaz de:

1. Cargar un Modelo de Lenguaje Grande (LLM) en formato `safetensors` desde el disco local.
2. Levantar un servidor HTTP nativo de alto rendimiento.
3. Exponer una API compatible con el estándar de la industria para recibir prompts y devolver respuestas generadas por el modelo en tiempo real (Streaming).

---

## ⚙️ 2. Especificación Funcional

### 2.1. Comportamiento en el Arranque

Al ejecutar el binario desde la línea de comandos, el sistema debe realizar las siguientes acciones en secuencia:

* **Validación de Entorno:** Comprobar la existencia de los artefactos del modelo en la ruta especificada (`config.json`, `tokenizer.json`, `*.safetensors`).
* **Carga Inicial (Warm-up):** Parsear la configuración estructural e inicializar los pesos del modelo en la memoria del sistema. El servidor no debe abrir los puertos de red hasta que el modelo esté completamente operativo en memoria.
* **Inicialización de Red:** Levantar el servicio HTTP en el puerto designado (por defecto `8080`) quedando a la escucha de peticiones.

### 2.2. Interfaz de Usuario y Consola

El binario debe actuar de forma silenciosa e informativa, reportando por la salida estándar (`stdout`) únicamente los hitos críticos:

* Estado de la carga del modelo (con marcas de tiempo).
* Dirección IP y puerto donde escucha el servidor.
* Registro básico de peticiones procesadas (método, ruta y código de estado HTTP).

---

## 🛠️ 3. Especificación Técnica y Arquitectura

### 3.1. Stack Tecnológico Nucleares

* **Lenguaje:** Rust (Edición 2021, Toolchain Stable).
* **Motor Matemático e Inferencia:** `candle-core` y `candle-transformers` (Hugging Face).
* **Servidor HTTP:** `axum` sobre el runtime asíncrono `tokio`.
* **Serialización de Datos:** `serde` y `serde_json`.
* **Tokenización:** `tokenizers` (Implementación nativa en Rust de Hugging Face).

### 3.2. Gestión de Memoria y Rendimiento

* **Mapeo de Memoria (mmap):** Los pesos del modelo se mapean de forma virtual usando la API del sistema operativo (`mmap`). Se prohíbe la lectura tradicional en búfer de bytes de archivos `.safetensors` para evitar picos de consumo de RAM.
* **Concurrencia Segura:** El modelo e hilos de ejecución de inferencia deben encapsularse en un puntero atómico compartido (`std::sync::Arc`). Las peticiones entrantes se encolan o procesan garantizando el acceso seguro al estado del modelo.
* **Aislamiento de Hardware:** La inferencia se ejecutará inicialmente sobre la CPU del sistema anfitrión, garantizando compatibilidad universal sin dependencias de drivers gráficos externos.

---

## 📊 4. Modelos de Datos y Especificación de la API

El servidor expondrá un único endpoint operativo que imita el estándar de OpenAI para facilitar la integración con herramientas existentes.

### Endpoint: `POST /v1/chat/completions`

#### 📥 Estructura de la Petición (Request JSON)

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

#### Estructura de la Respuesta en Streaming (Server-Sent Events - SSE)

La respuesta se transmitirá como un flujo continuo de eventos de tipo text/event-stream. Cada fragmento de datos emitido por el servidor tendrá la siguiente estructura:

```json
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"phi3","choices":[{"index":0,"delta":{"content":"palabra"},"finish_reason":null}]}
```

Al finalizar la generación, el servidor enviará un token de cierre estándar:

```json
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"phi3","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

## 5. Criterios de Portabilidad (El Ejecutable Final)

El producto final del desarrollo debe cumplir con las siguientes restricciones de empaquetado:

Cero Dependencias Dinámicas: En sistemas Linux, el binario debe compilarse contra el target x86_64-unknown-linux-musl para incrustar la librería C estáticamente.

Tamaño Optimizado: Se debe aplicar Link-Time Optimization (lto = true) en el perfil de producción para reducir el tamaño muerto del código binario generado por las librerías matemáticas.

Autonomía: El archivo ejecutable resultante debe ser capaz de arrancar en una instalación limpia del sistema operativo objetivo con el único requisito de tener acceso de lectura a la carpeta de los pesos del modelo.
