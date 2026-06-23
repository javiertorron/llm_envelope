# Candle LLM Inference Envelope

Este proyecto es un motor de inferencia nativo en Rust para modelos grandes de lenguaje (LLMs) como Gemma, diseñado para ser altamente portable, rápido y capaz de ejecutarse en distintos tipos de hardware (CPU, NVIDIA GPU o Apple Metal).

## Requisitos de Hardware y Dependencias

Dependiendo del hardware que quieras utilizar para la aceleración gráfica (`device_type` en tu configuración), necesitarás tener instaladas distintas librerías en tu sistema operativo antes de poder compilar el proyecto.

### 1. Inferencia en CPU (Modo por defecto)

El modo CPU garantiza la máxima portabilidad. Cuenta con un conversor dinámico `BF16` -> `F32` que permite a la CPU calcular tensores sin colapsar por incompatibilidades de tipo.

- **Sistemas Soportados**: Linux, macOS, Windows.
- **Dependencias**: Ninguna (Cero dependencias dinámicas).
- **Compilación**:

  ```bash
  cargo build --release
  ```

### 2. Inferencia en GPU NVIDIA (CUDA)

Para utilizar la VRAM y los núcleos CUDA de tu tarjeta gráfica NVIDIA, necesitas instalar el **NVIDIA CUDA Toolkit** en tu sistema. Esto proporcionará el compilador `nvcc` necesario para que Rust enlace con los drivers de tu tarjeta.

- **Sistemas Soportados**: Linux, Windows.
- **Dependencias por Sistema Operativo**:
  - **Ubuntu / Pop!_OS / Debian**:

    ```bash
    sudo apt update && sudo apt install nvidia-cuda-toolkit -y
    ```

  - **Fedora / RHEL**:

    ```bash
    sudo dnf install cuda
    ```

  - **Arch Linux / Manjaro**:

    ```bash
    sudo pacman -S cuda
    ```

  - **Windows**: Descarga el instalador del [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) y añádelo al PATH.
- **Compilación**:

  ```bash
  cargo build --release --features cuda
  ```

### 3. Inferencia en Apple Silicon (Metal)

Para ordenadores Mac con chips M1, M2, M3 o M4. Aprovecha la memoria unificada del sistema operativo macOS.

- **Sistemas Soportados**: macOS.
- **Dependencias**: Ninguna adicional (Metal Framework viene preinstalado en macOS).
- **Compilación**:

  ```bash
  cargo build --release --features metal
  ```

---

## Ejecución y Fichero de Configuración

El ejecutable buscará en su misma ruta un archivo llamado `envelope_config.json` para determinar qué hardware inicializar.

Crea un archivo `envelope_config.json` en la raíz de la aplicación:

```json
{
  "device_type": "cuda"
}
```

*Los valores permitidos para `device_type` son: `"cpu"`, `"cuda"`, o `"metal"`.*

### Modos de Arranque

1. **Modo Consola Interactiva**: Si lanzas el programa de forma estándar, se abrirá un chat en la terminal.

   ```bash
   cargo run --release --features cuda -- --model /ruta/absoluta/a/los/pesos/del/modelo/
   ```

2. **Modo Servidor HTTP (OpenAI API)**: Si pasas el parámetro `--serve`, el sistema levantará un servidor asíncrono `Axum` en el puerto `8080` que responderá a peticiones HTTP mediante *Server-Sent Events* (SSE).

   ```bash
   cargo run --release --features cuda -- --model /ruta/absoluta/a/los/pesos/del/modelo/ --serve
   ```

> **Nota de Resiliencia:** Si configuras el servidor para usar `cuda` o `metal` pero el hardware falla al inicializarse o los drivers están corruptos, el motor atrapará el error y realizará un **fallback automático a la CPU**, garantizando que el servicio web nunca se caiga durante el arranque.
