#!/bin/bash

sudo apt update && sudo apt install cuda-toolkit-12-8 -y

BASHRC_FILE="$HOME/.bashrc"
CUDA_BIN="/usr/local/cuda-12.8/bin"
CUDA_LIB="/usr/local/cuda-12.8/lib64"

echo "🔧 Configurando rutas de CUDA 12.8 en $BASHRC_FILE..."

# 1. Limpiar las rutas de CUDA genéricas que añadimos en pruebas anteriores
sed -i '/export PATH=\/usr\/local\/cuda\/bin:\$PATH/d' "$BASHRC_FILE"
sed -i '/export LD_LIBRARY_PATH=\/usr\/local\/cuda\/lib64:\$LD_LIBRARY_PATH/d' "$BASHRC_FILE"

# 2. Añadir la ruta de binarios de CUDA 12.8 (si no existe ya)
if ! grep -q "$CUDA_BIN" "$BASHRC_FILE"; then
    echo "export PATH=$CUDA_BIN:\$PATH" >> "$BASHRC_FILE"
    echo "✅ Añadida ruta de binarios: $CUDA_BIN"
else
    echo "ℹ️ La ruta de binarios ya estaba configurada."
fi

# 3. Añadir la ruta de librerías de CUDA 12.8 (si no existe ya)
if ! grep -q "$CUDA_LIB" "$BASHRC_FILE"; then
    echo "export LD_LIBRARY_PATH=$CUDA_LIB:\$LD_LIBRARY_PATH" >> "$BASHRC_FILE"
    echo "✅ Añadida ruta de librerías: $CUDA_LIB"
else
    echo "ℹ️ La ruta de librerías ya estaba configurada."
fi

echo ""
echo "🎉 ¡Configuración completada!"
echo "👉 Por favor, ejecuta el siguiente comando para aplicar los cambios ahora mismo:"
echo "   source ~/.bashrc"
