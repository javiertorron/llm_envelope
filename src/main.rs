use candle_core::{DType, Device, Tensor};

fn main() {
    // Queremos crear un Tensor muy simple en la CPU (una matriz llena de ceros) de tamaño 3x3.
    let device = Device::Cpu;

    // Tensor::zeros requiere 3 argumentos en las versiones recientes de candle: (shape, DType, &device)
    let tensor = Tensor::zeros((3, 3), DType::F32, &device).unwrap();
    println!("{:?}", tensor);
}
