use candle_core::{Device, Tensor};

fn main() {
    // Creamos un Tensor en la CPU de tamaño 3x3 lleno de números aleatorios.
    // Usamos una distribución normal con media 0.0 y desviación estándar 1.0
    let device = Device::Cpu;

    let tensor = Tensor::randn(0f32, 1f32, (3, 3), &device).unwrap();
    println!("{:?}", tensor);
}
