use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::thread_rng;
use rand::Rng;

#[derive(Clone)]
pub enum Activation {
    ReLU,
    LeakyReLU(f32),
    Sigmoid,
    Tanh,
}

impl Activation {
    // Forward pass for activation functions
    fn forward(&self, x: &mut Array1<f32>) {
        match self {
            Activation::ReLU => x.mapv_inplace(|a| a.max(0.0)),
            Activation::LeakyReLU(alpha) => x.mapv_inplace(|a| if a > 0.0 { a } else { a * alpha }),
            Activation::Sigmoid => x.mapv_inplace(|a| 1.0 / (1.0 + (-a).exp())),
            Activation::Tanh => x.mapv_inplace(|a| a.tanh()),
        }
    }
 // Backward pass for activation functions
    fn backward(&self, x: &Array1<f32>, grad: &mut Array1<f32>) {
        match self {
            Activation::ReLU => grad.zip_mut_with(x, |g, &x| *g *= if x > 0.0 { 1.0 } else { 0.0 }),
            Activation::LeakyReLU(alpha) => grad.zip_mut_with(x, |g, &x| *g *= if x > 0.0 { 1.0 } else { *alpha }),
            Activation::Sigmoid => grad.zip_mut_with(x, |g, &x| *g *= x * (1.0 - x)),
            Activation::Tanh => grad.zip_mut_with(x, |g, &x| *g *= 1.0 - x.powi(2)),
        }
    }
}

pub struct LayerNorm {
    gamma: Array1<f32>,
    beta: Array1<f32>,
    eps: f32,
}

impl LayerNorm {
    pub fn new(size: usize, eps: f32) -> Self {
        LayerNorm {
            gamma: Array1::ones(size),
            beta: Array1::zeros(size),
            eps,
        }
    }

    pub fn forward(&self, x: &mut Array1<f32>) {
        let mean = x.mean().unwrap();
        let var = x.var(0.0);
        *x = (&*x - mean) / (var + self.eps).sqrt() * &self.gamma + &self.beta;
    }

    pub fn backward(&self, x: &Array1<f32>, grad: &mut Array1<f32>) -> (Array1<f32>, Array1<f32>) {
        let mean = x.mean().unwrap();
        let var = x.var(0.0);
        let std = (var + self.eps).sqrt();
        let n = x.len() as f32;

        let dx_norm = grad * &self.gamma;
        let dvar = (-0.5 * &dx_norm * (x - mean) / (var + self.eps).powf(1.5)).sum();
        let dmean = (-&dx_norm / std).sum() - 2.0 * dvar * (x - mean).sum() / n;

        let dx = &dx_norm / std + dvar * 2.0 * (x - mean) / n + dmean / n;
        let dgamma = (grad * ((x - mean) / std)).to_owned();
        let dbeta = grad.to_owned();

        (dgamma, dbeta)
    }
}

pub struct Layer {
    weights: Array2<f32>,
    biases: Array1<f32>,
    activation: Activation,
    layer_norm: Option<LayerNorm>,
    dropout_rate: f32,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, activation: Activation, use_layer_norm: bool, dropout_rate: f32) -> Self {
        let mut rng = thread_rng();
        let weights = Array2::random_using((output_size, input_size), Uniform::new(-0.08, 0.08), &mut rng);
        let biases = Array1::zeros(output_size);
        let layer_norm = if use_layer_norm { Some(LayerNorm::new(output_size, 1e-5)) } else { None };

        Layer { weights, biases, activation, layer_norm, dropout_rate }
    }

    pub fn forward(&self, input: &ArrayView1<f32>, training: bool) -> Array1<f32> {
        let mut output = self.weights.dot(input) + &self.biases;
        self.activation.forward(&mut output);
        if let Some(ln) = &self.layer_norm {
            ln.forward(&mut output);
        }
        if training && self.dropout_rate > 0.0 {
            let mask = Array1::random(output.len(), Uniform::new(0.0, 1.0))
                .map(|&x| if x > self.dropout_rate { 1.0 } else { 0.0 }) / (1.0 - self.dropout_rate);
            output *= &mask;
        }
        output
    }

    pub fn backward(&self, grad_output: &mut Array1<f32>, input: &ArrayView1<f32>) -> (Array2<f32>, Array1<f32>, Array1<f32>, Option<(Array1<f32>, Array1<f32>)>) {
        let mut ln_grads = None;
    
        if let Some(ln) = &self.layer_norm {
            let (dgamma, dbeta) = ln.backward(grad_output, grad_output);
            ln_grads = Some((dgamma, dbeta));
        }
    
        self.activation.backward(grad_output, grad_output);
    
        let grad_weights = grad_output.outer(input);
        let grad_biases = grad_output.to_owned();
        let grad_input = self.weights.t().dot(grad_output);
    
        (grad_weights, grad_biases, grad_input, ln_grads)
    }
}

pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(layer_specs: Vec<(usize, Activation, bool, f32)>) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_specs.len() - 1 {
            let (input_size, _, _, _) = layer_specs[i];
            let (output_size, activation, use_layer_norm, dropout_rate) = layer_specs[i + 1];
            layers.push(Layer::new(input_size, output_size, activation, use_layer_norm, dropout_rate));
        }
        NeuralNetwork { layers }
    }

    pub fn forward(&self, input: &ArrayView1<f32>, training: bool) -> Array1<f32> {
        let mut output = input.to_owned();
        for layer in &self.layers {
            output = layer.forward(&output.view(), training);
        }
        output
    }

    pub fn backward(&self, mut grad_output: Array1<f32>, inputs: &[ArrayView1<f32>]) -> Vec<(Array2<f32>, Array1<f32>, Option<(Array1<f32>, Array1<f32>)>)> {
        let mut grads = Vec::new();
        let mut grad_input = grad_output;
        for (layer, input) in self.layers.iter().zip(inputs.iter()).rev() {
            let (grad_weights, grad_biases, new_grad_input, ln_grads) = layer.backward(&mut grad_input, input);
            grads.push((grad_weights, grad_biases, ln_grads));
            grad_input = new_grad_input;
        }
        grads.reverse();
        grads
    }
}
