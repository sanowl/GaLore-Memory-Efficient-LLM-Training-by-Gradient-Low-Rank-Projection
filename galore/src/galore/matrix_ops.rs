use ndarray::{Array2, ArrayView2, Axis};
use ndarray_linalg::{Eigh, SVD};
use std::sync::Arc;
use rayon::prelude::*;

pub struct GaLoreProjection {
    rank: usize,
    update_freq: usize,
    ema_decay: f32,
    step: usize,
    projections: Vec<(Arc<Array2<f32>>, Arc<Array2<f32>>)>,
}

impl GaLoreProjection {
    pub fn new(rank: usize, update_freq: usize, ema_decay: f32) -> Self {
        GaLoreProjection {
            rank,
            update_freq,
            ema_decay,
            step: 0,
            projections: Vec::new(),
        }
    }

    pub fn project_gradient(&mut self, gradients: Vec<ArrayView2<f32>>) -> Vec<Array2<f32>> {
        self.step += 1;

        if self.step % self.update_freq == 0 || self.projections.is_empty() {
            self.update_projections(&gradients);
        }

        gradients
            .par_iter()
            .zip(self.projections.par_iter())
            .map(|(grad, (p, q))| self.project(grad, p, q))
            .collect()
    }

    pub fn project_update(&self, updates: Vec<ArrayView2<f32>>) -> Vec<Array2<f32>> {
        updates
            .par_iter()
            .zip(self.projections.par_iter())
            .map(|(update, (p, q))| self.project_back(update, p, q))
            .collect()
    }

    fn update_projections(&mut self, gradients: &[ArrayView2<f32>]) {
        self.projections = gradients
            .par_iter()
            .map(|grad| {
                let (p, q) = self.compute_projection_matrices(grad);
                (Arc::new(p), Arc::new(q))
            })
            .collect();
    }

    fn compute_projection_matrices(&self, grad: &ArrayView2<f32>) -> (Array2<f32>, Array2<f32>) {
        let (m, n) = grad.dim();
        let (mut u, s, mut vt) = grad.svd(true, true).unwrap();

        u.slice_axis_inplace(Axis(1), ndarray::Slice::from(0..self.rank));
        vt.slice_axis_inplace(Axis(0), ndarray::Slice::from(0..self.rank));

        if let Some((p_old, q_old)) = self.projections.get(0) {
            let p = self.ema_update(&p_old, &u);
            let q = self.ema_update(&q_old, &vt.t());
            (p, q)
        } else {
            (u, vt.t().to_owned())
        }
    }

    fn project(&self, grad: &ArrayView2<f32>, p: &Array2<f32>, q: &Array2<f32>) -> Array2<f32> {
        p.t().dot(&grad.dot(q))
    }

    fn project_back(&self, update: &ArrayView2<f32>, p: &Array2<f32>, q: &Array2<f32>) -> Array2<f32> {
        p.dot(&update.dot(&q.t()))
    }

    fn ema_update(&self, old: &Array2<f32>, new: &Array2<f32>) -> Array2<f32> {
        old * self.ema_decay + new * (1.0 - self.ema_decay)
    }
}

pub struct GaLoreOptimizer<O: Optimizer> {
    base_optimizer: O,
    galore: GaLoreProjection,
}

impl<O: Optimizer> GaLoreOptimizer<O> {
    pub fn new(base_optimizer: O, rank: usize, update_freq: usize, ema_decay: f32) -> Self {
        GaLoreOptimizer {
            base_optimizer,
            galore: GaLoreProjection::new(rank, update_freq, ema_decay),
        }
    }

    pub fn step(&mut self, gradients: Vec<ArrayView2<f32>>) -> Vec<Array2<f32>> {
        let projected_grads = self.galore.project_gradient(gradients);
        let updates = self.base_optimizer.compute_updates(&projected_grads);
        self.galore.project_update(updates.iter().map(|u| u.view()).collect())
    }
}

pub trait Optimizer {
    fn compute_updates(&mut self, gradients: &[Array2<f32>]) -> Vec<Array2<f32>>;
}

// Example implementation of Adam optimizer
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    m: Vec<Array2<f32>>,
    v: Vec<Array2<f32>>,
    t: usize,
}

impl Adam {
    pub fn new(lr: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Adam {
            lr,
            beta1,
            beta2,
            epsilon,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }
}

impl Optimizer for Adam {
    fn compute_updates(&mut self, gradients: &[Array2<f32>]) -> Vec<Array2<f32>> {
        self.t += 1;
        if self.m.is_empty() {
            self.m = gradients.iter().map(|g| Array2::zeros(g.dim())).collect();
            self.v = gradients.iter().map(|g| Array2::zeros(g.dim())).collect();
        }

        gradients
            .iter()
            .zip(self.m.iter_mut())
            .zip(self.v.iter_mut())
            .map(|((g, m), v)| {
                *m = self.beta1 * &*m + (1.0 - self.beta1) * g;
                *v = self.beta2 * &*v + (1.0 - self.beta2) * g * g;

                let m_hat = m / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = v / (1.0 - self.beta2.powi(self.t as i32));

                -self.lr * &m_hat / (v_hat.map(|x| x.sqrt()) + self.epsilon)
            })
            .collect()
    }
}