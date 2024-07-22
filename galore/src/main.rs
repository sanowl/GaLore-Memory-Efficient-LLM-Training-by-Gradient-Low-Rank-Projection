use ndarray::{Array2, ArrayView2, s};
use ndarray_linalg::SVD;

fn svd_lowrank(matrix: &ArrayView2<f32>, rank: usize) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let (u_opt, s_opt, vt_opt) = matrix.svd(true, true).expect("SVD failed");

    let u = u_opt.unwrap().slice(s![.., ..rank]).to_owned();
    let s = Array2::from_diag(&s_opt.unwrap().slice(s![..rank]));
    let vt = vt_opt.unwrap().slice(s![..rank, ..]).to_owned();

    (u, s, vt)
}

fn project_gradient(gradient: &ArrayView2<f32>, p: &ArrayView2<f32>, q: &ArrayView2<f32>) -> Array2<f32> {
    p.t().dot(&gradient.dot(q))
}

fn main() {
    // Example usage
    let matrix = Array2::eye(5);
    let (u, s, vt) = svd_lowrank(&matrix.view(), 3);
    println!("U: {:?}", u);
    println!("S: {:?}", s);
    println!("V^T: {:?}", vt);

    let gradient = Array2::ones((5, 5));
    let p = Array2::eye(5);
    let q = Array2::eye(5);
    let projected_gradient = project_gradient(&gradient.view(), &p.view(), &q.view());
    println!("Projected gradient: {:?}", projected_gradient);
}
