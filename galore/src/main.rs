use ndarray::{Array2, ArrayView2};

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