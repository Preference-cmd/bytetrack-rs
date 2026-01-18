//! Kalman filter for bounding box tracking using ndarray and a manual/nalgebra-based inverse.

use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct KalmanFilter {
    motion_mat: Array2<f64>,
    update_mat: Array2<f64>,
    std_weight_position: f64,
    std_weight_velocity: f64,
}

impl Default for KalmanFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanFilter {
    pub fn new() -> Self {
        let ndim = 4;
        let mut motion_mat = Array2::eye(2 * ndim);
        for i in 0..ndim {
            motion_mat[[i, ndim + i]] = 1.0;
        }

        let mut update_mat = Array2::zeros((ndim, 2 * ndim));
        for i in 0..ndim {
            update_mat[[i, i]] = 1.0;
        }

        Self {
            motion_mat,
            update_mat,
            std_weight_position: 1.0 / 20.0,
            std_weight_velocity: 1.0 / 160.0,
        }
    }

    pub fn initiate(&self, measurement: [f64; 4]) -> (Array1<f64>, Array2<f64>) {
        let mut mean = Array1::zeros(8);
        for i in 0..4 {
            mean[i] = measurement[i];
        }

        let h = measurement[3];
        let std = [
            2.0 * self.std_weight_position * h,
            2.0 * self.std_weight_position * h,
            1e-2,
            2.0 * self.std_weight_position * h,
            10.0 * self.std_weight_velocity * h,
            10.0 * self.std_weight_velocity * h,
            1e-5,
            10.0 * self.std_weight_velocity * h,
        ];

        let mut cov = Array2::zeros((8, 8));
        for i in 0..8 {
            cov[[i, i]] = std[i] * std[i];
        }

        (mean, cov)
    }

    pub fn predict(
        &self,
        mean: &Array1<f64>,
        covariance: &Array2<f64>,
    ) -> (Array1<f64>, Array2<f64>) {
        let h = mean[3];
        let std = [
            self.std_weight_position * h,
            self.std_weight_position * h,
            1e-2,
            self.std_weight_position * h,
            self.std_weight_velocity * h,
            self.std_weight_velocity * h,
            1e-5,
            self.std_weight_velocity * h,
        ];

        let mut motion_cov = Array2::zeros((8, 8));
        for i in 0..8 {
            motion_cov[[i, i]] = std[i] * std[i];
        }

        let new_mean = self.motion_mat.dot(mean);
        let new_covariance = self.motion_mat.dot(covariance).dot(&self.motion_mat.t()) + motion_cov;

        (new_mean, new_covariance)
    }

    pub fn project(
        &self,
        mean: &Array1<f64>,
        covariance: &Array2<f64>,
    ) -> (Array1<f64>, Array2<f64>) {
        let h = mean[3];
        let std = [
            self.std_weight_position * h,
            self.std_weight_position * h,
            1e-1,
            self.std_weight_position * h,
        ];

        let mut innovation_cov = Array2::zeros((4, 4));
        for i in 0..4 {
            innovation_cov[[i, i]] = std[i] * std[i];
        }

        let mean_proj = self.update_mat.dot(mean);
        let covariance_proj =
            self.update_mat.dot(covariance).dot(&self.update_mat.t()) + innovation_cov;

        (mean_proj, covariance_proj)
    }

    pub fn update(
        &self,
        mean: &Array1<f64>,
        covariance: &Array2<f64>,
        measurement: [f64; 4],
    ) -> (Array1<f64>, Array2<f64>) {
        let (projected_mean, projected_cov) = self.project(mean, covariance);

        let measurement_arr = Array1::from_vec(measurement.to_vec());
        let innovation = measurement_arr - projected_mean;

        // K = P * H^T * S^-1
        // Since H is [I 0], P * H^T is the first 4 columns of P (8x4).
        // S is projected_cov (4x4).

        // We use nalgebra internally for 4x4 inversion to avoid BLAS/LAPACK.
        let s_inv = self.invert_4x4(&projected_cov);

        let pht = covariance.dot(&self.update_mat.t()); // 8x4
        let kalman_gain = pht.dot(&s_inv); // 8x4

        let new_mean = mean + kalman_gain.dot(&innovation);
        let new_covariance = covariance - kalman_gain.dot(&projected_cov).dot(&kalman_gain.t());

        (new_mean, new_covariance)
    }

    /// Helper to invert a 4x4 matrix using nalgebra (pure Rust).
    fn invert_4x4(&self, m: &Array2<f64>) -> Array2<f64> {
        let mut nm = nalgebra::Matrix4::zeros();
        for i in 0..4 {
            for j in 0..4 {
                nm[(i, j)] = m[[i, j]];
            }
        }
        let inv = nm.try_inverse().expect("4x4 matrix inversion failed");
        let mut res = Array2::zeros((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                res[[i, j]] = inv[(i, j)];
            }
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initiate() {
        let kf = KalmanFilter::new();
        let (mean, _) = kf.initiate([100.0, 200.0, 0.5, 50.0]);
        assert_eq!(mean[0], 100.0);
    }
}
