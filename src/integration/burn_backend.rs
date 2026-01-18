//! Burn inference backend for object detection.
//!
//! This module provides a `BurnDetector` that implements `DetectionSource`
//! for running object detection models built with the Burn framework.
//!
//! # Example
//!
//! ```ignore
//! use bytetrack_rs::burn_backend::{BurnDetector, BurnModel};
//! use burn::backend::NdArray;
//!
//! // Implement BurnModel for your detection model
//! struct MyYoloModel { /* ... */ }
//!
//! impl BurnModel<NdArray> for MyYoloModel {
//!     fn forward(&self, input: burn::tensor::Tensor<NdArray, 4>) -> Vec<RawDetection> {
//!         // Run inference
//!     }
//! }
//!
//! let model = MyYoloModel::load("model.bin");
//! let detector = BurnDetector::new(model);
//! ```

use super::{DetectionBuilder, DetectionSource};
use crate::tracker::Detection;
use burn::prelude::*;
use burn::tensor::Tensor;

/// Error type for Burn detection failures.
#[derive(Debug, Clone)]
pub enum BurnDetectorError {
    /// Input image has invalid dimensions.
    InvalidInputDimensions {
        expected: (u32, u32, u32),
        got: (u32, u32, u32),
    },
    /// Preprocessing failed.
    PreprocessingError(String),
    /// Model inference failed.
    InferenceError(String),
    /// Postprocessing failed.
    PostprocessingError(String),
}

impl std::fmt::Display for BurnDetectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInputDimensions { expected, got } => {
                write!(
                    f,
                    "Invalid input dimensions: expected {:?}, got {:?}",
                    expected, got
                )
            }
            Self::PreprocessingError(msg) => write!(f, "Preprocessing error: {}", msg),
            Self::InferenceError(msg) => write!(f, "Inference error: {}", msg),
            Self::PostprocessingError(msg) => write!(f, "Postprocessing error: {}", msg),
        }
    }
}

impl std::error::Error for BurnDetectorError {}

/// Raw detection output from the model before NMS.
#[derive(Debug, Clone)]
pub struct RawDetection {
    /// Bounding box: [x1, y1, x2, y2] or [cx, cy, w, h] depending on model
    pub bbox: [f32; 4],
    /// Confidence score
    pub score: f32,
    /// Class ID (optional, for multi-class detection)
    pub class_id: Option<usize>,
}

/// Trait for Burn-based detection models.
///
/// Implement this trait for your specific model architecture.
pub trait BurnModel<B: Backend>: Send + Sync {
    /// Run forward pass on the input tensor.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch, channels, height, width]
    ///
    /// # Returns
    /// Vector of raw detections before NMS filtering.
    fn forward(&self, input: Tensor<B, 4>) -> Vec<RawDetection>;

    /// Get the expected input size (channels, height, width).
    fn input_size(&self) -> (u32, u32, u32) {
        (3, 640, 640) // Default YOLO input size
    }

    /// Whether bbox output is in XYWH format (vs TLBR).
    fn bbox_is_xywh(&self) -> bool {
        true // Most YOLO variants use XYWH
    }
}

/// Burn-based object detector implementing `DetectionSource`.
pub struct BurnDetector<B: Backend, M: BurnModel<B>> {
    model: M,
    device: B::Device,
    conf_threshold: f32,
}

impl<B: Backend, M: BurnModel<B>> BurnDetector<B, M> {
    /// Create a new Burn detector with the given model and device.
    pub fn new(model: M, device: B::Device) -> Self {
        Self {
            model,
            device,
            conf_threshold: 0.25,
        }
    }

    /// Set the confidence threshold for filtering detections.
    pub fn with_conf_threshold(mut self, threshold: f32) -> Self {
        self.conf_threshold = threshold;
        self
    }

    /// Preprocess raw image bytes to a Burn tensor.
    ///
    /// Override this method for custom preprocessing.
    pub fn preprocess(
        &self,
        input: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Tensor<B, 4>, BurnDetectorError> {
        let (channels, target_h, target_w) = self.model.input_size();
        let expected_len = (width * height * channels) as usize;

        if input.len() != expected_len {
            return Err(BurnDetectorError::InvalidInputDimensions {
                expected: (channels, height, width),
                got: (channels, height, input.len() as u32 / (height * channels)),
            });
        }

        // Convert u8 to f32 and normalize to [0, 1]
        let data: Vec<f32> = input.iter().map(|&x| x as f32 / 255.0).collect();

        // Create tensor [C, H, W] then reshape to [1, C, H, W]
        let tensor = Tensor::<B, 1>::from_floats(data.as_slice(), &self.device).reshape([
            1,
            channels as usize,
            height as usize,
            width as usize,
        ]);

        // Resize if needed (simplified - real impl would use interpolation)
        if height != target_h || width != target_w {
            // For now, we expect input to match model size
            // Real implementation would resize here
            return Err(BurnDetectorError::PreprocessingError(format!(
                "Input size {}x{} doesn't match model size {}x{}. Resize not implemented.",
                width, height, target_w, target_h
            )));
        }

        Ok(tensor)
    }

    /// Convert raw model outputs to Detection objects.
    fn postprocess(&self, raw_detections: Vec<RawDetection>) -> Vec<Detection> {
        raw_detections
            .into_iter()
            .filter(|d| d.score >= self.conf_threshold)
            .map(|d| {
                let builder = DetectionBuilder::new().score(d.score);
                if self.model.bbox_is_xywh() {
                    builder
                        .xywh(d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3])
                        .build()
                } else {
                    builder
                        .tlbr(d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3])
                        .build()
                }
            })
            .collect()
    }
}

impl<B: Backend, M: BurnModel<B>> DetectionSource for BurnDetector<B, M> {
    type Error = BurnDetectorError;

    fn detect(
        &mut self,
        input: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<Detection>, Self::Error> {
        let tensor = self.preprocess(input, width, height)?;
        let raw_detections = self.model.forward(tensor);
        Ok(self.postprocess(raw_detections))
    }
}
