//! Integration module for connecting object detection backends with ByteTrack.
//!
//! This module provides traits and utilities for integrating various inference
//! backends (Burn, ONNX Runtime, etc.) with the ByteTrack tracker.

mod builder;
mod detector;
mod pipeline;

pub use builder::DetectionBuilder;
pub use detector::{DetectionSource, IntoDetections};
pub use pipeline::TrackerPipeline;

#[cfg(feature = "burn-backend")]
mod burn_backend;

#[cfg(feature = "burn-backend")]
pub use burn_backend::{BurnDetector, BurnDetectorError, BurnModel, RawDetection};
