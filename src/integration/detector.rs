//! Trait for object detection inference backends.

use crate::tracker::Detection;

/// Trait for object detection inference backends.
///
/// Implement this trait to connect any detection model to ByteTrack.
///
/// # Example
///
/// ```ignore
/// use bytetrack_rs::{DetectionSource, Detection};
///
/// struct MyDetector {
///     // Your model here
/// }
///
/// impl DetectionSource for MyDetector {
///     type Error = std::io::Error;
///
///     fn detect(&mut self, input: &[u8], width: u32, height: u32) -> Result<Vec<Detection>, Self::Error> {
///         // Run inference and return detections
///         Ok(vec![])
///     }
/// }
/// ```
pub trait DetectionSource {
    /// Error type for detection failures.
    type Error;

    /// Run inference on raw image data and return detections.
    ///
    /// # Arguments
    /// * `input` - Raw image bytes (format depends on implementation)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    ///
    /// # Returns
    /// A vector of `Detection` objects, or an error.
    fn detect(
        &mut self,
        input: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<Detection>, Self::Error>;
}

/// Helper trait for converting model-specific outputs to `Detection`.
///
/// Implement this for your model's output format to enable easy conversion.
pub trait IntoDetections {
    /// Convert the output into a vector of detections.
    fn into_detections(self) -> Vec<Detection>;
}

impl IntoDetections for Vec<Detection> {
    fn into_detections(self) -> Vec<Detection> {
        self
    }
}
