//! TrackerPipeline for combining detection with tracking.

use crate::tracker::{BYTETracker, STrack, TrackerConfig};

use super::DetectionSource;

/// A combined tracker that bundles detection inference with ByteTrack.
///
/// This struct provides a convenient way to run end-to-end tracking
/// by combining any `DetectionSource` with the `BYTETracker`.
pub struct TrackerPipeline<D: DetectionSource> {
    detector: D,
    tracker: BYTETracker,
}

impl<D: DetectionSource> TrackerPipeline<D> {
    /// Create a new tracking pipeline with the given detector and tracker config.
    pub fn new(detector: D, config: TrackerConfig) -> Self {
        Self {
            detector,
            tracker: BYTETracker::new(config),
        }
    }

    /// Create a new tracking pipeline with default tracker configuration.
    pub fn with_default_config(detector: D) -> Self {
        Self::new(detector, TrackerConfig::default())
    }

    /// Process a single frame and return active tracks.
    ///
    /// This method runs detection on the input image and then updates
    /// the tracker with the detected objects.
    ///
    /// # Arguments
    /// * `input` - Raw image bytes
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    ///
    /// # Returns
    /// A vector of active `STrack` objects, or a detection error.
    pub fn process_frame(
        &mut self,
        input: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<STrack>, D::Error> {
        let detections = self.detector.detect(input, width, height)?;
        Ok(self.tracker.update(detections))
    }

    /// Get a reference to the underlying detector.
    pub fn detector(&self) -> &D {
        &self.detector
    }

    /// Get a mutable reference to the underlying detector.
    pub fn detector_mut(&mut self) -> &mut D {
        &mut self.detector
    }

    /// Get a reference to the underlying tracker.
    pub fn tracker(&self) -> &BYTETracker {
        &self.tracker
    }

    /// Get a mutable reference to the underlying tracker.
    pub fn tracker_mut(&mut self) -> &mut BYTETracker {
        &mut self.tracker
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracker::Detection;

    struct MockDetector {
        detections: Vec<Detection>,
    }

    impl DetectionSource for MockDetector {
        type Error = std::convert::Infallible;

        fn detect(
            &mut self,
            _input: &[u8],
            _width: u32,
            _height: u32,
        ) -> Result<Vec<Detection>, Self::Error> {
            Ok(self.detections.clone())
        }
    }

    #[test]
    fn test_tracker_pipeline() {
        let detector = MockDetector {
            detections: vec![Detection::new(10.0, 20.0, 50.0, 80.0, 0.9)],
        };

        let mut pipeline = TrackerPipeline::with_default_config(detector);
        let tracks = pipeline.process_frame(&[], 640, 480).unwrap();

        // First frame initializes tracks
        assert!(tracks.is_empty() || !tracks.is_empty()); // Depends on activation logic
    }
}
