//! Builder for creating Detection objects from various input formats.

use crate::tracker::Detection;

/// Builder for creating `Detection` objects from various input formats.
#[derive(Debug, Clone, Default)]
pub struct DetectionBuilder {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    score: f32,
}

impl DetectionBuilder {
    /// Create a new detection builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set bounding box in TLBR format (x1, y1, x2, y2).
    pub fn tlbr(mut self, x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        self.x1 = x1;
        self.y1 = y1;
        self.x2 = x2;
        self.y2 = y2;
        self
    }

    /// Set bounding box in XYWH format (center_x, center_y, width, height).
    pub fn xywh(mut self, cx: f32, cy: f32, w: f32, h: f32) -> Self {
        self.x1 = cx - w / 2.0;
        self.y1 = cy - h / 2.0;
        self.x2 = cx + w / 2.0;
        self.y2 = cy + h / 2.0;
        self
    }

    /// Set bounding box in TLWH format (top, left, width, height).
    pub fn tlwh(mut self, t: f32, l: f32, w: f32, h: f32) -> Self {
        self.x1 = l;
        self.y1 = t;
        self.x2 = l + w;
        self.y2 = t + h;
        self
    }

    /// Set the confidence score.
    pub fn score(mut self, score: f32) -> Self {
        self.score = score;
        self
    }

    /// Build the final `Detection`.
    pub fn build(self) -> Detection {
        Detection::new(self.x1, self.y1, self.x2, self.y2, self.score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_builder() {
        let det = DetectionBuilder::new()
            .tlbr(10.0, 20.0, 50.0, 80.0)
            .score(0.95)
            .build();

        assert_eq!(det.score, 0.95);
    }
}
