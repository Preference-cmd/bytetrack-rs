/// Bounding box representation with format conversion utilities.
///
/// Supports three common bounding box formats:
/// - TLWH: Top-Left X, Top-Left Y, Width, Height
/// - TLBR: Top-Left X, Top-Left Y, Bottom-Right X, Bottom-Right Y
/// - XYAH: Center X, Center Y, Aspect Ratio (w/h), Height
#[derive(Debug, Clone, Copy, Default)]
pub struct Rect {
    /// Top-left x coordinate
    pub x: f32,
    /// Top-left y coordinate
    pub y: f32,
    /// Width of the bounding box
    pub width: f32,
    /// Height of the bounding box
    pub height: f32,
}

impl Rect {
    /// Create a new Rect from top-left coordinates and dimensions (TLWH format).
    #[inline]
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Create a Rect from TLBR format (top-left x, top-left y, bottom-right x, bottom-right y).
    #[inline]
    pub fn from_tlbr(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self {
            x: x1,
            y: y1,
            width: x2 - x1,
            height: y2 - y1,
        }
    }

    /// Create a Rect from XYAH format (center x, center y, aspect ratio, height).
    #[inline]
    pub fn from_xyah(cx: f32, cy: f32, aspect_ratio: f32, height: f32) -> Self {
        let width = aspect_ratio * height;
        Self {
            x: cx - width / 2.0,
            y: cy - height / 2.0,
            width,
            height,
        }
    }

    /// Convert to TLBR format: (x1, y1, x2, y2).
    #[inline]
    pub fn to_tlbr(&self) -> [f32; 4] {
        [self.x, self.y, self.x + self.width, self.y + self.height]
    }

    /// Convert to TLWH format: (x, y, width, height).
    #[inline]
    pub fn to_tlwh(&self) -> [f32; 4] {
        [self.x, self.y, self.width, self.height]
    }

    /// Convert to XYAH format: (center_x, center_y, aspect_ratio, height).
    #[inline]
    pub fn to_xyah(&self) -> [f32; 4] {
        let cx = self.x + self.width / 2.0;
        let cy = self.y + self.height / 2.0;
        let aspect_ratio = if self.height > 0.0 {
            self.width / self.height
        } else {
            0.0
        };
        [cx, cy, aspect_ratio, self.height]
    }

    /// Get the center point of the bounding box.
    #[inline]
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    /// Get the area of the bounding box.
    #[inline]
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Calculate Intersection over Union (IoU) with another bounding box.
    pub fn iou(&self, other: &Rect) -> f32 {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);

        let inter_width = (x2 - x1).max(0.0);
        let inter_height = (y2 - y1).max(0.0);
        let inter_area = inter_width * inter_height;

        let union_area = self.area() + other.area() - inter_area;

        if union_area > 0.0 {
            inter_area / union_area
        } else {
            0.0
        }
    }
}

use ndarray::Array2;

/// Calculate IoU matrix between two sets of bounding boxes.
///
/// Returns a matrix of shape (M, N) where M is the length of `boxes_a`
/// and N is the length of `boxes_b`.
pub fn iou_batch(boxes_a: &[Rect], boxes_b: &[Rect]) -> Array2<f32> {
    let mut dists = Array2::zeros((boxes_a.len(), boxes_b.len()));
    for (i, a) in boxes_a.iter().enumerate() {
        for (j, b) in boxes_b.iter().enumerate() {
            dists[[i, j]] = a.iou(b);
        }
    }
    dists
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rect_conversions() {
        let rect = Rect::new(10.0, 20.0, 30.0, 40.0);

        // TLWH
        assert_eq!(rect.to_tlwh(), [10.0, 20.0, 30.0, 40.0]);

        // TLBR
        assert_eq!(rect.to_tlbr(), [10.0, 20.0, 40.0, 60.0]);

        // XYAH
        let xyah = rect.to_xyah();
        assert_eq!(xyah[0], 25.0); // cx
        assert_eq!(xyah[1], 40.0); // cy
        assert!((xyah[2] - 0.75).abs() < 1e-6); // aspect ratio = 30/40
        assert_eq!(xyah[3], 40.0); // height
    }

    #[test]
    fn test_from_tlbr() {
        let rect = Rect::from_tlbr(10.0, 20.0, 40.0, 60.0);
        assert_eq!(rect.to_tlwh(), [10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_from_xyah() {
        let rect = Rect::from_xyah(25.0, 40.0, 0.75, 40.0);
        assert!((rect.x - 10.0).abs() < 1e-6);
        assert!((rect.y - 20.0).abs() < 1e-6);
        assert!((rect.width - 30.0).abs() < 1e-6);
        assert!((rect.height - 40.0).abs() < 1e-6);
    }

    #[test]
    fn test_iou() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        let b = Rect::new(5.0, 5.0, 10.0, 10.0);

        // Intersection: 5x5 = 25
        // Union: 100 + 100 - 25 = 175
        // IoU: 25/175 ≈ 0.1428
        let iou = a.iou(&b);
        assert!((iou - 25.0 / 175.0).abs() < 1e-6);
    }

    #[test]
    fn test_iou_no_overlap() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        let b = Rect::new(20.0, 20.0, 10.0, 10.0);
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn test_iou_same_box() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        assert!((a.iou(&a) - 1.0).abs() < 1e-6);
    }
}
