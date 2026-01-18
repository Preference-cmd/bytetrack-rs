//! Matching utilities for multi-object tracking.

use crate::tracker::rect::Rect;
use ndarray::Array2;

/// Detection input for the tracker.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Bounding box in TLBR format (x1, y1, x2, y2)
    pub bbox: Rect,
    /// Detection confidence score
    pub score: f32,
}

impl Detection {
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32, score: f32) -> Self {
        Self {
            bbox: Rect::from_tlbr(x1, y1, x2, y2),
            score,
        }
    }

    pub fn from_rect(bbox: Rect, score: f32) -> Self {
        Self { bbox, score }
    }
}

/// Compute IoU distance matrix between tracks and detections.
pub fn iou_distance(track_boxes: &[Rect], det_boxes: &[Rect]) -> Array2<f32> {
    let mut dists = Array2::zeros((track_boxes.len(), det_boxes.len()));
    for (i, t) in track_boxes.iter().enumerate() {
        for (j, d) in det_boxes.iter().enumerate() {
            dists[[i, j]] = 1.0 - t.iou(d);
        }
    }
    dists
}

#[derive(Debug, Clone)]
pub struct AssignmentResult {
    pub matches: Vec<(usize, usize)>,
    pub unmatched_tracks: Vec<usize>,
    pub unmatched_detections: Vec<usize>,
}

pub fn linear_assignment(cost_matrix: &Array2<f32>, thresh: f32) -> AssignmentResult {
    let (num_rows, num_cols) = cost_matrix.dim();

    if num_rows == 0 {
        return AssignmentResult {
            matches: vec![],
            unmatched_tracks: vec![],
            unmatched_detections: (0..num_cols).collect(),
        };
    }

    if num_cols == 0 {
        return AssignmentResult {
            matches: vec![],
            unmatched_tracks: (0..num_rows).collect(),
            unmatched_detections: vec![],
        };
    }

    let size = num_rows.max(num_cols);
    let mut padded = Array2::<f64>::from_elem((size, size), 1e6);

    for i in 0..num_rows {
        for j in 0..num_cols {
            padded[[i, j]] = cost_matrix[[i, j]] as f64;
        }
    }

    let result = lapjv::lapjv(&padded);
    let mut matches = vec![];
    let mut unmatched_tracks = vec![];
    let mut unmatched_detections_mask: Vec<bool> = vec![true; num_cols];

    match result {
        Ok((row_to_col, _)) => {
            for (row_idx, &col_idx) in row_to_col.iter().enumerate() {
                if row_idx >= num_rows {
                    continue;
                }
                if col_idx >= num_cols {
                    unmatched_tracks.push(row_idx);
                } else if cost_matrix[[row_idx, col_idx]] <= thresh {
                    matches.push((row_idx, col_idx));
                    unmatched_detections_mask[col_idx] = false;
                } else {
                    unmatched_tracks.push(row_idx);
                }
            }
        }
        Err(_) => {
            unmatched_tracks = (0..num_rows).collect();
        }
    }

    let unmatched_detections: Vec<usize> = unmatched_detections_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &u)| if u { Some(i) } else { None })
        .collect();

    AssignmentResult {
        matches,
        unmatched_tracks,
        unmatched_detections,
    }
}

pub fn fuse_score(cost_matrix: &mut Array2<f32>, detections: &[Detection]) {
    let (rows, cols) = cost_matrix.dim();
    for i in 0..rows {
        for j in 0..cols {
            let iou_sim = 1.0 - cost_matrix[[i, j]];
            let fused_sim = iou_sim * detections[j].score;
            cost_matrix[[i, j]] = 1.0 - fused_sim;
        }
    }
}
