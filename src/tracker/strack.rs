//! Single object track (STrack) for multi-object tracking.

use std::sync::atomic::{AtomicU64, Ordering};

use ndarray::{Array1, Array2};

use crate::tracker::kalman_filter::KalmanFilter;
use crate::tracker::rect::Rect;
use crate::tracker::track_state::TrackState;

/// Global track ID counter for unique ID generation.
static TRACK_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Reset the global track ID counter (useful for testing).
pub fn reset_track_id_counter() {
    TRACK_ID_COUNTER.store(0, Ordering::SeqCst);
}

/// Get the next unique track ID.
fn next_track_id() -> u64 {
    TRACK_ID_COUNTER.fetch_add(1, Ordering::SeqCst) + 1
}

/// Single object track.
#[derive(Debug, Clone)]
pub struct STrack {
    /// Unique track identifier
    pub track_id: u64,
    /// Current track state
    pub state: TrackState,
    /// Whether the track has been activated (confirmed)
    pub is_activated: bool,
    /// Detection confidence score
    pub score: f32,
    /// Current frame ID
    pub frame_id: u32,
    /// Frame ID when track was started
    pub start_frame: u32,
    /// Number of frames since track was last seen
    pub tracklet_len: u32,
    /// Kalman filter state mean (8-dim)
    pub mean: Option<Array1<f64>>,
    /// Kalman filter state covariance (8x8)
    pub covariance: Option<Array2<f64>>,
    /// Original detection bounding box (TLWH format)
    pub tlwh: Rect,
}

impl STrack {
    /// Create a new STrack from a detection.
    pub fn new(tlwh: Rect, score: f32) -> Self {
        Self {
            track_id: 0,
            state: TrackState::New,
            is_activated: false,
            score,
            frame_id: 0,
            start_frame: 0,
            tracklet_len: 0,
            mean: None,
            covariance: None,
            tlwh,
        }
    }

    /// Get the current bounding box in TLWH format.
    pub fn tlwh(&self) -> Rect {
        match &self.mean {
            Some(mean) => {
                let cx = mean[0] as f32;
                let cy = mean[1] as f32;
                let aspect = mean[2] as f32;
                let h = mean[3] as f32;
                Rect::from_xyah(cx, cy, aspect, h)
            }
            None => self.tlwh,
        }
    }

    pub fn rect(&self) -> Rect {
        self.tlwh()
    }

    pub fn end_frame(&self) -> u32 {
        self.frame_id
    }

    pub fn activate(&mut self, kalman_filter: &KalmanFilter, frame_id: u32) {
        self.track_id = next_track_id();

        let xyah = self.tlwh.to_xyah();
        let xyah_f64 = [
            xyah[0] as f64,
            xyah[1] as f64,
            xyah[2] as f64,
            xyah[3] as f64,
        ];
        let (mean, covariance) = kalman_filter.initiate(xyah_f64);

        self.mean = Some(mean);
        self.covariance = Some(covariance);
        self.tracklet_len = 0;
        self.state = TrackState::Tracked;

        if frame_id == 1 {
            self.is_activated = true;
        }

        self.frame_id = frame_id;
        self.start_frame = frame_id;
    }

    pub fn re_activate(
        &mut self,
        new_track: &STrack,
        kalman_filter: &KalmanFilter,
        frame_id: u32,
        new_id: bool,
    ) {
        let xyah = new_track.tlwh.to_xyah();
        let xyah_f64 = [
            xyah[0] as f64,
            xyah[1] as f64,
            xyah[2] as f64,
            xyah[3] as f64,
        ];

        if let (Some(mean), Some(cov)) = (&self.mean, &self.covariance) {
            let (new_mean, new_cov) = kalman_filter.update(mean, cov, xyah_f64);
            self.mean = Some(new_mean);
            self.covariance = Some(new_cov);
        }

        self.tracklet_len = 0;
        self.state = TrackState::Tracked;
        self.is_activated = true;
        self.frame_id = frame_id;
        self.score = new_track.score;

        if new_id {
            self.track_id = next_track_id();
        }
    }

    pub fn update(&mut self, new_track: &STrack, kalman_filter: &KalmanFilter, frame_id: u32) {
        self.frame_id = frame_id;
        self.tracklet_len += 1;

        let xyah = new_track.tlwh.to_xyah();
        let xyah_f64 = [
            xyah[0] as f64,
            xyah[1] as f64,
            xyah[2] as f64,
            xyah[3] as f64,
        ];

        if let (Some(mean), Some(cov)) = (&self.mean, &self.covariance) {
            let (new_mean, new_cov) = kalman_filter.update(mean, cov, xyah_f64);
            self.mean = Some(new_mean);
            self.covariance = Some(new_cov);
        }

        self.state = TrackState::Tracked;
        self.is_activated = true;
        self.score = new_track.score;
    }

    pub fn predict(&mut self, kalman_filter: &KalmanFilter) {
        if let (Some(mean), Some(cov)) = (&self.mean, &self.covariance) {
            let mut mean_to_predict = mean.clone();
            if self.state != TrackState::Tracked {
                mean_to_predict[7] = 0.0;
            }
            let (new_mean, new_cov) = kalman_filter.predict(&mean_to_predict, cov);
            self.mean = Some(new_mean);
            self.covariance = Some(new_cov);
        }
    }

    pub fn mark_lost(&mut self) {
        self.state = TrackState::Lost;
    }

    pub fn mark_removed(&mut self) {
        self.state = TrackState::Removed;
    }

    pub fn multi_predict(stracks: &mut [STrack], kalman_filter: &KalmanFilter) {
        for strack in stracks.iter_mut() {
            strack.predict(kalman_filter);
        }
    }
}
