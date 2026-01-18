//! Main BYTETracker algorithm implementation.

use crate::tracker::kalman_filter::KalmanFilter;
use crate::tracker::matching::{self, AssignmentResult, Detection};
use crate::tracker::rect::{Rect, iou_batch};
use crate::tracker::strack::STrack;
use crate::tracker::track_state::TrackState;

/// Configuration for the BYTETracker.
#[derive(Debug, Clone)]
pub struct TrackerConfig {
    pub track_thresh: f32,
    pub match_thresh: f32,
    pub track_buffer: u32,
    pub frame_rate: f32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            track_thresh: 0.5,
            match_thresh: 0.8,
            track_buffer: 30,
            frame_rate: 30.0,
        }
    }
}

pub struct BYTETracker {
    tracked_stracks: Vec<STrack>,
    lost_stracks: Vec<STrack>,
    removed_stracks: Vec<STrack>,
    frame_id: u32,
    config: TrackerConfig,
    max_time_lost: u32,
    kalman_filter: KalmanFilter,
}

impl BYTETracker {
    pub fn new(config: TrackerConfig) -> Self {
        let max_time_lost = (config.frame_rate / 30.0 * config.track_buffer as f32) as u32;
        Self {
            tracked_stracks: Vec::new(),
            lost_stracks: Vec::new(),
            removed_stracks: Vec::new(),
            frame_id: 0,
            config,
            max_time_lost,
            kalman_filter: KalmanFilter::default(),
        }
    }

    pub fn update(&mut self, detections: Vec<Detection>) -> Vec<STrack> {
        self.frame_id += 1;

        let mut activated_stracks = Vec::new();
        let mut refind_stracks = Vec::new();
        let mut lost_stracks = Vec::new();
        let mut removed_stracks = Vec::new();

        // Step 1: Split detections into high-score and low-score
        let mut remain_detections = Vec::new();
        let mut detections_low = Vec::new();

        for det in detections {
            if det.score >= self.config.track_thresh {
                remain_detections.push(det);
            } else if det.score > 0.1 {
                detections_low.push(det);
            }
        }

        let detections = remain_detections
            .into_iter()
            .map(|d| STrack::new(d.bbox, d.score))
            .collect::<Vec<_>>();

        // Create track pool
        let mut unconfirmed = Vec::new();
        let mut tracked_stracks = Vec::new();
        for track in self.tracked_stracks.drain(..) {
            if !track.is_activated {
                unconfirmed.push(track);
            } else {
                tracked_stracks.push(track);
            }
        }

        let mut strack_pool = joint_stracks(tracked_stracks, &self.lost_stracks);

        // Step 2: First association, with high score detections
        STrack::multi_predict(&mut strack_pool, &self.kalman_filter);

        let pool_rects: Vec<Rect> = strack_pool.iter().map(|t| t.rect()).collect();
        let det_rects: Vec<Rect> = detections.iter().map(|t| t.rect()).collect();
        let mut dists = matching::iou_distance(&pool_rects, &det_rects);

        let det_wrappers: Vec<Detection> = detections
            .iter()
            .map(|t| Detection::from_rect(t.rect(), t.score))
            .collect();
        matching::fuse_score(&mut dists, &det_wrappers);

        let AssignmentResult {
            matches,
            unmatched_tracks,
            unmatched_detections,
        } = matching::linear_assignment(&dists, self.config.match_thresh);

        for (itracked, idet) in matches {
            let mut track = strack_pool[itracked].clone();
            let det = &detections[idet];
            if track.state == TrackState::Tracked {
                track.update(det, &self.kalman_filter, self.frame_id);
                activated_stracks.push(track);
            } else {
                track.re_activate(det, &self.kalman_filter, self.frame_id, false);
                refind_stracks.push(track);
            }
        }

        // Step 3: Second association, with low score detection boxes
        let detections_second = detections_low
            .into_iter()
            .map(|d| STrack::new(d.bbox, d.score))
            .collect::<Vec<_>>();

        let mut r_tracked_stracks = Vec::new();
        for &idx in &unmatched_tracks {
            if strack_pool[idx].state == TrackState::Tracked {
                r_tracked_stracks.push(strack_pool[idx].clone());
            }
        }

        let r_rects: Vec<Rect> = r_tracked_stracks.iter().map(|t| t.rect()).collect();
        let det_low_rects: Vec<Rect> = detections_second.iter().map(|t| t.rect()).collect();
        let dists_second = matching::iou_distance(&r_rects, &det_low_rects);

        let AssignmentResult {
            matches: matches_second,
            unmatched_tracks: unmatched_tracks_second,
            ..
        } = matching::linear_assignment(&dists_second, 0.5);

        for (itracked, idet) in matches_second {
            let mut track = r_tracked_stracks[itracked].clone();
            let det = &detections_second[idet];
            if track.state == TrackState::Tracked {
                track.update(det, &self.kalman_filter, self.frame_id);
                activated_stracks.push(track);
            } else {
                track.re_activate(det, &self.kalman_filter, self.frame_id, false);
                refind_stracks.push(track);
            }
        }

        for idx in unmatched_tracks_second {
            let mut track = r_tracked_stracks[idx].clone();
            if track.state != TrackState::Lost {
                track.mark_lost();
                lost_stracks.push(track);
            }
        }

        // Deal with unconfirmed tracks, usually tracks with only one beginning frame
        let mut detections_rem = Vec::new();
        for idx in unmatched_detections {
            detections_rem.push(detections[idx].clone());
        }

        let unconfirmed_rects: Vec<Rect> = unconfirmed.iter().map(|t| t.rect()).collect();
        let det_rem_rects: Vec<Rect> = detections_rem.iter().map(|t| t.rect()).collect();
        let mut dist_unconfirmed = matching::iou_distance(&unconfirmed_rects, &det_rem_rects);

        let det_rem_wrappers: Vec<Detection> = detections_rem
            .iter()
            .map(|t| Detection::from_rect(t.rect(), t.score))
            .collect();
        matching::fuse_score(&mut dist_unconfirmed, &det_rem_wrappers);

        let AssignmentResult {
            matches: matches_unconfirmed,
            unmatched_tracks: unmatched_unconfirmed,
            unmatched_detections: unmatched_new,
        } = matching::linear_assignment(&dist_unconfirmed, 0.7);

        for (itracked, idet) in matches_unconfirmed {
            unconfirmed[itracked].update(&detections_rem[idet], &self.kalman_filter, self.frame_id);
            activated_stracks.push(unconfirmed[itracked].clone());
        }
        for idx in unmatched_unconfirmed {
            let mut track = unconfirmed[idx].clone();
            track.mark_removed();
            removed_stracks.push(track);
        }

        // Step 4: Init new stracks
        for idx in unmatched_new {
            let mut track = detections_rem[idx].clone();
            if track.score < self.config.track_thresh + 0.1 {
                continue;
            }
            track.activate(&self.kalman_filter, self.frame_id);
            activated_stracks.push(track);
        }

        // Step 5: Update state
        for mut track in self.lost_stracks.drain(..) {
            if self.frame_id - track.end_frame() > self.max_time_lost {
                track.mark_removed();
                removed_stracks.push(track);
            } else {
                lost_stracks.push(track);
            }
        }

        self.tracked_stracks = activated_stracks
            .into_iter()
            .chain(refind_stracks.into_iter())
            .filter(|t| t.state == TrackState::Tracked)
            .collect();

        self.lost_stracks = sub_stracks(lost_stracks, &self.tracked_stracks);
        self.removed_stracks.extend(removed_stracks);

        let (tracked, lost) = remove_duplicate_stracks(&self.tracked_stracks, &self.lost_stracks);
        self.tracked_stracks = tracked;
        self.lost_stracks = lost;

        self.tracked_stracks
            .iter()
            .filter(|t| t.is_activated)
            .cloned()
            .collect()
    }
}

pub fn joint_stracks(tlista: Vec<STrack>, tlistb: &[STrack]) -> Vec<STrack> {
    let mut exists = std::collections::HashSet::new();
    let mut res = Vec::new();
    for t in tlista {
        exists.insert(t.track_id);
        res.push(t);
    }
    for t in tlistb {
        if !exists.contains(&t.track_id) {
            exists.insert(t.track_id);
            res.push(t.clone());
        }
    }
    res
}

pub fn sub_stracks(tlista: Vec<STrack>, tlistb: &[STrack]) -> Vec<STrack> {
    let mut b_ids = std::collections::HashSet::new();
    for t in tlistb {
        b_ids.insert(t.track_id);
    }
    tlista
        .into_iter()
        .filter(|t| !b_ids.contains(&t.track_id))
        .collect()
}

pub fn remove_duplicate_stracks(
    stracksa: &[STrack],
    stracksb: &[STrack],
) -> (Vec<STrack>, Vec<STrack>) {
    if stracksa.is_empty() || stracksb.is_empty() {
        return (stracksa.to_vec(), stracksb.to_vec());
    }

    let a_rects: Vec<Rect> = stracksa.iter().map(|t| t.rect()).collect();
    let b_rects: Vec<Rect> = stracksb.iter().map(|t| t.rect()).collect();
    let ious = iou_batch(&a_rects, &b_rects);

    let mut dupa = vec![false; stracksa.len()];
    let mut dupb = vec![false; stracksb.len()];

    let (rows, cols) = ious.dim();
    for i in 0..rows {
        for j in 0..cols {
            if ious[[i, j]] > 0.85 {
                let time_a = stracksa[i].frame_id - stracksa[i].start_frame;
                let time_b = stracksb[j].frame_id - stracksb[j].start_frame;
                if time_a > time_b {
                    dupb[j] = true;
                } else {
                    dupa[i] = true;
                }
            }
        }
    }

    let resa = stracksa
        .iter()
        .enumerate()
        .filter(|(i, _)| !dupa[*i])
        .map(|(_, t)| t.clone())
        .collect();
    let resb = stracksb
        .iter()
        .enumerate()
        .filter(|(j, _)| !dupb[*j])
        .map(|(_, t)| t.clone())
        .collect();

    (resa, resb)
}
