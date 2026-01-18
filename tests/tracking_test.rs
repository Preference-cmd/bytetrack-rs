use bytetrack_rs::tracker::reset_track_id_counter;
use bytetrack_rs::{BYTETracker, Detection, TrackerConfig};

#[test]
fn test_basic_tracking() {
    reset_track_id_counter();
    let mut tracker = BYTETracker::new(TrackerConfig::default());

    // Frame 1: One detection
    let dets1 = vec![Detection::new(100.0, 100.0, 200.0, 200.0, 0.9)];
    let tracks1 = tracker.update(dets1);

    // In frame 1, new tracks are initialized but might not be activated
    // depending on the threshold (track_thresh + 0.1).
    // Our track_thresh is 0.5, 0.9 > 0.6, so it should be activated if frame_id == 1.
    assert_eq!(tracks1.len(), 1);
    let id1 = tracks1[0].track_id;

    // Frame 2: Same object moved slightly
    let dets2 = vec![Detection::new(105.0, 105.0, 205.0, 205.0, 0.9)];
    let tracks2 = tracker.update(dets2);
    assert_eq!(tracks2.len(), 1);
    assert_eq!(tracks2[0].track_id, id1); // ID should persist

    // Frame 3: Object occluded (low score)
    let dets3 = vec![
        Detection::new(110.0, 110.0, 210.0, 210.0, 0.2), // Low score
    ];
    let tracks3 = tracker.update(dets3);
    // ByteTrack should recover it using the second association
    assert_eq!(tracks3.len(), 1);
    assert_eq!(tracks3[0].track_id, id1);

    // Frame 4: Object disappears
    let dets4 = vec![];
    let tracks4 = tracker.update(dets4);
    assert_eq!(tracks4.len(), 0);

    // Frame 5: Object reappears
    let dets5 = vec![Detection::new(115.0, 115.0, 215.0, 215.0, 0.9)];
    let tracks5 = tracker.update(dets5);
    // Should be refound if within track_buffer
    assert_eq!(tracks5.len(), 1);
    assert_eq!(tracks5[0].track_id, id1);
}
