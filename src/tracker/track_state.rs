/// Track state enumeration for object tracking lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrackState {
    /// Newly created track, not yet confirmed
    #[default]
    New,
    /// Actively tracked object
    Tracked,
    /// Temporarily lost track
    Lost,
    /// Removed from tracking
    Removed,
}
