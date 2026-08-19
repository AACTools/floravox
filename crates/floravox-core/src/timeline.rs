//! Sample-accurate event scheduling for playback threads.
//!
//! The synthesizer emits [`SynthesisEvent`]s tagged with absolute sample
//! positions. A consumer that is pushing PCM to an output device keeps a
//! **playback cursor** (how many samples it has actually written) and calls
//! [`EventTimeline::drain_until`] after every buffer write. Events fire at
//! the exact sample they were scheduled for — no timers, no drift.

use crate::events::SynthesisEvent;

/// An ordered queue of events keyed by sample position.
#[derive(Debug, Default)]
pub struct EventTimeline {
    events: Vec<(u64, SynthesisEvent)>,
    sorted: bool,
}

impl EventTimeline {
    /// A new, empty timeline.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue an event; its fire position is taken from
    /// [`SynthesisEvent::sample`].
    pub fn push(&mut self, sample: u64, event: SynthesisEvent) {
        self.events.push((sample, event));
        self.sorted = false;
    }

    /// Queue an event using its own sample position.
    pub fn push_event(&mut self, event: SynthesisEvent) {
        let s = event.sample();
        self.push(s, event);
    }

    fn ensure_sorted(&mut self) {
        if !self.sorted {
            self.events.sort_by_key(|(s, _)| *s);
            self.sorted = true;
        }
    }

    /// Remove and return every event at or before `cursor` samples, in
    /// sample order (stable for equal positions).
    pub fn drain_until(&mut self, cursor: u64) -> Vec<SynthesisEvent> {
        self.ensure_sorted();
        let split = self.events.partition_point(|(s, _)| *s <= cursor);
        self.events.drain(..split).map(|(_, e)| e).collect()
    }

    /// The sample position of the next pending event, if any. Lets a
    /// playback loop know how far it may write before checking again.
    #[must_use]
    pub fn next_event_sample(&mut self) -> Option<u64> {
        self.ensure_sorted();
        self.events.first().map(|(s, _)| *s)
    }

    /// Number of pending events.
    #[must_use]
    pub fn pending(&mut self) -> usize {
        self.events.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::WordTiming;

    fn word(text: &str, sample_start: u64, sample_end: u64) -> SynthesisEvent {
        SynthesisEvent::WordBoundary(WordTiming {
            text: text.into(),
            byte_offset: 0,
            byte_len: text.len(),
            char_offset: 0,
            char_len: text.len(),
            sample_start,
            sample_end,
            ms_start: sample_start / 22,
            ms_end: sample_end / 22,
            estimated: false,
        })
    }

    #[test]
    fn drains_in_order_at_cursor() {
        let mut tl = EventTimeline::new();
        tl.push_event(word("b", 100, 200));
        tl.push_event(word("a", 0, 100));
        tl.push_event(SynthesisEvent::MarkReached {
            name: "m".into(),
            sample: 150,
            ms: 7,
        });

        assert_eq!(tl.drain_until(0).len(), 1); // "a"
        let batch = tl.drain_until(160); // "b" + mark
        assert_eq!(batch.len(), 2);
        // stable order among equals (100 < 150 anyway)
        assert_eq!(tl.pending(), 0);
    }

    #[test]
    fn next_event_sample_guides_writes() {
        let mut tl = EventTimeline::new();
        tl.push_event(word("x", 5000, 6000));
        assert_eq!(tl.next_event_sample(), Some(5000));
        assert_eq!(tl.drain_until(4999).len(), 0);
        assert_eq!(tl.next_event_sample(), Some(5000));
        assert_eq!(tl.drain_until(5000).len(), 1);
        assert_eq!(tl.next_event_sample(), None);
    }

    #[test]
    fn equal_sample_positions_fire_in_push_order() {
        let mut tl = EventTimeline::new();
        tl.push_event(SynthesisEvent::BreakStarted { ms: 0, sample: 10 });
        tl.push_event(SynthesisEvent::BreakEnded { sample: 10 });
        let fired = tl.drain_until(10);
        assert!(matches!(fired[0], SynthesisEvent::BreakStarted { .. }));
        assert!(matches!(fired[1], SynthesisEvent::BreakEnded { .. }));
    }
}
