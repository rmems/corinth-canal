// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Corinth-side guard for the `SparseGifHiddenLayer` upstream port (#157).
//!
//! `neuromod` 0.7.0 ships `neuromod::gif_layer::SparseGifHiddenLayer`, ported
//! from `src/funnel.rs`. Its `layer_corinth_parity_tests.rs` replays
//! `tests/reference/corinth_gif_parity.json`, a bit-exact fixture captured from
//! this crate's layer at commit `8e54e234ac005dd84e4ad2bedbf9f5bceb082355`.
//!
//! That check only runs in neuromod, against a frozen snapshot. This test is the
//! reverse direction: it replays the fixture's deterministic input against the
//! live local copy and compares digests of the fixture's spike IDs and final
//! membrane state. A failure means the local layer no longer matches what
//! neuromod pins, so the fixture must be regenerated before adoption.
//!
//! The local copy stays until adoption, which is a separate follow-up issue.
//! No neuromod dependency is taken here.
//!
//! The expected digests were computed from the neuromod fixture, not from this
//! crate: FNV-1a 64 over `per_step_fired_ids` as `(step: u32 LE, id: u32 LE)`
//! pairs, and over `(membrane / (0.65 * 2.0)).clamp(0, 1)` f32 bits (LE) from
//! `final_membrane_bits`, which is the normalisation `run()` applies to the
//! potentials it returns.

use crate::funnel::{FUNNEL_HIDDEN_NEURONS, FUNNEL_INPUT_NEURONS, SparseGifHiddenLayer};

/// Steps in the neuromod fixture's audited case.
const FIXTURE_STEPS: usize = 512;
/// `counts.total_spikes` in `corinth_gif_parity.json`.
const FIXTURE_TOTAL_SPIKES: usize = 8357;
/// FNV-1a 64 of the fixture's `per_step_fired_ids`.
const FIXTURE_SPIKE_DIGEST: u64 = 0xeb0d_2449_8f9d_36f8;
/// FNV-1a 64 of the normalised fixture `final_membrane_bits`.
const FIXTURE_POTENTIAL_DIGEST: u64 = 0x5357_d42d_26b3_e53e;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// The fixture's input rule: channel `i` fires at step `t` iff
/// `(i * 31 + t * 17) % 23 == 0`.
fn fixture_input_train() -> Vec<Vec<usize>> {
    (0..FIXTURE_STEPS)
        .map(|t| {
            (0..FUNNEL_INPUT_NEURONS)
                .filter(|i| (i * 31 + t * 17) % 23 == 0)
                .collect()
        })
        .collect()
}

#[test]
fn local_gif_layer_matches_neuromod_corinth_parity_fixture() {
    let mut layer = SparseGifHiddenLayer::new();
    let (spike_train, potentials, _) = layer.run(&fixture_input_train());

    assert_eq!(spike_train.len(), FIXTURE_STEPS);
    assert_eq!(potentials.len(), FUNNEL_HIDDEN_NEURONS);

    let total_spikes: usize = spike_train.iter().map(Vec::len).sum();
    assert_eq!(
        total_spikes, FIXTURE_TOTAL_SPIKES,
        "total spike count drifted"
    );

    let mut spike_digest = FNV_OFFSET;
    for (step, fired) in spike_train.iter().enumerate() {
        assert!(fired.is_sorted(), "step {step} fired IDs must be ascending");
        for &id in fired {
            spike_digest = fnv1a(spike_digest, &(step as u32).to_le_bytes());
            spike_digest = fnv1a(spike_digest, &(id as u32).to_le_bytes());
        }
    }
    assert_eq!(
        spike_digest, FIXTURE_SPIKE_DIGEST,
        "per-step spike IDs drifted from neuromod's Corinth parity fixture"
    );

    let potential_digest = potentials.iter().fold(FNV_OFFSET, |hash, value| {
        fnv1a(hash, &value.to_bits().to_le_bytes())
    });
    assert_eq!(
        potential_digest, FIXTURE_POTENTIAL_DIGEST,
        "final membrane state drifted from neuromod's Corinth parity fixture"
    );
}
