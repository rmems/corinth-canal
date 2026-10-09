// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Corinth-side guard for the `SparseGifHiddenLayer` upstream port (#157).
//!
//! `neuromod` 0.7.0 ships `neuromod::gif_layer::SparseGifHiddenLayer`, ported
//! from `src/funnel.rs`. Its `layer_corinth_parity_tests.rs` replays
//! `tests/reference/corinth_gif_parity.json`, a bit-exact fixture captured from
//! this crate's layer at commit `8e54e234ac005dd84e4ad2bedbf9f5bceb082355`.
//!
//! That check only runs in neuromod, against a frozen snapshot. This test is the
//! reverse direction: it rebuilds the fixture's deterministic input, runs it
//! through the live local layer, and compares the full state the fixture
//! records: the fan-in topology in edge order, every per-step spike ID, and the
//! raw final membrane and adaptation f32 bits. A failure means the local layer
//! no longer matches what neuromod pins, so the fixture must be regenerated
//! before adoption.
//!
//! The local copy stays until adoption, which is a separate follow-up issue.
//! No neuromod dependency is taken here.
//!
//! The expected digests were computed from the neuromod fixture, not from this
//! crate. Each is FNV-1a 64 over little-endian `u32` words:
//! - topology: `topology_edge_order` flattened, i.e. `(source, weight bits)` per
//!   edge in Corinth edge order, neuron by neuron;
//! - spikes: `(step, id)` for every entry of `per_step_fired_ids`;
//! - membrane / adaptation: `final_membrane_bits` / `final_adaptation_bits`.

use super::{FUNNEL_HIDDEN_NEURONS, FUNNEL_INPUT_NEURONS, SparseGifHiddenLayer};

/// Steps in the neuromod fixture's audited case.
const FIXTURE_STEPS: usize = 512;
/// `counts.total_spikes` in `corinth_gif_parity.json`.
const FIXTURE_TOTAL_SPIKES: usize = 8357;
const FIXTURE_TOPOLOGY_DIGEST: u64 = 0xaa05_8a43_29f4_1b29;
const FIXTURE_SPIKE_DIGEST: u64 = 0xeb0d_2449_8f9d_36f8;
const FIXTURE_MEMBRANE_DIGEST: u64 = 0x8f8f_30d5_a1c0_d1c7;
const FIXTURE_ADAPTATION_DIGEST: u64 = 0xd3c3_2579_fcb5_728e;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a_words(words: impl IntoIterator<Item = u32>) -> u64 {
    words.into_iter().fold(FNV_OFFSET, |mut hash, word| {
        for byte in word.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    })
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

    let topology_digest = fnv1a_words(
        layer
            .weight_indices
            .iter()
            .zip(&layer.weight_values)
            .flat_map(|(indices, values)| indices.iter().zip(values))
            .flat_map(|(&source, value)| [source as u32, value.to_bits()]),
    );
    assert_eq!(
        topology_digest, FIXTURE_TOPOLOGY_DIGEST,
        "fan-in topology drifted from neuromod's Corinth parity fixture"
    );

    let (spike_train, _, _) = layer.run(&fixture_input_train());
    assert_eq!(spike_train.len(), FIXTURE_STEPS);

    let total_spikes: usize = spike_train.iter().map(Vec::len).sum();
    assert_eq!(
        total_spikes, FIXTURE_TOTAL_SPIKES,
        "total spike count drifted"
    );
    for (step, fired) in spike_train.iter().enumerate() {
        assert!(fired.is_sorted(), "step {step} fired IDs must be ascending");
    }
    let spike_digest = fnv1a_words(
        spike_train
            .iter()
            .enumerate()
            .flat_map(|(step, fired)| fired.iter().flat_map(move |&id| [step as u32, id as u32])),
    );
    assert_eq!(
        spike_digest, FIXTURE_SPIKE_DIGEST,
        "per-step spike IDs drifted from neuromod's Corinth parity fixture"
    );

    assert_eq!(layer.membrane.len(), FUNNEL_HIDDEN_NEURONS);
    assert_eq!(
        fnv1a_words(layer.membrane.iter().map(|value| value.to_bits())),
        FIXTURE_MEMBRANE_DIGEST,
        "final membrane bits drifted from neuromod's Corinth parity fixture"
    );
    assert_eq!(
        fnv1a_words(layer.adaptation.iter().map(|value| value.to_bits())),
        FIXTURE_ADAPTATION_DIGEST,
        "final adaptation bits drifted from neuromod's Corinth parity fixture"
    );
}
