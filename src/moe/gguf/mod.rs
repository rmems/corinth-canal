// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Internal GGUF checkpoint implementation modules.
//!
//! Split from the former monolithic `checkpoint.rs` for reviewability.
//! Public (to `moe`) surface is re-exported here and mirrored by the
//! `checkpoint` façade so existing `super::checkpoint::{...}` imports keep
//! working.
//!
//! #115 will consume `rmems/engram-parser` (`mmap` feature + packed
//! Q8_0/Q5_K/Q6_K and internal-only `IQ3_M_BLOCK` — not a GGUF wire type) for
//! parse/mmap/dequant. That surface shipped in engram-parser#73 after #144 chose
//! option 1 (upstream first, adopt second). CUDA host-register stays in this
//! crate. Until the swap lands, this directory remains the live reference copy.

mod cuda_register;
mod dequant;
mod map;
mod metadata;

pub(super) use dequant::f16_to_f32;
pub(super) use map::{MappedGgufCheckpoint, probe_and_map_checkpoint};
pub(super) use metadata::{GgufMetadata, GgufTensorInfo};
