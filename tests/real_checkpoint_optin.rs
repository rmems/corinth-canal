// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Smoke tests against real local checkpoints. They are ignored by default and
//! fail loudly, rather than skip, when run without their variable:
//!
//! ```bash
//! CHECKPOINT_PATH=/path/to/model.gguf \
//! OLMOE_SAFETENSORS_DIR=/path/to/olmoe-dir \
//! cargo test --no-default-features --test real_checkpoint_optin -- --ignored
//! ```

use corinth_canal::moe::{Router, RoutingMode};
use corinth_canal::{EMBEDDING_DIM, ModelFamily};

#[test]
#[ignore = "needs a local checkpoint; run with --ignored"]
fn test_real_checkpoint_probe_via_env() {
    let path = std::env::var("CHECKPOINT_PATH").expect("CHECKPOINT_PATH must be set");

    let metadata = Router::probe_model(&path, None).unwrap();
    assert!(!metadata.architecture.is_empty());
    assert!(metadata.hidden_size > 0);
    assert!(metadata.num_experts > 0);
    assert!(!metadata.routing_tensor_name.is_empty());
}

#[test]
#[ignore = "needs a local checkpoint; run with --ignored"]
fn test_safetensors_olmoe_load_and_route() {
    let path = std::env::var("OLMOE_SAFETENSORS_DIR").expect("OLMOE_SAFETENSORS_DIR must be set");
    let path_str = path.as_str();

    let metadata = Router::probe_model(path_str, None).unwrap();
    assert_eq!(metadata.family, ModelFamily::Olmoe);
    assert_eq!(metadata.hidden_size, 2048);
    assert_eq!(metadata.num_experts, 64);
    assert_eq!(metadata.expert_used_count, 8);
    assert_eq!(metadata.architecture, "OlmoeForCausalLM");
    assert_eq!(metadata.quantization, "safetensors");

    let mut model = Router::load_with_mode(path_str, 0, 0, RoutingMode::DenseSim).unwrap();
    assert!(model.is_loaded());
    assert_eq!(model.family(), ModelFamily::Olmoe);
    assert_eq!(model.hidden_size(), 2048);
    assert_eq!(model.checkpoint_num_experts(), 64);
    assert_eq!(model.checkpoint_expert_used_count(), 8);
    assert_eq!(
        model.routing_tensor_name(),
        "model.layers.0.mlp.gate.weight"
    );

    // Forward pass with a dummy embedding
    let embedding = vec![0.0f32; EMBEDDING_DIM];
    let out = model.forward(&embedding).unwrap();
    assert_eq!(out.expert_weights.len(), 64);
    assert_eq!(out.selected_experts.len(), 8);

    // Token embedding extraction
    let token_emb = model.extract_token_embedding(0).unwrap();
    assert_eq!(token_emb.len(), EMBEDDING_DIM);
}
