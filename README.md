# corinth-canal

Neuromorphic-ANN hybrid framework implemented in this repository.

[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue)](LICENSE)

## Overview

`corinth-canal` provides a hybrid pipeline with:
- telemetry-to-spike encoding
- spiking network simulation
- dense/spiking projection into embedding space
- frozen OLMoE-style routing simulation
- optional training signal publishing via ZMQ

## Architecture

```text
TelemetrySnapshot
       │
       ▼  Encoder
 [f32; 16] Poisson stimuli
       │
       ▼  SpikingNetwork × snn_steps
 spike_train + membrane_potentials
       │
       ▼  Projector
 dense embedding [2048]
       │
       ▼  OLMoE (frozen/simulated)
 expert_weights + selected_experts + hidden
       │
       ▼  Optional spine publish
 TrainSignal
```

## Quick start

```rust
use corinth_canal::{HybridConfig, HybridModel};
use spikenaut_telemetry::TelemetrySnapshot;

let cfg = HybridConfig::default();
let mut model = HybridModel::new(cfg)?;

let snap = TelemetrySnapshot::default();
let output = model.forward(&snap)?;

println!("Selected experts: {:?}", output.selected_experts);
```

## Features

| Feature | Effect |
|---------|--------|
| `gguf` | GGUF model header parsing support |
| `safetensors` | safetensors model loading support |
| `spine-zmq` | ZMQ publishing support for train signals |

## Run example

```bash
cargo run --example telemetry_bridge
```

With GGUF feature enabled:

```bash
OLMOE_PATH=/models/OLMoE-1B-7B-Q5_K_M.gguf \
  cargo run --example telemetry_bridge --features gguf --release
```

## Project layout

| Path | Responsibility |
|------|----------------|
| `Cargo.toml` | crate config and dependencies |
| `src/lib.rs` | public API and re-exports |
| `src/types.rs` | `HybridConfig`, `HybridOutput`, enums/constants |
| `src/error.rs` | `HybridError`, `Result` |
| `src/tensor/mod.rs` | candle-free tensor utilities |
| `src/transformer/mod.rs` | transformer helpers |
| `src/hybrid/mod.rs` | hybrid module wiring |
| `src/hybrid/projector.rs` | 2-bit spiking projector logic |
| `src/hybrid/olmoe.rs` | GGUF loader and OLMoE simulation |
| `src/hybrid/hybrid.rs` | top-level orchestrator (`HybridModel`) |
| `examples/telemetry_bridge.rs` | end-to-end example |

## Acknowledgments

This work builds directly on prior research in spiking and spike-driven LLMs.

- Xing et al. (2024), *SpikeLLM: Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking*.
  - arXiv: [2407.04752](https://arxiv.org/abs/2407.04752)
  - DOI: [10.48550/arXiv.2407.04752](https://doi.org/10.48550/arXiv.2407.04752)
- Xu et al. (2025), *Neuromorphic spike-based large language model* (`NSLLM`).
  - Journal page: [National Science Review](https://academic.oup.com/nsr/advance-article/doi/10.1093/nsr/nwaf551/8365570)
  - DOI: [10.1093/nsr/nwaf551](https://doi.org/10.1093/nsr/nwaf551)
- Qiu et al. (2025), *Quantized Spike-driven Transformer* (`QSD-Transformer`).
  - ICLR poster: [OpenReview / ICLR 2025](https://iclr.cc/virtual/2025/poster/30954)
  - arXiv: [2501.13492](https://arxiv.org/abs/2501.13492)
- The Allen Institute for AI for releasing the open OLMoE models used as the primary testbed.

## Citation

If you use `corinth-canal` or the SNN-logic quantization approach in your research, please cite:

```bibtex
@misc{corinth-canal2026,
  title        = {corinth-canal: Turning MOE Architecture into SNN Quantization},
  author       = {Raul Montoya Cardenas},
  year         = {2026},
  howpublished = {\url{https://github.com/Spikenaut/corinth-canal}},
  note         = {SNN-logic quantization with GIF-Ternary spiking for MoE models}
}
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).

## Built By

Raul Montoya Cardenas

Fully open source. Contributions and forks welcome.
