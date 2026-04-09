# corinth-canal

Standalone SNN-logic quantization crate focused on the spiking projector and OLMoE routing path.

[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue)](LICENSE)

## Overview

`corinth-canal` keeps the real projector and OLMoE simulation logic, while the old telemetry/SNN front-end is replaced by a deterministic in-repo spike generator. That keeps the crate self-contained and runnable from a fresh clone.

## Architecture

```text
TelemetrySnapshot
       |
       v  deterministic dummy spike generator
spike_train + membrane_potentials
       |
       v  Projector
dense embedding [2048]
       |
       v  OLMoE (stub/dense/spiking sim)
expert_weights + selected_experts + hidden
```

## Quick start

```rust
use corinth_canal::{HybridConfig, HybridModel, TelemetrySnapshot};

let cfg = HybridConfig::default();
let mut model = HybridModel::new(cfg)?;

let snap = TelemetrySnapshot::default();
let output = model.forward(&snap)?;

println!("Selected experts: {:?}", output.selected_experts);
```

## Features

| Feature | Effect |
|---------|--------|
| `gguf` | Enables GGUF header validation for model-file probing |

## Run example

```bash
cargo run --example telemetry_bridge
```

With GGUF probing enabled:

```bash
OLMOE_PATH=/models/OLMoE-1B-7B-Q5_K_M.gguf \
  cargo run --example telemetry_bridge --features gguf --release
```

## Project layout

| Path | Responsibility |
|------|----------------|
| `Cargo.toml` | crate config and dependencies |
| `src/lib.rs` | public API and crate docs |
| `src/types.rs` | `TelemetrySnapshot`, config, enums, output types |
| `src/error.rs` | `HybridError`, `Result` |
| `src/tensor/mod.rs` | candle-free tensor utilities |
| `src/transformer/mod.rs` | transformer helpers |
| `src/hybrid/projector.rs` | 2-bit spiking projector logic |
| `src/hybrid/olmoe.rs` | GGUF-aware OLMoE simulation |
| `src/hybrid/hybrid.rs` | deterministic front-end + projector + OLMoE orchestration |
| `examples/telemetry_bridge.rs` | end-to-end standalone example |

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
