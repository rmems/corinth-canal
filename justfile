set dotenv-load := true
set dotenv-filename := ".env.local"

# Default: list all recipes.
default:
    @just --list

# One-time setup sanity check: verify every doc + manifest exists.
setup:
    @test -f .env.local || echo "warn: .env.local missing (copy from .env.example)"
    @test -d artifacts  || mkdir -p artifacts
    @echo "ok: scaffolding present"

# Fast compile sweep.
check:
    cargo check --all-targets

# Full test suite (CPU-only paths; GPU tests gated on hardware).
test:
    cargo test
    python3 -m unittest discover -s benchmarks/tests -v

# GPU smoke test — 10k direct GPU ticks against a real GGUF checkpoint.
# Requires CHECKPOINT_PATH in .env.local.
smoke:
    cargo run --release --example gpu_smoke_test

# CSV replay demo.
#   just replay /path/to/telemetry.csv
replay PATH:
    cargo run --release --example csv_replay -- {{PATH}}

# Full SAAQ latent calibration sweep using current .env.local values.
saaq:
    cargo run --release --example saaq_latent_calibration

# Phases: synthetic baseline, csv replay baseline.
# Reads LINEUP_CONFIG and (for phases 2-3) TELEMETRY_CSV_PATH from .env.local.
# Falls back to configs/local_gguf_lineup.toml when LINEUP_CONFIG is unset.
# That file is gitignored (machine-local); copy it from
# configs/local_gguf_lineup.template.toml first.
# Full SAAQ 1.5 MoE baseline campaign (2 phases x REPEAT_COUNT runs per model).
saaq-campaign:
    @echo ">>> phase 1/2: synthetic baseline, repeat=2"
    LINEUP_CONFIG="${LINEUP_CONFIG:-configs/local_gguf_lineup.toml}" \
        SAAQ_RULE=saaq_v1_5 REPEAT_COUNT=2 TELEMETRY_SOURCE=synthetic \
        RUN_TAG=campaign_syn \
        cargo run --release --example saaq_latent_calibration
    @echo ">>> phase 2/2: csv replay baseline, repeat=2"
    LINEUP_CONFIG="${LINEUP_CONFIG:-configs/local_gguf_lineup.toml}" \
        SAAQ_RULE=saaq_v1_5 REPEAT_COUNT=2 TELEMETRY_SOURCE=csv \
        RUN_TAG=campaign_csv \
        cargo run --release --example saaq_latent_calibration
    @echo "ok: campaign finished; see artifacts/index.csv"

# Force CSV-replay mode for the SAAQ sweep. TELEMETRY_CSV_PATH overrides the
# CSV to replay; when it is unset, telemetry_csv_path_from_env falls back to
# `telemetry.csv` relative to the working directory, so the sweep runs against
# whatever that resolves to rather than failing. This recipe takes no
# parameters, so set it in .env.local or pass it as a variable assignment:
#   TELEMETRY_CSV_PATH=/path/to/telemetry.csv just saaq-csv
saaq-csv:
    TELEMETRY_SOURCE=csv cargo run --release --example saaq_latent_calibration

# Matrix sweep over configured models/telemetry with dual SAAQ emission.
saaq-sweep:
    cargo run --release --example saaq_latent_calibration

# Telemetry bridge demo (routing_mode switchable via ROUTING_MODE env).
bridge:
    cargo run --release --example telemetry_bridge

# Probe the configured lineup (LINEUP_CONFIG / CHECKPOINT_PATH /
# autodiscovery) and print the preferred GPU synapse tensor + ggml_type per
# model. Writes <output_root>/synapse_diagnostic.json. No SAAQ ticks and no
# campaign side-effects (issue #31).
synapse-diag:
    cargo run --release --example synapse_diagnostic

# Wipe everything under ./artifacts except the .gitkeep anchor.
clean-artifacts:
    find artifacts -mindepth 1 ! -name .gitkeep -exec rm -rf {} +
    @echo "ok: artifacts/ emptied"
