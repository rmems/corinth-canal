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
#   just replay PATH=/path/to/telemetry.csv
replay PATH:
    cargo run --release --example csv_replay -- {{PATH}}

# Full SAAQ latent calibration sweep using current .env.local values.
saaq:
    cargo run --release --example saaq_latent_calibration

# Phases: synthetic baseline, csv replay baseline.
# Reads LINEUP_CONFIG and TELEMETRY_CSV_PATH from .env.local.
#
# Both are REQUIRED, and both are checked before phase 1 so a misconfigured
# campaign fails immediately rather than after a full run:
#
#   LINEUP_CONFIG      A baseline campaign has to pin its model set, or the
#                      two phases are not comparable. It must not be defaulted
#                      either: naming a missing file is a hard error
#                      (config.rs::resolve_validation_models), and leaving it
#                      unset falls through to autodiscovery, whose candidate
#                      list includes the dense `glm46v_flash_q8_0` — which
#                      aborts the sweep on missing `expert_count`
#                      (adapter.rs:201). Copy
#                      configs/local_gguf_lineup.template.toml and point at it.
#   TELEMETRY_CSV_PATH Phase 2 sets TELEMETRY_SOURCE=csv, but a CSV the runner
#                      cannot use makes it degrade to synthetic telemetry,
#                      turning this into a synthetic-vs-synthetic comparison
#                      that still reports success. The preflight mirrors
#                      parse_telemetry_csv_data_line: header compared after
#                      trimming (so CRLF is accepted, as the Rust loader
#                      accepts it), and at least one row must have 5 fields —
#                      a u64 timestamp and 4 finite floats. Duplicating those
#                      predicates in awk is bounded because CLAUDE.md freezes
#                      this CSV schema; if that ever changes, the real fix is
#                      a strict mode in resolve_telemetry_from, not more shell.
#
# Full SAAQ 1.5 MoE baseline campaign (2 phases x REPEAT_COUNT runs per model).
saaq-campaign:
    @[ -n "${LINEUP_CONFIG:-}" ] && [ -f "${LINEUP_CONFIG}" ] || { echo "error: saaq-campaign requires LINEUP_CONFIG to name an existing lineup. A baseline campaign must pin its model set, and autodiscovery includes the dense glm46v_flash_q8_0, which aborts the sweep on missing expert_count. Copy configs/local_gguf_lineup.template.toml and set LINEUP_CONFIG in .env.local." >&2; exit 1; }
    @[ -n "${TELEMETRY_CSV_PATH:-}" ] && [ -f "${TELEMETRY_CSV_PATH}" ] || { echo "error: phase 2/2 needs TELEMETRY_CSV_PATH to point at an existing CSV. Without it the runner degrades to synthetic telemetry (stamped synthetic_fallback) and this campaign would compare synthetic against synthetic." >&2; exit 1; }
    @awk -v HDR='timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w' 'BEGIN{FS=","} {line=$0; sub(/\r$/,"",line); gsub(/^[ \t]+|[ \t]+$/,"",line)} NR==1{if(line!=HDR){hdrbad=1;exit 2} next} line==""{next} {if(split(line,f,",")!=5)next; if(f[1] !~ /^[0-9]+$/)next; bad=0; for(i=2;i<=5;i++) if(f[i] !~ /^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$/) bad=1; if(bad)next; found=1; exit 0} END{if(hdrbad)exit 2; if(!found)exit 3}' "${TELEMETRY_CSV_PATH}" && rc=0 || rc=$?; [ "$rc" = 0 ] || { [ "$rc" = 2 ] && echo "error: TELEMETRY_CSV_PATH header is not the canonical schema (timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w)." >&2 || echo "error: TELEMETRY_CSV_PATH contains no row the runner would accept (needs 5 fields: u64 timestamp + 4 finite floats). Every row is blank or malformed, so the runner would skip them all and degrade to synthetic." >&2; exit 1; }
    @echo ">>> phase 1/2: synthetic baseline, repeat=2"
    SAAQ_RULE=saaq_v1_5 REPEAT_COUNT=2 TELEMETRY_SOURCE=synthetic \
        RUN_TAG=campaign_syn \
        cargo run --release --example saaq_latent_calibration
    @echo ">>> phase 2/2: csv replay baseline, repeat=2"
    SAAQ_RULE=saaq_v1_5 REPEAT_COUNT=2 TELEMETRY_SOURCE=csv \
        RUN_TAG=campaign_csv \
        cargo run --release --example saaq_latent_calibration
    @echo "ok: campaign finished; see artifacts/index.csv"

# Force CSV-replay mode for the SAAQ sweep. TELEMETRY_CSV_PATH must be set
# in the environment or passed explicitly:
#   just saaq-csv TELEMETRY_CSV_PATH=/path/to/telemetry.csv
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
