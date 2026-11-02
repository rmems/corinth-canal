# Sentry Validation Framework Plan

> **Status: unimplemented proposal, kept for reference (added alongside #52).** Only the "Current Status (Phase 2A)" section describes real code. There is no `src/sentry/`, no `validate-*` or `check-reproducibility` binary, no `schemas/` directory and none of the listed fixtures. The `jsonschema` and `csv` crates under Dependencies are not in `Cargo.toml` and would need the justification `AGENTS.md` requires for any new dependency. The version floors below are also out of date: `.github/workflows/gpu-tests.yml` requires CUDA 13.2 and sm_120, so treat that workflow as the source of truth.

## Overview

This document outlines the comprehensive Sentry validation framework to be implemented in Phase 2B. This framework will provide robust validation, testing, and reproducibility guarantees for the entire observability pipeline.

## Current Status (Phase 2A - Complete)

✅ **GPU Launch Failure Telemetry**
- Sentry capture for CUDA kernel launch failures
- 10 instrumented failure capture sites
- Integration tests in `tests/gpu_sentry_telemetry.rs`
- Documentation in `docs/ARCHITECTURE.md`

## Phase 2B: Comprehensive Validation Framework

### Architecture

```
src/sentry/
├── mod.rs                  # Public API and module coordination
├── input.rs                # Input validation (telemetry CSV, configs)
├── config.rs               # Configuration validation and schema checks
├── runtime.rs              # Runtime validation (GPU state, memory)
├── output.rs               # Output validation (manifests, summaries)
└── reproducibility.rs      # Determinism checks and replay validation

tests/fixtures/
├── telemetry_good.csv              # Valid baseline telemetry
├── telemetry_missing_header.csv    # Missing CSV header
├── telemetry_nan.csv               # NaN values in data
├── telemetry_nonmonotonic.csv      # Non-monotonic timestamps
├── telemetry_wrong_columns.csv     # Incorrect column count
├── telemetry_negative_values.csv   # Invalid negative values
├── config_valid.toml               # Valid configuration
├── config_missing_required.toml    # Missing required fields
├── config_invalid_types.toml       # Type mismatches
└── manifest_examples/
    ├── valid_manifest.json
    ├── invalid_schema_manifest.json
    └── incomplete_manifest.json

schemas/
├── manifest.schema.json    # JSON schema for run manifests
├── summary.schema.json     # JSON schema for run summaries
└── telemetry.schema.json   # JSON schema for telemetry metadata
```

### Module Responsibilities

#### 1. `src/sentry/input.rs` - Input Validation

**Purpose**: Validate all inputs before processing

**Functions**:
```rust
pub fn validate_telemetry_csv(path: &Path) -> ValidationResult<TelemetryMetadata>
pub fn validate_config_toml(path: &Path) -> ValidationResult<ConfigMetadata>
pub fn validate_checkpoint_path(path: &Path) -> ValidationResult<CheckpointMetadata>
```

**Checks**:
- CSV header matches canonical schema
- All required columns present
- No NaN or Inf values
- Timestamps are monotonic
- Values within expected ranges
- File permissions and accessibility

#### 2. `src/sentry/config.rs` - Configuration Validation

**Purpose**: Validate runtime configuration against schemas

**Functions**:
```rust
pub fn validate_model_config(config: &ModelConfig) -> ValidationResult<()>
pub fn validate_run_profile(profile: &RunProfile) -> ValidationResult<()>
pub fn check_config_compatibility(config: &ModelConfig, checkpoint: &Path) -> ValidationResult<()>
```

**Checks**:
- Required fields present
- Type correctness
- Value ranges (e.g., ticks > 0)
- Path existence and accessibility
- Model family compatibility
- GPU requirements vs availability

#### 3. `src/sentry/runtime.rs` - Runtime Validation

**Purpose**: Validate runtime state and resource availability

**Functions**:
```rust
pub fn validate_gpu_state() -> ValidationResult<GpuState>
pub fn validate_memory_availability(required: usize) -> ValidationResult<()>
pub fn validate_cuda_version() -> ValidationResult<CudaVersion>
pub fn check_driver_compatibility() -> ValidationResult<()>
```

**Checks**:
- GPU availability and capability
- CUDA driver version ≥ 570
- CUDA toolkit version ≥ 12.8
- Sufficient device memory
- Compute capability (sm_120 for Blackwell)

#### 4. `src/sentry/output.rs` - Output Validation

**Purpose**: Validate generated outputs against schemas

**Functions**:
```rust
pub fn validate_manifest(path: &Path) -> ValidationResult<()>
pub fn validate_summary(path: &Path) -> ValidationResult<()>
pub fn validate_latent_csv(path: &Path) -> ValidationResult<()>
pub fn check_output_completeness(run_dir: &Path) -> ValidationResult<Vec<String>>
```

**Checks**:
- JSON schema compliance
- Required fields present
- Cross-references valid (e.g., run_id consistency)
- File sizes reasonable
- CSV row counts match expected
- No truncated files

#### 5. `src/sentry/reproducibility.rs` - Reproducibility Validation

**Purpose**: Ensure deterministic execution and replay capability

**Functions**:
```rust
pub fn validate_repeat_determinism(runs: &[RunSummary]) -> ValidationResult<()>
pub fn compare_latent_csvs(path1: &Path, path2: &Path) -> ValidationResult<ComparisonReport>
pub fn verify_replay_fidelity(original: &Path, replay: &Path) -> ValidationResult<()>
```

**Checks**:
- Byte-identical latent CSVs across repeats
- Consistent routing decisions
- Stable SAAQ scores
- Reproducible expert selections

### Validation Report Structure

```rust
pub struct ValidationReport {
    pub timestamp: SystemTime,
    pub run_id: String,
    pub validation_status: ValidationStatus,
    pub checks: Vec<ValidationCheck>,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
}

pub enum ValidationStatus {
    Pass,
    PassWithWarnings,
    Fail,
}

pub struct ValidationCheck {
    pub name: String,
    pub category: CheckCategory,
    pub status: CheckStatus,
    pub message: Option<String>,
    pub duration_ms: u64,
}
```

### Exit Code Discipline

Standardized exit codes for CI/CD integration:

```rust
pub enum ExitCode {
    Success = 0,
    ValidationFailed = 1,
    ConfigError = 2,
    GpuError = 3,
    IoError = 4,
    RepeatMismatch = 5,
    SchemaViolation = 6,
}
```

### Local Verification Commands

```bash
# Validate telemetry CSV
cargo run --bin validate-telemetry -- telemetry.csv

# Validate configuration
cargo run --bin validate-config -- config.toml

# Validate run outputs
cargo run --bin validate-run -- artifacts/run_xyz/

# Full validation suite
cargo run --bin validate-all -- --run-dir artifacts/run_xyz/

# Reproducibility check
cargo run --bin check-reproducibility -- \
  artifacts/run_xyz_repeat_0/ \
  artifacts/run_xyz_repeat_1/
```

### GitHub Actions Integration

```yaml
name: Validation Smoke Test

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      
      - name: Run validation tests
        run: cargo test --test validation_suite
      
      - name: Validate test fixtures
        run: |
          cargo run --bin validate-telemetry -- tests/fixtures/telemetry_good.csv
          ! cargo run --bin validate-telemetry -- tests/fixtures/telemetry_nan.csv
      
      - name: Schema validation
        run: cargo test --test schema_validation
```

### JSON Schemas

#### `schemas/manifest.schema.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Run Manifest",
  "type": "object",
  "required": ["run_id", "model_slug", "validation_status"],
  "properties": {
    "run_id": { "type": "string", "pattern": "^[a-zA-Z0-9_-]+$" },
    "model_slug": { "type": "string" },
    "validation_status": { "enum": ["completed", "failed"] },
    "ticks": { "type": "integer", "minimum": 1 },
    "error": { "type": ["string", "null"] }
  }
}
```

#### `schemas/summary.schema.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Run Summary",
  "type": "object",
  "required": ["run_id", "model_slug", "validation_status", "metrics"],
  "properties": {
    "run_id": { "type": "string" },
    "model_slug": { "type": "string" },
    "validation_status": { "type": "string" },
    "metrics": {
      "type": "object",
      "required": ["ticks_completed", "latent_rows"],
      "properties": {
        "ticks_completed": { "type": "integer", "minimum": 0 },
        "latent_rows": { "type": "integer", "minimum": 0 }
      }
    }
  }
}
```

### Test Fixtures

#### `tests/fixtures/telemetry_good.csv`
```csv
timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w
1000,45.0,150.0,55.0,65.0
2000,46.0,155.0,56.0,66.0
3000,47.0,160.0,57.0,67.0
```

#### `tests/fixtures/telemetry_nan.csv`
```csv
timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w
1000,NaN,150.0,55.0,65.0
2000,46.0,155.0,56.0,66.0
```

#### `tests/fixtures/telemetry_nonmonotonic.csv`
```csv
timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w
1000,45.0,150.0,55.0,65.0
3000,47.0,160.0,57.0,67.0
2000,46.0,155.0,56.0,66.0
```

### Implementation Priority

1. **High Priority** (Week 1):
   - `input.rs` - CSV and config validation
   - `output.rs` - Manifest and summary validation
   - JSON schemas
   - Basic test fixtures

2. **Medium Priority** (Week 2):
   - `runtime.rs` - GPU state validation
   - `reproducibility.rs` - Determinism checks
   - Validation CLI tools
   - Extended test fixtures

3. **Low Priority** (Week 3):
   - GitHub Actions integration
   - Validation report generation
   - Performance benchmarks
   - Documentation and examples

### Success Criteria

- ✅ All test fixtures validate correctly
- ✅ Schema violations detected and reported
- ✅ Exit codes properly propagated
- ✅ CI/CD integration functional
- ✅ Reproducibility checks pass
- ✅ Documentation complete

### Dependencies

- `jsonschema` crate for JSON schema validation
- `csv` crate for CSV parsing and validation
- `serde_json` for JSON manipulation
- Existing `sentry` integration for error reporting

### Migration Path

1. Implement validation framework in `src/sentry/`
2. Add test fixtures and schemas
3. Create validation CLI tools
4. Integrate into existing examples
5. Add GitHub Actions workflow
6. Update documentation

## Related Issues

- Issue #43: GPU launch failure telemetry (Phase 1 - Complete)
- Future: Comprehensive validation framework (Phase 2B)

## References

- [JSON Schema Specification](https://json-schema.org/)
- [CSV RFC 4180](https://tools.ietf.org/html/rfc4180)
- [Sentry Error Tracking](https://docs.sentry.io/)
