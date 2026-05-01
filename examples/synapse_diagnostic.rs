//! Diagnostic pass for issue #31.
//!
//! Loads each validation model in the configured lineup (LINEUP_CONFIG, then
//! GGUF_CHECKPOINT_PATH, then autodiscovery — same precedence as the SAAQ
//! runner) and prints, per model, the GGUF facts that drive the
//! synthetic-vs-real synapse decision in
//! `src/moe/adapter.rs::resolve_adapter`. Writes a JSON report to
//! `<output_root>/synapse_diagnostic.json`.
//!
//! No SAAQ ticks, no heartbeat, no GPU bring-up, no campaign side effects.

mod support;

use std::fs;
use std::io::{BufWriter, Write};

use serde::Serialize;

use corinth_canal::ModelFamily;
use corinth_canal::moe::{GpuSynapseTensorDescriptor, OlmoeRouter, RoutingMode};
use support::ValidationModelSpec;
use support::config::RunConfig;

#[derive(Debug, Serialize)]
struct SynapseDiagnosticRow {
    model_slug: String,
    checkpoint_path: String,
    family: Option<String>,
    architecture: Option<String>,
    quantization: Option<String>,
    preferred_gpu_synapse_tensor_name: Option<String>,
    real_gpu_synapse_tensor_name: Option<String>,
    preferred_tensor_dims: Option<Vec<usize>>,
    preferred_tensor_ggml_type_id: Option<u32>,
    preferred_tensor_ggml_type_label: Option<&'static str>,
    synapse_source: Option<String>,
    real_f16_available: bool,
    dequant_supported_by_current_code: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn probe_one(
    spec: &ValidationModelSpec,
    model_family_override: Option<ModelFamily>,
) -> SynapseDiagnosticRow {
    let mut row = SynapseDiagnosticRow {
        model_slug: spec.slug.clone(),
        checkpoint_path: spec.path.clone(),
        family: None,
        architecture: None,
        quantization: None,
        preferred_gpu_synapse_tensor_name: None,
        real_gpu_synapse_tensor_name: None,
        preferred_tensor_dims: None,
        preferred_tensor_ggml_type_id: None,
        preferred_tensor_ggml_type_label: None,
        synapse_source: None,
        real_f16_available: false,
        dequant_supported_by_current_code: false,
        error: None,
    };

    // StubUniform routing keeps the load cheap: no gate-score precompute, no
    // membrane state, no GPU bring-up. We only need the metadata + checkpoint
    // map that `OlmoeRouter::load_with_family_and_mode` already builds. Apply
    // the same `model_family_override.or(spec.family)` precedence as the SAAQ
    // runner (`run_validation` in `examples/saaq_latent_calibration.rs`) so
    // the diagnostic agrees with production routing on lineups that rely on
    // a global `MODEL_FAMILY` override.
    let effective_family = model_family_override.or(spec.family);
    match OlmoeRouter::load_with_family_and_mode(
        &spec.path,
        0,
        0,
        effective_family,
        RoutingMode::StubUniform,
    ) {
        Ok(router) => {
            row.family = Some(format!("{:?}", router.family()));
            row.architecture = Some(router.architecture().to_owned());
            row.quantization = Some(router.quantization().to_owned());
            row.preferred_gpu_synapse_tensor_name = router
                .preferred_gpu_synapse_tensor_name()
                .map(str::to_owned);
            row.real_gpu_synapse_tensor_name =
                router.real_gpu_synapse_tensor_name().map(str::to_owned);
            row.synapse_source = Some(router.synapse_source().to_owned());
            row.real_f16_available = router.real_gpu_synapse_tensor_name().is_some();

            if let Some(descriptor) = router.preferred_gpu_synapse_tensor_descriptor() {
                let GpuSynapseTensorDescriptor {
                    name: _,
                    ggml_type_id,
                    ggml_type_label,
                    dims,
                    has_dequant_path,
                } = descriptor;
                row.preferred_tensor_dims = Some(dims);
                row.preferred_tensor_ggml_type_id = Some(ggml_type_id);
                row.preferred_tensor_ggml_type_label = Some(ggml_type_label);
                row.dequant_supported_by_current_code = has_dequant_path;
            }
        }
        Err(err) => {
            row.error = Some(err.to_string());
        }
    }

    row
}

fn print_row(row: &SynapseDiagnosticRow) {
    let preferred = row
        .preferred_gpu_synapse_tensor_name
        .as_deref()
        .unwrap_or("-");
    let real = row.real_gpu_synapse_tensor_name.as_deref().unwrap_or("-");
    let ggml_label = row.preferred_tensor_ggml_type_label.unwrap_or("-");
    let dims = row
        .preferred_tensor_dims
        .as_ref()
        .map(|d| format!("{:?}", d))
        .unwrap_or_else(|| "-".to_owned());
    let synapse_source = row.synapse_source.as_deref().unwrap_or("-");
    let family = row.family.as_deref().unwrap_or("-");
    let arch = row.architecture.as_deref().unwrap_or("-");

    println!(
        "[{slug:<32}] family={family:<10} arch={arch:<10} preferred={preferred:<22} \
         type={ggml_label:<7} dims={dims:<14} real={real:<24} \
         source={synapse_source:<20} real_f16={real_f16:<5} dequant={dequant}",
        slug = row.model_slug,
        family = family,
        arch = arch,
        preferred = preferred,
        ggml_label = ggml_label,
        dims = dims,
        real = real,
        synapse_source = synapse_source,
        real_f16 = row.real_f16_available,
        dequant = row.dequant_supported_by_current_code,
    );
    if let Some(err) = row.error.as_deref() {
        println!("    error: {err}");
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::from_filename(".env.local");
    let cfg = RunConfig::from_env();

    fs::create_dir_all(&cfg.output_root)?;

    if cfg.validation_models.is_empty() {
        eprintln!(
            "synapse_diagnostic: no validation models resolved (LINEUP_CONFIG / \
             GGUF_CHECKPOINT_PATH / autodiscovery all returned empty)"
        );
    }

    println!(
        "synapse_diagnostic: probing {} model(s) (output_root={})",
        cfg.validation_models.len(),
        cfg.output_root.display()
    );

    let mut report = Vec::with_capacity(cfg.validation_models.len());
    for spec in &cfg.validation_models {
        let row = probe_one(spec, cfg.model_family_override);
        print_row(&row);
        report.push(row);
    }

    // Stream the JSON report through a buffered writer so peak memory stays
    // bounded as the lineup grows beyond the current 5-model fixture.
    let json_path = cfg.output_root.join("synapse_diagnostic.json");
    let file = fs::File::create(&json_path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &report)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    println!("ok: wrote {}", json_path.display());

    Ok(())
}
