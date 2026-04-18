use corinth_canal::{
    FunnelActivity, HybridConfig, HybridModel, OLMoE, OlmoeExecutionMode, ProjectionMode,
    SnnLatentCalibrator, SnnLatentCsvExporter, TelemetrySnapshot,
    encode_prompt_with_tokenizer_json, gpu::GpuAccelerator,
};
use serde::Serialize;
use std::fs::{self, File};
use std::io::{BufWriter, Error, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_PROMPT_SLUG: &str = "math_logic";
const DEFAULT_PROMPT_TEXT: &str = "The derivative of a constant is mathematically zero.";
const DEFAULT_RUN_TAG: &str = "run01";
const DEFAULT_TICKS: usize = 10_000;
const DEFAULT_GPU_SYNAPSE_TENSOR_NAME: &str = "blk.0.attn_q.weight";
const DEFAULT_TOKEN_IDS: [usize; 9] = [402, 11492, 286, 257, 4568, 318, 12056, 4202, 13];

#[derive(Debug, Clone)]
struct ModelInputs {
    gguf_path: String,
    local_checkpoint_dir: String,
    active_model_path: String,
}

#[derive(Debug, Serialize)]
struct RunManifest {
    model_name: String,
    model_slug: String,
    model_path: String,
    local_checkpoint_dir: Option<String>,
    checkpoint_format: String,
    model_architecture: String,
    checkpoint_hidden_size: usize,
    snn_hidden_size: usize,
    checkpoint_num_experts: usize,
    checkpoint_expert_used_count: Option<usize>,
    top_k_experts: usize,
    routing_tensor_name: String,
    routing_source: String,
    recurrent_synapse_source: String,
    quantization: String,
    prompt_slug: String,
    prompt_text: String,
    prompt_token_ids: Vec<usize>,
    prompt_token_source: String,
    tokenizer_path: Option<String>,
    run_tag: String,
    run_id: String,
    ticks: usize,
    started_at_unix_ms: u64,
    generated_files: Vec<String>,
    expected_plot_path: String,
}

fn env_or(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        std::env::var(name)
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
    })
}

fn resolve_model_inputs() -> Result<ModelInputs, Box<dyn std::error::Error>> {
    let gguf_path = env_or(&["MOE_GGUF_PATH", "OLMOE_PATH"]).unwrap_or_default();
    let local_checkpoint_dir =
        env_or(&["MOE_LOCAL_CHECKPOINT_DIR", "LOCAL_CHECKPOINT_DIR"]).unwrap_or_default();

    if gguf_path.is_empty() && local_checkpoint_dir.is_empty() {
        return Err(Error::other(
            "set MOE_GGUF_PATH/OLMOE_PATH for GGUF or MOE_LOCAL_CHECKPOINT_DIR/LOCAL_CHECKPOINT_DIR for a local safetensors checkpoint",
        )
        .into());
    }

    let active_model_path = if !local_checkpoint_dir.is_empty() {
        local_checkpoint_dir.clone()
    } else {
        gguf_path.clone()
    };

    Ok(ModelInputs {
        gguf_path,
        local_checkpoint_dir,
        active_model_path,
    })
}

fn resolve_tokenizer_path(inputs: &ModelInputs) -> Option<PathBuf> {
    env_or(&["TOKENIZER_JSON_PATH"])
        .map(PathBuf::from)
        .or_else(|| {
            if inputs.local_checkpoint_dir.is_empty() {
                return None;
            }
            let path = PathBuf::from(&inputs.local_checkpoint_dir).join("tokenizer.json");
            path.exists().then_some(path)
        })
}

fn default_output_root() -> PathBuf {
    PathBuf::from("outputs/routing")
}

fn slugify(value: &str) -> String {
    let mut slug = String::with_capacity(value.len());
    let mut last_was_sep = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            last_was_sep = false;
        } else if !last_was_sep {
            slug.push('_');
            last_was_sep = true;
        }
    }
    slug.trim_matches('_').to_owned()
}

fn parse_usize_env(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    match env_or(&[name]) {
        Some(value) => Ok(value.parse::<usize>()?),
        None => Ok(default),
    }
}

fn parse_token_ids_csv(raw: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    raw.split(',')
        .map(|part| part.trim().parse::<usize>().map_err(|err| err.into()))
        .collect()
}

fn resolve_prompt_token_ids(
    prompt_text: &str,
    inputs: &ModelInputs,
) -> Result<(Vec<usize>, String, Option<PathBuf>), Box<dyn std::error::Error>> {
    if let Some(raw) = env_or(&["PROMPT_TOKEN_IDS"]) {
        return Ok((
            parse_token_ids_csv(&raw)?,
            "explicit-token-ids".into(),
            resolve_tokenizer_path(inputs),
        ));
    }

    if let Some(tokenizer_path) = resolve_tokenizer_path(inputs) {
        let token_ids = encode_prompt_with_tokenizer_json(&tokenizer_path, prompt_text)?;
        return Ok((token_ids, "tokenizer-json".into(), Some(tokenizer_path)));
    }

    if !inputs.local_checkpoint_dir.is_empty() {
        return Err(Error::other(
            "PROMPT_TOKEN_IDS was not supplied and tokenizer.json was not found for the local checkpoint",
        )
        .into());
    }

    Ok((DEFAULT_TOKEN_IDS.to_vec(), "default-token-ids".into(), None))
}

fn prepare_run_dir(
    root: &Path,
    model_slug: &str,
    prompt_slug: &str,
    run_id: &str,
) -> std::io::Result<PathBuf> {
    let run_dir = root.join(model_slug).join(prompt_slug).join(run_id);
    fs::create_dir_all(&run_dir)?;
    Ok(run_dir)
}

fn writer_for(path: &Path) -> Result<BufWriter<File>, Box<dyn std::error::Error>> {
    Ok(BufWriter::new(File::create(path)?))
}

fn default_model_slug(active_model_path: &str) -> String {
    let path = Path::new(active_model_path);
    let inferred = if path.is_dir() {
        path.file_name()
    } else {
        path.file_stem()
    };
    inferred
        .and_then(|value| value.to_str())
        .map(slugify)
        .unwrap_or_else(|| "model".to_owned())
}

fn model_name(active_model_path: &str) -> String {
    Path::new(active_model_path)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("checkpoint")
        .to_owned()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_inputs = resolve_model_inputs()?;
    let checkpoint = OLMoE::probe_checkpoint_with_local(
        &model_inputs.gguf_path,
        &model_inputs.local_checkpoint_dir,
    )?;
    let ticks = parse_usize_env("TICKS", DEFAULT_TICKS)?;
    let prompt_text = env_or(&["PROMPT_TEXT"]).unwrap_or_else(|| DEFAULT_PROMPT_TEXT.to_owned());
    let prompt_slug = env_or(&["PROMPT_SLUG"]).unwrap_or_else(|| DEFAULT_PROMPT_SLUG.to_owned());
    let (prompt_token_ids, prompt_token_source, tokenizer_path) =
        resolve_prompt_token_ids(&prompt_text, &model_inputs)?;
    if prompt_token_ids.is_empty() {
        return Err(Error::other("resolved prompt token IDs are empty").into());
    }
    let run_tag = env_or(&["RUN_TAG"]).unwrap_or_else(|| DEFAULT_RUN_TAG.to_owned());
    let model_slug = env_or(&["MODEL_SLUG"])
        .unwrap_or_else(|| default_model_slug(&model_inputs.active_model_path));
    let output_root = env_or(&["RUN_OUTPUT_ROOT", "RESEARCH_OUTPUT_ROOT"])
        .map(PathBuf::from)
        .unwrap_or_else(default_output_root);
    let num_experts = parse_usize_env("NUM_EXPERTS", checkpoint.num_experts)?;
    let top_k_experts = parse_usize_env(
        "TOP_K_EXPERTS",
        checkpoint.expert_used_count.unwrap_or(1).max(1),
    )?
    .min(num_experts);
    let started_at_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
    let run_id = format!(
        "{}_{}_{}_{}",
        started_at_unix_ms / 1000,
        model_slug,
        slugify(&prompt_slug),
        slugify(&run_tag)
    );
    let run_dir = prepare_run_dir(&output_root, &model_slug, &prompt_slug, &run_id)?;
    let tick_path = run_dir.join("tick_telemetry.txt");
    let latent_path = run_dir.join("latent_telemetry.csv");
    let manifest_path = run_dir.join("run_manifest.json");
    let plot_path = run_dir.join("routing_map.png");

    let mut tick_writer = writer_for(&tick_path)?;
    let mut latent_exporter = SnnLatentCsvExporter::create(&latent_path)?;
    let mut calibrator = SnnLatentCalibrator::new();

    let mut accelerator = GpuAccelerator::new();
    let mut model = HybridModel::new(HybridConfig {
        olmoe_model_path: model_inputs.gguf_path.clone(),
        local_checkpoint_dir: model_inputs.local_checkpoint_dir.clone(),
        gpu_synapse_tensor_name: env_or(&["GPU_SYNAPSE_TENSOR_NAME"])
            .unwrap_or_else(|| DEFAULT_GPU_SYNAPSE_TENSOR_NAME.to_owned()),
        num_experts,
        top_k_experts,
        olmoe_execution_mode: OlmoeExecutionMode::SpikingSim,
        snn_steps: 1,
        projection_mode: ProjectionMode::SpikingTernary,
    })?;

    if !model.olmoe_loaded() {
        return Err(Error::other(
            "checkpoint did not load from the configured GGUF or local directory source",
        )
        .into());
    }

    let target_neurons = model.projector_mut().input_neurons();
    let mut pooled = vec![0.0f32; target_neurons];
    for &token in &prompt_token_ids {
        let emb = model.extract_token_embedding(token)?;
        if emb.len() != target_neurons {
            return Err(Error::other(format!(
                "token embedding length mismatch: expected {target_neurons}, got {}",
                emb.len()
            ))
            .into());
        }
        for (dst, src) in pooled.iter_mut().zip(emb.iter()) {
            *dst += *src;
        }
    }
    for value in &mut pooled {
        *value /= prompt_token_ids.len() as f32;
    }
    let l2_norm = pooled.iter().map(|&v| v * v).sum::<f32>().sqrt();
    if l2_norm > 1e-8 {
        for value in &mut pooled {
            *value /= l2_norm;
        }
    }

    let checkpoint_format = model.checkpoint_format().to_owned();
    let checkpoint_architecture = model.checkpoint_architecture().to_owned();
    let checkpoint_hidden_size = model.checkpoint_source_hidden_size();
    let snn_hidden_size = model.checkpoint_hidden_size();
    let checkpoint_num_experts = model.checkpoint_num_experts();
    let checkpoint_expert_used_count = model.checkpoint_expert_used_count();
    let routing_tensor_name = model.routing_tensor_name().to_owned();
    let routing_source = model.routing_source().to_owned();

    let write_manifest =
        |recurrent_synapse_source: &str| -> Result<(), Box<dyn std::error::Error>> {
            let manifest = RunManifest {
                model_name: model_name(&model_inputs.active_model_path),
                model_slug: model_slug.clone(),
                model_path: model_inputs.active_model_path.clone(),
                local_checkpoint_dir: (!model_inputs.local_checkpoint_dir.is_empty())
                    .then_some(model_inputs.local_checkpoint_dir.clone()),
                checkpoint_format: checkpoint_format.clone(),
                model_architecture: checkpoint_architecture.clone(),
                checkpoint_hidden_size,
                snn_hidden_size,
                checkpoint_num_experts,
                checkpoint_expert_used_count,
                top_k_experts,
                routing_tensor_name: routing_tensor_name.clone(),
                routing_source: routing_source.clone(),
                recurrent_synapse_source: recurrent_synapse_source.to_owned(),
                quantization: checkpoint.quantization.clone(),
                prompt_slug: prompt_slug.clone(),
                prompt_text: prompt_text.clone(),
                prompt_token_ids: prompt_token_ids.clone(),
                prompt_token_source: prompt_token_source.clone(),
                tokenizer_path: tokenizer_path
                    .clone()
                    .map(|path| path.display().to_string()),
                run_tag: run_tag.clone(),
                run_id: run_id.clone(),
                ticks,
                started_at_unix_ms,
                generated_files: vec![
                    "tick_telemetry.txt".into(),
                    "latent_telemetry.csv".into(),
                    "run_manifest.json".into(),
                ],
                expected_plot_path: plot_path.display().to_string(),
            };
            fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;
            Ok(())
        };

    write_manifest("pending-gpu-prepare")?;
    if !accelerator.is_ready() {
        return Err(Error::other("GpuAccelerator is not ready").into());
    }

    model.prepare_gpu_temporal(&mut accelerator)?;
    write_manifest(model.recurrent_synapse_source())?;
    println!(
        "startup model_path={} checkpoint_format={} architecture={} checkpoint_hidden_size={} snn_hidden_size={} checkpoint_experts={} top_k={} routing_tensor={} routing_source={} recurrent_synapse_source={} prompt_token_source={} gpu_ready={} output_dir={}",
        model_inputs.active_model_path,
        model.checkpoint_format(),
        model.checkpoint_architecture(),
        model.checkpoint_source_hidden_size(),
        model.checkpoint_hidden_size(),
        model.checkpoint_num_experts(),
        top_k_experts,
        model.routing_tensor_name(),
        model.routing_source(),
        model.recurrent_synapse_source(),
        prompt_token_source,
        accelerator.is_ready(),
        run_dir.display(),
    );

    let zero_iz = vec![0.0f32; 5];
    for tick in 0..ticks {
        let started = Instant::now();
        let best_walker = model.tick_gpu_temporal(&mut accelerator, &pooled)?;
        let elapsed_us = started.elapsed().as_micros();
        writeln!(
            tick_writer,
            "tick={} best_walker={} elapsed_us={}",
            tick + 1,
            best_walker,
            elapsed_us
        )?;

        let spikes = accelerator.temporal_spikes_to_vec(target_neurons)?;
        let membrane = accelerator.temporal_membrane_to_vec(target_neurons)?;
        let active_neurons: Vec<usize> = spikes
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != 0)
            .map(|(idx, _)| idx)
            .collect();
        let potentials: Vec<f32> = membrane
            .iter()
            .map(|&value| value.clamp(0.0, 1.0))
            .collect();
        let activity = FunnelActivity {
            ternary_events: [0, 0, 0, 0],
            input_spike_train: vec![Vec::new()],
            spike_train: vec![active_neurons],
            potentials: potentials.clone(),
            iz_potentials: zero_iz.clone(),
        };
        let output = model.forward_activity(
            &activity.spike_train,
            &activity.potentials,
            &activity.iz_potentials,
        )?;
        let snap = TelemetrySnapshot {
            timestamp_ms: started_at_unix_ms + tick as u64,
            ..TelemetrySnapshot::default()
        };
        let latent = calibrator.observe(&snap, &activity, &output)?;
        latent_exporter.write_row(&latent)?;
    }

    tick_writer.flush()?;
    latent_exporter.flush()?;
    write_manifest(model.recurrent_synapse_source())?;

    println!("saved tick telemetry to {}", tick_path.display());
    println!("saved latent telemetry to {}", latent_path.display());
    println!("saved manifest to {}", manifest_path.display());
    println!(
        "to generate the plot: julia plot_latent_space.jl {} {}",
        tick_path.display(),
        plot_path.display()
    );

    drop(model);
    drop(accelerator);

    Ok(())
}
