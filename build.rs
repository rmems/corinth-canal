use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const MYELIN_SHIM_STUB: &str = r#"
#include <stdint.h>

int myelin_launch_gif_step_weighted_f16(
    void* stream,
    unsigned int grid_x,
    unsigned int block_x,
    unsigned int shared_bytes,
    void* membrane,
    void* adaptation,
    void* weights,
    void* input_spikes,
    void* refractory,
    void* spikes_out,
    int n_neurons,
    int n_inputs)
{
    (void)stream;
    (void)grid_x;
    (void)block_x;
    (void)shared_bytes;
    (void)membrane;
    (void)adaptation;
    (void)weights;
    (void)input_spikes;
    (void)refractory;
    (void)spikes_out;
    (void)n_neurons;
    (void)n_inputs;
    return 1;
}

int myelin_launch_saaq_find_best_walker(
    void* stream,
    unsigned int grid_x,
    unsigned int block_x,
    unsigned int shared_bytes,
    void* membrane,
    void* adaptation,
    void* partial_scores,
    void* partial_walkers,
    void* best_walker_out,
    int n_neurons,
    float adaptation_scale)
{
    (void)stream;
    (void)grid_x;
    (void)block_x;
    (void)shared_bytes;
    (void)membrane;
    (void)adaptation;
    (void)partial_scores;
    (void)partial_walkers;
    (void)best_walker_out;
    (void)n_neurons;
    (void)adaptation_scale;
    return 1;
}
"#;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cu_dir = manifest_dir.join("src").join("gpu").join("kernels");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=src/gpu/kernels/common.cuh");
    println!("cargo:rerun-if-changed=src/gpu/kernels/spiking_network.cu");
    println!("cargo:rerun-if-changed=src/gpu/kernels/vector_similarity.cu");
    println!("cargo:rerun-if-changed=src/gpu/kernels/satsolver.cu");
    println!("cargo:rerun-if-changed=src/gpu/kernels/myelin_shim.h");
    println!("cargo:rerun-if-changed=src/gpu/kernels/myelin_shim.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=NVCC");

    // Each entry: (CUDA source, output fatbin name embedded by the loader).
    // We deliberately ship fatbin (SASS for sm_120 + PTX for compute_120 as a
    // forward-compat fallback) so that the driver loads precompiled SASS by
    // default and only invokes the JIT when no exact-match SASS is found.
    let kernels: &[(&str, &str)] = &[
        ("spiking_network.cu", "spiking_network_sm_120.fatbin"),
        ("vector_similarity.cu", "vector_similarity_sm_120.fatbin"),
        ("satsolver.cu", "satsolver_sm_120.fatbin"),
    ];

    let stub_enabled = env::var_os("CARGO_FEATURE_GPU_STUB").is_some();

    let nvcc = match find_nvcc() {
        Some(nvcc) => nvcc,
        None => {
            if stub_enabled {
                println!(
                    "cargo:warning=nvcc not found and `gpu-stub` feature enabled — writing empty fatbin files; GPU will be unavailable at runtime"
                );
                for &(_, fatbin_name) in kernels {
                    fs::write(out_dir.join(fatbin_name), &[] as &[u8]).unwrap_or_else(|e| {
                        panic!("Failed to write stub fatbin {fatbin_name}: {e}")
                    });
                }
                build_myelin_shim_stub(&out_dir);
                return;
            }
            panic!(
                "nvcc not found. Install CUDA Toolkit (>= 12.8 for sm_120) or set NVCC=/path/to/nvcc. \
                 To allow building without nvcc and disable GPU at runtime, enable the `gpu-stub` feature."
            );
        }
    };

    for &(cu_name, fatbin_name) in kernels {
        let source = cu_dir.join(cu_name);
        let output = out_dir.join(fatbin_name);

        // -fatbin asks nvcc to emit a fat binary that bundles:
        //   * device-specific SASS for sm_120 (loaded directly, no JIT), and
        //   * PTX for compute_120 (used as a JIT fallback on future archs).
        run_nvcc(
            &nvcc,
            &[
                "-fatbin".into(),
                "-gencode=arch=compute_120,code=sm_120".into(),
                "-gencode=arch=compute_120,code=compute_120".into(),
                "-O3".into(),
                "--use_fast_math".into(),
                "--restrict".into(),
                "--threads".into(),
                "0".into(),
                "-D__STRICT_ANSI__".into(),
                "--allow-unsupported-compiler".into(),
                "-I".into(),
                cu_dir.display().to_string(),
                "-o".into(),
                output.display().to_string(),
                source.display().to_string(),
            ],
            cu_name,
        );

        if !output.exists() {
            panic!("nvcc completed but {fatbin_name} was not produced");
        }

        println!(
            "cargo:warning=✓ compiled {cu_name} → {fatbin_name} (sm_120 SASS + compute_120 PTX)"
        );
    }

    build_myelin_shim(&nvcc, &cu_dir, &out_dir);
    emit_cuda_runtime_linking(&nvcc);
}

fn find_nvcc() -> Option<PathBuf> {
    if let Some(path) = env::var_os("NVCC") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }

    Command::new("sh")
        .args(["-lc", "command -v nvcc"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| {
            let path = String::from_utf8(output.stdout).ok()?;
            let trimmed = path.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(PathBuf::from(trimmed))
            }
        })
}

fn run_nvcc(nvcc: &Path, args: &[String], label: &str) {
    let status = Command::new(nvcc)
        .args(args)
        .status()
        .unwrap_or_else(|e| panic!("Failed to invoke nvcc for {label}: {e}"));

    if !status.success() {
        panic!("nvcc failed while building {label}");
    }
}

fn build_myelin_shim(nvcc: &Path, cu_dir: &Path, out_dir: &Path) {
    let source = cu_dir.join("myelin_shim.cu");
    let object = out_dir.join("myelin_shim.o");

    run_nvcc(
        nvcc,
        &[
            "-c".into(),
            "-arch=sm_120".into(),
            "-O3".into(),
            "--use_fast_math".into(),
            "--restrict".into(),
            "--threads".into(),
            "0".into(),
            "--allow-unsupported-compiler".into(),
            "-I".into(),
            cu_dir.display().to_string(),
            "-Xcompiler".into(),
            "-fPIC".into(),
            "-o".into(),
            object.display().to_string(),
            source.display().to_string(),
        ],
        "myelin_shim.cu",
    );

    cc::Build::new()
        .cpp(true)
        .object(&object)
        .compile("myelin_shim");
    println!("cargo:warning=✓ compiled myelin_shim.cu → libmyelin_shim.a");
}

fn build_myelin_shim_stub(out_dir: &Path) {
    let stub_path = out_dir.join("myelin_shim_stub.c");
    fs::write(&stub_path, MYELIN_SHIM_STUB)
        .unwrap_or_else(|e| panic!("Failed to write myelin shim stub: {e}"));

    cc::Build::new().file(&stub_path).compile("myelin_shim");
    println!("cargo:warning=✓ compiled myelin shim stub → libmyelin_shim.a");
}

fn emit_cuda_runtime_linking(nvcc: &Path) {
    for search_dir in cuda_library_search_paths(nvcc) {
        println!("cargo:rustc-link-search=native={}", search_dir.display());
    }
    println!("cargo:rustc-link-lib=dylib=cudart");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

fn cuda_library_search_paths(nvcc: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    for env_var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = env::var_os(env_var) {
            let root = PathBuf::from(root);
            candidates.push(root.join("lib64"));
            candidates.push(root.join("lib"));
            candidates.push(root.join("targets").join("x86_64-linux").join("lib"));
        }
    }

    if let Some(root) = nvcc.parent().and_then(|bin| bin.parent()) {
        candidates.push(root.join("lib64"));
        candidates.push(root.join("lib"));
        candidates.push(root.join("targets").join("x86_64-linux").join("lib"));
    }

    candidates.push(PathBuf::from("/usr/local/cuda/lib64"));
    candidates.push(PathBuf::from("/usr/local/cuda/lib"));
    candidates.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));

    let mut deduped = Vec::new();
    for candidate in candidates {
        if candidate.exists() && !deduped.iter().any(|existing| existing == &candidate) {
            deduped.push(candidate);
        }
    }
    deduped
}
