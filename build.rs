use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const PTX_STUB: &str = ".version 8.5\n.target sm_80\n.address_size 64\n";

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cu_dir = manifest_dir.join("src").join("gpu").join("kernels");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=src/gpu/kernels/common.cuh");
    println!("cargo:rerun-if-changed=src/gpu/kernels/spiking_network.cu");
    println!("cargo:rerun-if-changed=src/gpu/kernels/vector_similarity.cu");
    println!("cargo:rerun-if-changed=src/gpu/kernels/satsolver.cu");
    println!("cargo:rerun-if-changed=build.rs");

    let kernels: &[(&str, &str)] = &[
        ("spiking_network.cu", "spiking_network_sm_120.ptx"),
        ("vector_similarity.cu", "vector_similarity_sm_120.ptx"),
        ("satsolver.cu", "satsolver_sm_120.ptx"),
    ];

    let nvcc_ok = Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !nvcc_ok {
        println!(
            "cargo:warning=nvcc not found — writing stub PTX files; GPU unavailable at runtime"
        );
        for &(_, ptx_name) in kernels {
            fs::write(out_dir.join(ptx_name), PTX_STUB)
                .unwrap_or_else(|e| panic!("Failed to write stub PTX {ptx_name}: {e}"));
        }
        return;
    }

    for &(cu_name, ptx_name) in kernels {
        let source = cu_dir.join(cu_name);
        let output = out_dir.join(ptx_name);

        let status = Command::new("nvcc")
            .arg("-ptx")
            .arg("-arch=sm_120")
            .arg("-O3")
            .arg("--use_fast_math")
            .arg("--restrict")
            .arg("--threads")
            .arg("0")
            .arg("-D__STRICT_ANSI__")
            .arg("--allow-unsupported-compiler")
            .arg("-I")
            .arg(&cu_dir)
            .arg("-o")
            .arg(&output)
            .arg(&source)
            .status()
            .unwrap_or_else(|e| panic!("Failed to invoke nvcc for {cu_name}: {e}"));

        if !status.success() {
            panic!("nvcc failed to compile {cu_name} → {ptx_name}");
        }

        if !output.exists() {
            panic!("nvcc completed but {ptx_name} was not produced");
        }

        patch_ptx_version(&output, "9.1", "8.5");
        patch_ptx_version(&output, "9.0", "8.5");

        println!("cargo:warning=✓ compiled {cu_name} → {ptx_name} (PTX version patched to 8.5)");
    }
}

fn patch_ptx_version(path: &std::path::Path, old_ver: &str, new_ver: &str) {
    let Ok(text) = fs::read_to_string(path) else {
        return;
    };
    let old_directive = format!(".version {old_ver}");
    let new_directive = format!(".version {new_ver}");
    if text.contains(&old_directive) {
        let patched = text.replace(&old_directive, &new_directive);
        fs::write(path, patched)
            .unwrap_or_else(|e| panic!("Failed to patch PTX version in {}: {e}", path.display()));
    }
}
