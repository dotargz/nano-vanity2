//! Generates the lookup table that is embedded in the binary, so that short
//! searches can start instantly without any start-up precomputation.

#[allow(dead_code)]
#[path = "src/fe51.rs"]
mod fe51;

#[allow(dead_code)]
#[path = "src/edwards.rs"]
mod edwards;

#[allow(dead_code)]
#[path = "src/tablegen.rs"]
mod tablegen;

use std::io::Write;

/// Window width of the embedded table. Must match `table::STATIC_WINDOW`.
const STATIC_WINDOW: u32 = 8;

fn main() {
    for f in ["build.rs", "src/fe51.rs", "src/edwards.rs", "src/tablegen.rs"] {
        println!("cargo:rerun-if-changed={f}");
    }
    let layout = tablegen::Layout::new(STATIC_WINDOW);
    let data = tablegen::generate(&layout);
    let mut bytes = Vec::with_capacity(data.len() * 8);
    for w in &data {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("static_table.bin");
    std::fs::File::create(out).unwrap().write_all(&bytes).unwrap();
    gpu_kernel();
}

/// Compile the CUDA kernel to PTX if the CUDA toolkit is installed;
/// otherwise use the PTX that is checked in next to it.
fn gpu_kernel() {
    println!("cargo:rerun-if-changed=src/gpu/kernel.cu");
    println!("cargo:rerun-if-changed=src/gpu/kernel.ptx");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("kernel.ptx");
    let prebuilt = std::path::Path::new("src/gpu/kernel.ptx");
    let fresh = |a: &std::path::Path, b: &str| {
        let t = |p: &std::path::Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
        match (t(a), t(std::path::Path::new(b))) {
            (Some(x), Some(y)) => x >= y,
            _ => false,
        }
    };
    if !fresh(prebuilt, "src/gpu/kernel.cu") {
        let nvcc = match std::env::var("CUDA_PATH") {
            Ok(p) => std::path::Path::new(&p).join("bin").join("nvcc"),
            Err(_) => "nvcc".into(),
        };
        let mut cmd = std::process::Command::new(nvcc);
        cmd.args(["-ptx", "-O3", "-arch=sm_75", "src/gpu/kernel.cu", "-o"]).arg(&out);
        // nvcc preprocesses with the host compiler.
        let target = std::env::var("TARGET").unwrap();
        if let Some(cl) = cc::windows_registry::find_tool(&target, "cl.exe") {
            for (k, v) in cl.get_envs() {
                cmd.env(k, v);
            }
        }
        match cmd.status() {
            Ok(s) if s.success() => {
                // Keep the checked-in copy current for machines without nvcc.
                let _ = std::fs::copy(&out, prebuilt);
                return;
            }
            _ => println!("cargo:warning=could not run nvcc; using the prebuilt src/gpu/kernel.ptx, which is older than kernel.cu"),
        }
    }
    std::fs::copy(prebuilt, &out).expect("src/gpu/kernel.ptx is missing");
}
