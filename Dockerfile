# Build stage
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

ARG RUST_VERSION=stable
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    cmake \
    clang \
    llvm \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain ${RUST_VERSION}
ENV PATH=/root/.cargo/bin:$PATH

WORKDIR /app
COPY . .

RUN cargo build --release --features cuda --example gpu_smoke_test

# Runtime stage
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

WORKDIR /app
COPY --from=builder /app/target/release/examples/gpu_smoke_test /app/

ENTRYPOINT ["/app/gpu_smoke_test"]
