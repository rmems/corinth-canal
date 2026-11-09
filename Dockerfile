ARG CUDA_VERSION=13.3.0
ARG CUDAHOSTCXX=/usr/bin/clang++

# Build stage
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder

ARG RUST_VERSION=stable
ARG CUDAHOSTCXX
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    cmake \
    clang \
    llvm \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*
ENV CUDAHOSTCXX=${CUDAHOSTCXX}

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh && \
    sh /tmp/rustup.sh -y --default-toolchain ${RUST_VERSION} && \
    rm /tmp/rustup.sh
ENV PATH=/root/.cargo/bin:$PATH

WORKDIR /app
COPY . .

RUN cargo build --release --features cuda --example gpu_smoke_test

# Runtime stage
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

WORKDIR /app
COPY --from=builder /app/target/release/examples/gpu_smoke_test /app/

ENTRYPOINT ["/app/gpu_smoke_test"]
