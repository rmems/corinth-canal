# Build stage
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS builder

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

RUN cargo build --release --features cuda

# Runtime stage
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

WORKDIR /app
COPY --from=builder /app/target/release/corinth-canal /app/

ENTRYPOINT ["/app/corinth-canal"]
