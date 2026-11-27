FROM golang:1.24-bookworm AS go
FROM rust:1.88-bookworm AS rust
FROM node:22-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git build-essential python3 python3-pytest python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=go /usr/local/go /usr/local/go
COPY --from=rust /usr/local/rustup/toolchains /opt/rust
RUN ln -s /usr/local/go/bin/go /usr/local/bin/go \
    && ln -s /opt/rust/*/bin/cargo /usr/local/bin/cargo \
    && ln -s /opt/rust/*/bin/rustc /usr/local/bin/rustc \
    && ln -s /opt/rust/*/bin/rustdoc /usr/local/bin/rustdoc

WORKDIR /work
