FROM rust:1.92-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY tests/ tests/
RUN cargo build --locked --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/responses-adapter /usr/local/bin/responses-adapter

EXPOSE 8787

ENTRYPOINT ["responses-adapter"]
