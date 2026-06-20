# morsel-bake

A CLI that turns a trained model's safetensors file into a standalone Rust source file — one `pub const` array per tensor — so the weights compile straight into a binary with no runtime model load.

## Why it exists

The [`morsel`](https://github.com/j0yen/morsel) approach to tiny neural networks is to bake the trained weights into the consumer crate as `const` arrays. That only works if something writes those arrays. The alternative is to hand-transcribe them from a numpy dump, which is tedious and a fresh chance to flip a sign every time the model retrains.

`morsel-bake` is that something. You train a model wherever you train models, export it to safetensors, and run one command. Out comes a `.rs` file you commit into the consumer crate. The format and emission shape are deliberately fixed first, so a model author can rely on the output before any runtime primitive is built against it.

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/j0yen/morsel-bake/main/install.sh | bash
```

Or from a clone:

```sh
git clone --depth 1 https://github.com/j0yen/morsel-bake.git
cd morsel-bake
./install.sh
```

`install.sh` runs `cargo install --path . --locked`, so the `morsel-bake` binary lands in `~/.cargo/bin/`. Requires `cargo` / `rustc 1.85+` and `git`. To build without installing, `cargo build --release` puts the binary at `target/release/morsel-bake`.

## First run

```sh
morsel-bake --in model.safetensors --arch lstm_classifier --out weights.rs
```

Reads every tensor in `model.safetensors` and writes `weights.rs`, containing one `pub const` per tensor plus an `ARCH_FINGERPRINT` derived from `--arch`. The output compiles on its own — it pulls in no dependencies, not even `morsel` — so you can drop it into any crate. `--out` will not overwrite an existing file unless you pass `--force`.

```
Usage: morsel-bake [OPTIONS] --in <PATH> --arch <NAME> --out <PATH>

Options:
      --in <PATH>    Input safetensors file
      --arch <NAME>  Architecture name (baked into `ARCH_FINGERPRINT`)
      --out <PATH>   Output Rust source file
      --force        Overwrite the output file if it already exists
```

## How it works

Each tensor becomes a const whose Rust type mirrors its shape: a 1-D tensor emits `[f32; N]`, a matrix `[[f32; N]; M]`, and so on up to rank 4; a scalar emits a bare `f32`. Tensor names that carry `.` or `/` — `weight.ih`, `layer/0/weight` — are sanitized to uppercase Rust identifiers (`WEIGHT_IH`, `LAYER_0_WEIGHT`). Floats are written with enough precision to round-trip bit-identically, so the const you compile is the weight you trained, not a rounded copy.

The exit code tells you what went wrong, and the contract is stable enough that tests pin it:

| Code | Meaning |
| --- | --- |
| 2 | `--in` file does not exist |
| 4 | two tensor names collide after sanitization |
| 5 | a tensor's dtype is not f32 (only f32 is supported) |
| 6 | a tensor's rank exceeds 4 |
| 7 | `--out` already exists and `--force` was not passed |
| 1 | other I/O or safetensors parse error |

## Where it fits

`morsel-bake` is the build-time half of a two-crate pair. [`morsel`](https://github.com/j0yen/morsel) is the runtime half — the scalar inference primitives the baked weights feed. Train elsewhere, bake the weights here, link `morsel`, ship one binary.

## Status

v0.1. Inputs are f32 safetensors only — f16, bf16, and int8 quantization are not handled. Tensors of rank 5 or higher are rejected.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
