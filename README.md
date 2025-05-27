<!-- [![`curtana` on crates.io](https://img.shields.io/crates/v/curtana)](https://crates.io/crates/curtana) -->
<!-- [![`curtana` on docs.rs](https://img.shields.io/docsrs/curtana)](https://docs.rs/curtana/) -->
[![`curtana` is MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/with-caer/curtana/blob/main/LICENSE.txt)

Simplified, low-overhead wrapper over [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
powered by the [`llama-cpp-2`](https://github.com/utilityai/llama-cpp-rs/tree/main) crate
supporting most `.gguf` formatted "Chat" and "Embedding" models.

## Testing

1. `wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf`
2. `wget https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf`
3. `cargo test`

## License

Copyright © 2025 With Caer, LLC.

Licensed under the MIT license. Refer to [the license file](https://github.com/with-caer/curtana/blob/main/LICENSE.txt) for more info.