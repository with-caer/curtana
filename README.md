[![`curtana` on crates.io](https://img.shields.io/crates/v/curtana)](https://crates.io/crates/curtana)
[![`curtana` on docs.rs](https://img.shields.io/docsrs/curtana)](https://docs.rs/curtana/)
[![`curtana` is MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/with-caer/curtana/blob/main/LICENSE.txt)

An accessible low-overhead wrapper over [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
powered by [`llama-cpp-2`](https://github.com/utilityai/llama-cpp-rs/tree/main), supporting
most `.gguf`-formatted "Chat" and "Embedding" models.

## Examples

These examples assume the following models are downloaded into the working directory:

- [`Llama-3.2-3B-Instruct-Q6_K.gguf`](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf)
- [`nomic-embed-text-v1.5.f16.gguf`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf)

### Chat (via [Llama 3.2 Instruct](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF))

```rust
// Create a new local model registry and load
// a chat model into it with a system prompt
// of "You are a cupcake."
let registry = ModelRegistry::new().unwrap();
let mut model = registry
    .load_chat_model("Llama-3.2-3B-Instruct-Q6_K.gguf", "You are a cupcake.")
    .unwrap();

// Run ("infer") the model with the prompt
// "What are you?", capturing its output
// as UTF-8 encoded bytes.
let mut output = vec![];
model.infer("What are you?", &mut output).unwrap();
let output = String::from_utf8_lossy(&output);

// Hopefully, the model thinks it's a cupcake due
// to the system prompt.
assert!(output.to_lowercase().contains("cupcake"));
```

### Embedding (via [Nomic Embedding 1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF))

```rust
// Create a new local model registry and load
// an embedding model into it.
let registry = ModelRegistry::new().unwrap();
let mut model = registry
    .load_text_embedding_model("nomic-embed-text-v1.5.f16.gguf")
    .unwrap();

// Embed some fanciful document titles with the model.
let embeddings = model
    .embed(&[
        "search_document: might and magic in fantasy realms",
        "search_document: swords and sorcery for fantasy authors",
        "search_document: practical engineering for scientists",
    ])
    .unwrap();
assert_eq!(3, embeddings.len());

// Embed a search query with the model.
let query_embeddings = model.embed(&["query_document: fantasy"]).unwrap();
assert_eq!(1, query_embeddings.len());

// Calculate the cosine distance (or "similarity") between the embeddings.
let distance_a = cosine_distance(&query_embeddings[0], &embeddings[0]);
let distance_b = cosine_distance(&query_embeddings[0], &embeddings[1]);
let distance_c = cosine_distance(&query_embeddings[0], &embeddings[2]);

// The fantasy embeddings should be more similar
// than the scientific embedding.
assert!(distance_a < distance_c);
assert!(distance_b < distance_c);
```

## License

Copyright © 2025 With Caer, LLC.

Licensed under the MIT license. Refer to [the license file](https://github.com/with-caer/curtana/blob/main/LICENSE.txt) for more info.