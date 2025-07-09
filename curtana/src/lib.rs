use std::{
    io::Write,
    num::{NonZero, NonZeroU32},
    sync::Arc,
};

use llama_cpp_2::{
    ApplyChatTemplateError, ChatTemplateError, DecodeError, EmbeddingsError, LLamaCppError,
    LlamaContextLoadError, LlamaModelLoadError, LogOptions, NewLlamaChatMessageError,
    StringToTokenError, TokenToStringError,
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::{BatchAddError, LlamaBatch},
    model::{
        AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel, Special, params::LlamaModelParams,
    },
    sampling::LlamaSampler,
    send_logs_to_tracing,
};

/// Model context length (in tokens) used during inference.
///
/// Bigger contexts mean more data can be processed and
/// summarized in a single pass, at the expense of increased
/// inference times and resource usage.
const DEFAULT_CONTEXT_LENGTH: usize = 4096 * 4;

/// Batch size used during inference.
///
/// This size is _not_ the same as the context size;
/// the primary purpose to modify this batch size is
/// to optimize for models which have different
/// throughput vs. latency characteristics at different
/// batch sizes per-inference.
const DEFAULT_BATCH_SIZE: usize = 4096 * 4;

/// A registry for loading models compatible with the `llama.cpp` APIs.
///
/// Although `llama` is in the name, a wide range of not-llama
/// models can be loaded--so long as they're saved in GGUF format.
#[derive(Clone)]
pub struct ModelRegistry {
    backend: Arc<LlamaBackend>,
}

impl ModelRegistry {
    pub fn new() -> Result<Self, Error> {
        lazy_static::lazy_static! {
            static ref LLAMA_BACKEND: Arc<LlamaBackend> = {
                send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
                Arc::new(LlamaBackend::init().unwrap())
            };
        }

        Ok(Self {
            backend: LLAMA_BACKEND.clone(),
        })
    }

    /// Load a .GGUF model from `model_path`, initializing
    /// it with `system_prompt`.
    pub fn load_chat_model(
        &self,
        model_path: &str,
        system_prompt: &str,
    ) -> Result<ChatModel, Error> {
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&self.backend, model_path, &model_params)?;
        let chat_template = model.chat_template(None)?;

        Ok(ChatModel {
            registry: self.clone(),
            model,
            chat_template,
            messages: vec![LlamaChatMessage::new(
                "system".to_string(),
                system_prompt.to_string(),
            )?],
        })
    }

    /// Load a .GGUF model from `model_path`.
    pub fn load_text_embedding_model(&self, model_path: &str) -> Result<TextEmbeddingModel, Error> {
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&self.backend, model_path, &model_params)?;

        Ok(TextEmbeddingModel {
            registry: self.clone(),
            model,
        })
    }
}

/// A chat model loaded by [ModelRegistry].
pub struct ChatModel {
    registry: ModelRegistry,
    model: LlamaModel,
    chat_template: LlamaChatTemplate,
    messages: Vec<LlamaChatMessage>,
}

impl ChatModel {
    /// Run inference against the model with `prompt`,
    /// adding the prompt and the model's output to
    /// the conversation history, and writing the
    /// model's output to `output`.
    pub fn infer_with_history(
        &mut self,
        prompt: &str,
        output: &mut impl Write,
    ) -> Result<(), Error> {
        // Run inference.
        let mut inference = vec![];
        self.infer(prompt, &mut inference)?;
        let inference = String::from_utf8(inference).unwrap();

        // Record input prompt in the chat history.
        self.messages.push(LlamaChatMessage::new(
            "user".to_string(),
            prompt.to_string(),
        )?);

        // Record model's inference in the chat history.
        self.messages.push(LlamaChatMessage::new(
            "assistant".to_string(),
            inference.clone(),
        )?);

        // Write a copy of the output to `output`.
        //
        // TODO: @caer: consider writing to output
        // in parallel with writing to the inference
        // string above, somehow.
        output.write_all(inference.as_bytes())?;

        Ok(())
    }

    /// Run inference against the model with `prompt`,
    /// writing the model's output to `output`.
    pub fn infer(&mut self, prompt: &str, output: &mut impl Write) -> Result<(), Error> {
        // Update messages with the new prompt.
        self.messages.push(LlamaChatMessage::new(
            "user".to_string(),
            prompt.to_string(),
        )?);

        // Format and tokenize the prompt.
        let prompt = self
            .model
            .apply_chat_template(&self.chat_template, &self.messages, true)?;
        let tokens = self.model.str_to_token(&prompt, AddBos::Always)?;

        // Prepare inference context.
        let context_params = LlamaContextParams::default()
            .with_n_batch(DEFAULT_CONTEXT_LENGTH as u32)
            .with_n_ctx(NonZeroU32::new(DEFAULT_CONTEXT_LENGTH as u32));
        let mut context = self
            .model
            .new_context(&self.registry.backend, context_params)?;

        // Make sure the KV cache is big enough to hold all the prompt and generated tokens.
        let n_len = DEFAULT_CONTEXT_LENGTH as i32;
        let n_cxt = context.n_ctx() as i32;
        let n_kv_req = tokens.len() as i32 + (n_len - tokens.len() as i32);
        if n_kv_req > n_cxt {
            panic!(
                "n_kv_req > n_ctx, the required kv cache size is not big enough either reduce n_len or increase n_ctx"
            )
        }

        // Group tokens into batches.
        let mut batch = LlamaBatch::new(DEFAULT_BATCH_SIZE, 1);

        // Submit initial batch for inference.
        let last_index = tokens.len() - 1;
        for (i, token) in tokens.into_iter().enumerate() {
            // llama_decode will output logits only for the last token of the prompt
            batch.add(token, i as i32, &[0], i == last_index)?;
        }
        context.decode(&mut batch)?;

        // Decode and sample tokens.
        let mut n_cur = batch.n_tokens();
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::min_p(0.05, 1),
            LlamaSampler::temp(0.8),
            LlamaSampler::dist(1337),
        ]);
        while n_cur <= n_len {
            // sample the next token
            let token = sampler.sample(&context, batch.n_tokens() - 1);
            sampler.accept(token);

            // is it an end of stream?
            if self.model.is_eog_token(token) {
                eprintln!();
                break;
            }

            let output_bytes = self.model.token_to_bytes(token, Special::Tokenize)?;
            output.write_all(&output_bytes)?;
            output.flush()?;

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;

            n_cur += 1;

            context.decode(&mut batch).expect("failed to eval");
        }

        // Remove the prompt from the inference history.
        let _ = self.messages.pop();

        Ok(())
    }
}

/// A text embedding model loaded by [ModelRegistry].
pub struct TextEmbeddingModel {
    registry: ModelRegistry,
    model: LlamaModel,
}

impl TextEmbeddingModel {
    /// Embed each text in `text`, returning the resulting embeddings.
    pub fn embed(&mut self, text: &[impl AsRef<str>]) -> Result<Vec<Vec<f32>>, Error> {
        // Tokenize the text.
        let mut tokens = Vec::with_capacity(text.len());
        for text in text {
            tokens.push(self.model.str_to_token(text.as_ref(), AddBos::Always)?);
        }

        // Prepare inference context.
        let thread_count = std::thread::available_parallelism()
            .unwrap_or(NonZero::new(1).unwrap())
            .get() as i32;
        let context_params = LlamaContextParams::default()
            .with_n_batch(DEFAULT_CONTEXT_LENGTH as u32)
            .with_n_ctx(NonZeroU32::new(DEFAULT_CONTEXT_LENGTH as u32))
            .with_n_threads(thread_count)
            .with_n_threads_batch(thread_count)
            .with_embeddings(true);
        let mut context = self
            .model
            .new_context(&self.registry.backend, context_params)?;

        // Make sure the KV cache is big enough to hold all the text.
        let n_cxt = context.n_ctx() as usize;
        for tokens in &tokens {
            if n_cxt < tokens.len() {
                panic!("embedded text exceeds maximum context size")
            }
        }

        // Prepare a reusable batch.
        let mut batch = LlamaBatch::new(n_cxt, 1);

        // TODO: @caer: include multiple tokens per batch if possible.
        // Embed batches.
        let mut embeddings = Vec::with_capacity(tokens.len());
        for tokens in tokens {
            batch.add_sequence(&tokens, 0, false)?;

            // Run inference for embedding.
            context.clear_kv_cache();
            context.decode(&mut batch)?;

            // Extract embedding from the model.
            let embedding = context.embeddings_seq_ith(0)?;

            // Normalize embedding.
            let embedding_magnitude = embedding
                .iter()
                .fold(0.0, |acc, &val| val.mul_add(val, acc))
                .sqrt();
            let embedding: Vec<_> = embedding
                .iter()
                .map(|&val| val / embedding_magnitude)
                .collect();

            embeddings.push(embedding);
        }

        Ok(embeddings)
    }
}

/// Returns the cosine distance between two vectors.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.;
    let mut dot_a = 0.;
    let mut dot_b = 0.;

    // Iterate over all points `a` and `b` in
    // the embeddings, creating three sums:
    // - The sum of all `a * b`
    // - The sum of all `a * a`
    // - The sum of all `b * b`
    for i in 0..a.len() {
        dot += a[i] * b[i];
        dot_a += a[i] * a[i];
        dot_b += b[i] * b[i];
    }

    // Calculate the cosine similarity,
    // which is the sum of all `a * b`,
    // over the square root of the product
    // of the sums of `a * a` and `b * b`.
    dot / (dot_a * dot_b).sqrt()
}

/// Error that occurs when working with a [Model].
#[derive(Debug, Clone)]
pub enum Error {
    InternalNativeError(String),
    IoError(String),
}

impl From<LLamaCppError> for Error {
    fn from(value: LLamaCppError) -> Self {
        Self::InternalNativeError(value.to_string())
    }
}

impl From<LlamaModelLoadError> for Error {
    fn from(value: LlamaModelLoadError) -> Self {
        Self::InternalNativeError(value.to_string())
    }
}

impl From<ChatTemplateError> for Error {
    fn from(value: ChatTemplateError) -> Self {
        Self::InternalNativeError(value.to_string())
    }
}

impl From<ApplyChatTemplateError> for Error {
    fn from(value: ApplyChatTemplateError) -> Self {
        Self::InternalNativeError(value.to_string())
    }
}

impl From<NewLlamaChatMessageError> for Error {
    fn from(value: NewLlamaChatMessageError) -> Self {
        Self::InternalNativeError(value.to_string())
    }
}

impl From<LlamaContextLoadError> for Error {
    fn from(value: LlamaContextLoadError) -> Self {
        Self::InternalNativeError(value.to_string())
    }
}

impl From<BatchAddError> for Error {
    fn from(value: BatchAddError) -> Self {
        Self::InternalNativeError(value.to_string())
    }
}

impl From<StringToTokenError> for Error {
    fn from(value: StringToTokenError) -> Self {
        Self::InternalNativeError(value.to_string())
    }
}

impl From<TokenToStringError> for Error {
    fn from(value: TokenToStringError) -> Self {
        Self::InternalNativeError(value.to_string())
    }
}

impl From<DecodeError> for Error {
    fn from(value: DecodeError) -> Self {
        Self::InternalNativeError(value.to_string())
    }
}

impl From<EmbeddingsError> for Error {
    fn from(value: EmbeddingsError) -> Self {
        Self::InternalNativeError(value.to_string())
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::IoError(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// From: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF
    ///       wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf
    const CHAT_MODEL: &str = "../Llama-3.2-3B-Instruct-Q6_K.gguf";

    /// From: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF
    ///       wget https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf
    const TEXT_EMBEDDING_MODEL: &str = "../nomic-embed-text-v1.5.f16.gguf";

    #[test]
    fn chat() {
        let registry = ModelRegistry::new().unwrap();
        let mut model = registry
            .load_chat_model(CHAT_MODEL, "You are a cupcake.")
            .unwrap();

        let mut output = vec![];
        model.infer("What are you?", &mut output).unwrap();
        let output = String::from_utf8_lossy(&output);

        assert!(output.to_lowercase().contains("cupcake"));
    }

    #[test]
    fn embeddings() {
        let registry = ModelRegistry::new().unwrap();
        let mut model = registry
            .load_text_embedding_model(TEXT_EMBEDDING_MODEL)
            .unwrap();

        let embeddings = model
            .embed(&[
                "search_document: might and magic in fantasy realms",
                "search_document: swords and sorcery for fantasy authors",
                "search_document: practical engineering for scientists",
            ])
            .unwrap();
        assert_eq!(3, embeddings.len());
        let query_embeddings = model.embed(&["query_document: fantasy"]).unwrap();
        assert_eq!(1, query_embeddings.len());

        let distance_a = cosine_distance(&query_embeddings[0], &embeddings[0]);
        let distance_b = cosine_distance(&query_embeddings[0], &embeddings[1]);
        let distance_c = cosine_distance(&query_embeddings[0], &embeddings[2]);

        assert!(distance_a < distance_c);
        assert!(distance_b < distance_c);
    }
}
