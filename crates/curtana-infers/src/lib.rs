use std::{
    io::Write,
    num::{NonZero, NonZeroU32},
    sync::Arc,
};

use llama_cpp_2::{
    ApplyChatTemplateError, ChatTemplateError, DecodeError, EmbeddingsError, LlamaContextLoadError,
    LlamaCppError, LlamaModelLoadError, LogOptions, NewLlamaChatMessageError, StringToTokenError,
    TokenToStringError,
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::{BatchAddError, LlamaBatch},
    model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel, params::LlamaModelParams},
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
        use std::sync::OnceLock;
        static LLAMA_BACKEND: OnceLock<Result<Arc<LlamaBackend>, String>> = OnceLock::new();

        let result = LLAMA_BACKEND.get_or_init(|| {
            send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
            LlamaBackend::init()
                .map(Arc::new)
                .map_err(|e| e.to_string())
        });

        match result {
            Ok(backend) => Ok(Self {
                backend: backend.clone(),
            }),
            Err(e) => Err(Error::InternalNativeError(format!(
                "failed to initialize llama backend: {e}"
            ))),
        }
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

/// RAII guard that pops the last message from the chat history on drop.
/// Used by `infer()` to ensure the user prompt is always removed, even
/// if an intermediate operation returns early via `?`.
struct MessageGuard<'a> {
    messages: &'a mut Vec<LlamaChatMessage>,
}

impl Drop for MessageGuard<'_> {
    fn drop(&mut self) {
        self.messages.pop();
    }
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
        let inference = String::from_utf8_lossy(&inference).into_owned();

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

    /// Non-streaming inference that accumulates history and returns
    /// the full output as a String.
    pub fn infer_with_history_to_string(&mut self, prompt: &str) -> Result<String, Error> {
        let mut buf = Vec::new();
        self.infer_with_history(prompt, &mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    /// Replace the system prompt (messages[0]).
    pub fn replace_system_prompt(&mut self, prompt: String) -> Result<(), Error> {
        if self.messages.is_empty() {
            self.messages
                .push(LlamaChatMessage::new("system".to_string(), prompt)?);
        } else {
            self.messages[0] = LlamaChatMessage::new("system".to_string(), prompt)?;
        }
        Ok(())
    }

    /// Clear all history except the system prompt.
    pub fn clear_history(&mut self) {
        self.messages.truncate(1);
    }

    /// Returns the number of messages in the conversation history.
    pub fn history_len(&self) -> usize {
        self.messages.len()
    }

    /// Run inference against the model with `prompt`,
    /// writing the model's output to `output`.
    pub fn infer(&mut self, prompt: &str, output: &mut impl Write) -> Result<(), Error> {
        // Update messages with the new prompt.
        self.messages.push(LlamaChatMessage::new(
            "user".to_string(),
            prompt.to_string(),
        )?);

        // Format and tokenize the prompt (borrows model, chat_template, messages immutably).
        let prompt = self
            .model
            .apply_chat_template(&self.chat_template, &self.messages, true)?;
        let tokens = self.model.str_to_token(&prompt, AddBos::Always)?;

        // Guard ensures the pushed message is always popped, even on early `?` returns.
        let _guard = MessageGuard {
            messages: &mut self.messages,
        };

        // Prepare inference context.
        let context_params = LlamaContextParams::default()
            .with_n_batch(DEFAULT_CONTEXT_LENGTH as u32)
            .with_n_ctx(NonZeroU32::new(DEFAULT_CONTEXT_LENGTH as u32));
        let mut context = self
            .model
            .new_context(&self.registry.backend, context_params)?;

        // Make sure the KV cache is big enough to hold all the prompt and generated tokens.
        let n_len: i32 = i32::try_from(DEFAULT_CONTEXT_LENGTH).map_err(|_| Error::ContextSize {
            maximum: i32::MAX as usize,
            actual: DEFAULT_CONTEXT_LENGTH,
        })?;
        let n_cxt = context.n_ctx() as i32;
        let tokens_len_i32 = i32::try_from(tokens.len()).map_err(|_| Error::ContextSize {
            maximum: i32::MAX as usize,
            actual: tokens.len(),
        })?;
        let n_kv_req = tokens_len_i32 + (n_len - tokens_len_i32);
        if n_kv_req > n_cxt {
            return Err(Error::ContextSize {
                maximum: n_cxt as usize,
                actual: n_kv_req as usize,
            });
        }

        // Group tokens into batches.
        let mut batch = LlamaBatch::new(DEFAULT_BATCH_SIZE, 1);

        // Submit initial batch for inference.
        let last_index = tokens.len() - 1;
        for (i, token) in tokens.into_iter().enumerate() {
            let pos = i32::try_from(i).map_err(|_| Error::ContextSize {
                maximum: i32::MAX as usize,
                actual: i,
            })?;
            // llama_decode will output logits only for the last token of the prompt
            batch.add(token, pos, &[0], i == last_index)?;
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
                break;
            }

            let output_bytes =
                self.model
                    .token_to_piece_bytes(token, DEFAULT_CONTEXT_LENGTH, true, None)?;
            output.write_all(&output_bytes)?;
            output.flush()?;

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;

            n_cur += 1;

            context.decode(&mut batch)?;
        }

        // Guard drops here, popping the message.
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
        // Use the model's trained context length to avoid exceeding
        // position embedding table bounds (e.g. BERT-style models).
        let n_ctx_train = self.model.n_ctx_train();
        let thread_count = std::thread::available_parallelism()
            .unwrap_or(NonZero::new(1).unwrap())
            .get() as i32;
        let context_params = LlamaContextParams::default()
            .with_n_batch(n_ctx_train)
            .with_n_ubatch(n_ctx_train)
            .with_n_ctx(NonZeroU32::new(n_ctx_train))
            .with_n_threads(thread_count)
            .with_n_threads_batch(thread_count)
            .with_embeddings(true);
        let mut context = self
            .model
            .new_context(&self.registry.backend, context_params)?;

        // Make sure the KV cache is big enough to hold all the text.
        let n_ctx = context.n_ctx() as usize;
        let n_ubatch = context.n_ubatch() as usize;
        for tokens in &tokens {
            if n_ctx < tokens.len() {
                return Err(Error::ContextSize {
                    maximum: n_ubatch,
                    actual: tokens.len(),
                });
            } else if n_ubatch < tokens.len() {
                return Err(Error::MicrobatchSize {
                    maximum: n_ubatch,
                    actual: tokens.len(),
                });
            }
        }

        // Prepare a reusable batch.
        let mut batch = LlamaBatch::new(n_ctx, 1);

        // TODO: @caer: include multiple tokens per batch if possible.
        // Embed batches.
        let mut embeddings = Vec::with_capacity(tokens.len());
        for tokens in tokens {
            batch.clear();
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

    // Guard against zero-magnitude vectors.
    if dot_a == 0.0 || dot_b == 0.0 {
        return 0.0;
    }

    // Calculate the cosine similarity,
    // which is the sum of all `a * b`,
    // over the square root of the product
    // of the sums of `a * a` and `b * b`.
    dot / (dot_a * dot_b).sqrt()
}

/// Error that occurs when working with a model.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    TextTooLong(&'static str),

    /// An input was too large for the configured
    /// context size.
    ContextSize {
        maximum: usize,
        actual: usize,
    },

    /// An input was too large for the configured
    /// microbatch size.
    MicrobatchSize {
        maximum: usize,
        actual: usize,
    },

    InternalNativeError(String),
    IoError(String),
}

impl From<LlamaCppError> for Error {
    fn from(value: LlamaCppError) -> Self {
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

    /// From: https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF
    ///       wget https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf
    const CHAT_MODEL: &str = "../../Ministral-3-3B-Instruct-2512-Q4_K_M.gguf";

    // From: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF
    //       wget https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf
    // const TEXT_EMBEDDING_MODEL: &str = "../../nomic-embed-text-v1.5.f16.gguf";

    /// From: https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-ggml-model-f16.gguf
    ///       wget https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-ggml-model-f16.gguf
    const TEXT_EMBEDDING_MODEL: &str = "../../all-MiniLM-L6-v2-ggml-model-f16.gguf";

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
    fn embeds_text() {
        let registry = ModelRegistry::new().unwrap();
        let mut model = registry
            .load_text_embedding_model(TEXT_EMBEDDING_MODEL)
            .unwrap();

        let embeddings = model
            .embed(&[
                "might and magic in fantasy realms",
                "swords and sorcery for fantasy authors",
                "practical engineering for scientists",
            ])
            .unwrap();
        assert_eq!(3, embeddings.len());
        let query_embeddings = model.embed(&["fantasy"]).unwrap();
        assert_eq!(1, query_embeddings.len());

        let similarity_a = cosine_distance(&query_embeddings[0], &embeddings[0]);
        let similarity_b = cosine_distance(&query_embeddings[0], &embeddings[1]);
        let similarity_c = cosine_distance(&query_embeddings[0], &embeddings[2]);

        // Fantasy texts should be more similar to the "fantasy" query
        // than an engineering text is.
        assert!(similarity_a > similarity_c);
        assert!(similarity_b > similarity_c);
    }

    #[test]
    fn embeddings_reject_large_texts() {
        let registry = ModelRegistry::new().unwrap();
        let mut model = registry
            .load_text_embedding_model(TEXT_EMBEDDING_MODEL)
            .unwrap();

        // Create a string which should tokenize into a size
        // much greater than the supported microbatch size.
        let mut text = String::from("search document:");
        while text.len() < DEFAULT_CONTEXT_LENGTH * 8 {
            text.push_str(" might and magic in fantasy realms and");
        }

        let embedding = model.embed(&[text]);

        assert!(
            matches!(
                embedding,
                Err(Error::ContextSize { .. }) | Err(Error::MicrobatchSize { .. })
            ),
            "expected ContextSize or MicrobatchSize error, got: {embedding:?}"
        );
    }
}
