// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Prompt embedding provider resolution for example binaries.
//!
//! Self-contained so integration tests can `#[path]`-include this file under
//! `--no-default-features`. Heavier Ollama/synthetic embedding math stays in
//! `mod.rs`.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptEmbeddingProvider {
    Ollama,
    SyntheticFallback,
}

pub fn resolve_prompt_embedding_provider(value: Option<&str>) -> PromptEmbeddingProvider {
    match value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        None | Some("ollama") => PromptEmbeddingProvider::Ollama,
        _ => PromptEmbeddingProvider::SyntheticFallback,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_embedding_provider_defaults_to_ollama() {
        assert_eq!(
            resolve_prompt_embedding_provider(None),
            PromptEmbeddingProvider::Ollama
        );
    }

    #[test]
    fn prompt_embedding_provider_accepts_ollama_case_insensitively() {
        assert_eq!(
            resolve_prompt_embedding_provider(Some("  OLLAMA  ")),
            PromptEmbeddingProvider::Ollama
        );
    }

    #[test]
    fn prompt_embedding_provider_rejects_legacy_llama_cpp() {
        assert_eq!(
            resolve_prompt_embedding_provider(Some("llama_cpp")),
            PromptEmbeddingProvider::SyntheticFallback
        );
    }

    #[test]
    fn prompt_embedding_provider_maps_hash_and_synthetic_to_fallback() {
        assert_eq!(
            resolve_prompt_embedding_provider(Some("hash")),
            PromptEmbeddingProvider::SyntheticFallback
        );
        assert_eq!(
            resolve_prompt_embedding_provider(Some("synthetic")),
            PromptEmbeddingProvider::SyntheticFallback
        );
    }
}
