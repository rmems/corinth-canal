//! Prompt tokenization helpers for local tokenizer.json files.

use crate::error::{HybridError, Result};
use std::path::Path;
use tokenizers::Tokenizer;

/// Encode prompt text with a local `tokenizer.json`.
pub fn encode_prompt_with_tokenizer_json(
    tokenizer_path: &Path,
    prompt: &str,
) -> Result<Vec<usize>> {
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|err| HybridError::ModelLoad {
        path: tokenizer_path.display().to_string(),
        reason: format!("failed to load tokenizer.json: {err}"),
    })?;
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|err| HybridError::ModelLoad {
            path: tokenizer_path.display().to_string(),
            reason: format!("failed to encode prompt text: {err}"),
        })?;

    Ok(encoding.get_ids().iter().map(|&id| id as usize).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const CANONICAL_PROMPTS: [&str; 3] = [
        "Let's teach this MoE model the language of SNN",
        "fn main() { println!(\"Hello, World!\"); }",
        "The derivative of a constant is mathematically zero.",
    ];

    #[test]
    fn test_qwen_tokenizer_encodes_canonical_prompts_when_present() {
        let tokenizer_path = PathBuf::from(
            "/home/raulmc/Downloads/SNN_Quantization/Qwen-MoE-2.7B-Int4/tokenizer.json",
        );
        if !tokenizer_path.exists() {
            return;
        }

        for prompt in CANONICAL_PROMPTS {
            let token_ids = encode_prompt_with_tokenizer_json(&tokenizer_path, prompt).unwrap();
            assert!(
                !token_ids.is_empty(),
                "tokenizer should emit at least one token for '{prompt}'"
            );
        }
    }
}
