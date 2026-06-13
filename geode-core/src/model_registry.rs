use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub hf_url: String,
    pub description: String,
}

pub fn list_available_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            name: "llama3.1-8b".to_string(),
            hf_url: "https://huggingface.co/bartowski/Llama-3.1-8B-Instruct-GGUF/resolve/main/Llama-3.1-8B-Instruct-Q4_K_M.gguf".to_string(),
            description: "Meta Llama 3.1 Instruct 8B, Q4 quantization".to_string(),
        },
        ModelInfo {
            name: "llama3.1-70b".to_string(),
            hf_url: "https://huggingface.co/bartowski/Llama-3.1-70B-Instruct-GGUF/resolve/main/Llama-3.1-70B-Instruct-Q4_K_M.gguf".to_string(),
            description: "Meta Llama 3.1 Instruct 70B, Q4 quantization".to_string(),
        },
        ModelInfo {
            name: "mistral-7b".to_string(),
            hf_url: "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf".to_string(),
            description: "Mistral v0.3 Instruct 7B, Q4 quantization".to_string(),
        },
        ModelInfo {
            name: "phi3-mini".to_string(),
            hf_url: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-GGUF/resolve/main/phi-3-mini-4k-instruct-q4.gguf".to_string(),
            description: "Microsoft Phi-3 Mini 4K Instruct, Q4 quantization".to_string(),
        },
        ModelInfo {
            name: "gemma-2-2b".to_string(),
            hf_url: "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf".to_string(),
            description: "Google Gemma-2 2B Instruct, Q4 quantization".to_string(),
        },
        ModelInfo {
            name: "qwen2.5-7b".to_string(),
            hf_url: "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf".to_string(),
            description: "Qwen 2.5 Instruct 7B, Q4 quantization".to_string(),
        },
    ]
}

pub fn get_model(name: &str) -> Option<ModelInfo> {
    list_available_models()
        .into_iter()
        .find(|m| m.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_models_returns_entries() {
        let models = list_available_models();
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.name == "llama3.1-8b"));
    }

    #[test]
    fn test_get_model_found() {
        let model = get_model("llama3.1-8b");
        assert!(model.is_some());
        assert_eq!(model.unwrap().description.len(), "Meta Llama 3.1 Instruct 8B, Q4 quantization".len());
    }

    #[test]
    fn test_get_model_not_found() {
        assert!(get_model("nonexistent-model").is_none());
    }
}
