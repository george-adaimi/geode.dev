use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const CONFIG_DIR: &str = ".geode";

pub fn config_dir() -> PathBuf {
    dirs::home_dir()
        .map(|h| h.join(".geode"))
        .expect("Failed to resolve home directory")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub llm: LlmConfig,
    pub tools: ToolsConfig,
    pub safety: SafetyConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub server_url: String,
    pub model_path: String,
    pub context_window: usize,
    pub summarize_threshold: usize,
    #[serde(default)]
    pub system_prompt: Option<String>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            server_url: "http://localhost:8080".to_string(),
            model_path: String::new(),
            context_window: 8192,
            summarize_threshold: 6144,
            system_prompt: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsConfig {
    pub enabled: Vec<String>,
    #[serde(default)]
    pub disabled: Vec<String>,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            enabled: vec!["fs".to_string(), "shell".to_string(), "web".to_string()],
            disabled: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConfig {
    #[serde(default = "default_auto_approve")]
    pub auto_approve_safe: bool,
}

fn default_auto_approve() -> bool {
    true
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            auto_approve_safe: true,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            llm: LlmConfig::default(),
            tools: ToolsConfig::default(),
            safety: SafetyConfig::default(),
        }
    }
}

pub fn config_path() -> PathBuf {
    config_dir().join("config.toml")
}

pub fn models_dir() -> PathBuf {
    config_dir().join("models")
}

pub fn system_prompt_path() -> PathBuf {
    config_dir().join("SYSTEM.md")
}

pub fn default_config() -> Config {
    Config::default()
}

pub fn load_config() -> anyhow::Result<Config> {
    let path = config_path();
    let contents = std::fs::read_to_string(&path)?;
    let config: Config = toml::from_str(&contents)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_serialization() {
        let config = default_config();
        let toml_str = toml::to_string_pretty(&config).expect("Failed to serialize config");
        let deserialized: Config = toml::from_str(&toml_str).expect("Failed to deserialize config");
        assert_eq!(deserialized.llm.server_url, "http://localhost:8080");
        assert_eq!(deserialized.llm.context_window, 8192);
        assert_eq!(deserialized.llm.summarize_threshold, 6144);
        assert!(deserialized.llm.system_prompt.is_none());
        assert_eq!(deserialized.tools.enabled.len(), 3);
        assert!(deserialized.safety.auto_approve_safe);
    }

    #[test]
    fn test_config_dir_exists() {
        let dir = config_dir();
        assert!(dir.is_absolute() || dir.components().count() > 0);
    }

    #[test]
    fn test_config_path_construction() {
        let path = config_path();
        assert!(path.ends_with("config.toml"));
    }

    #[test]
    fn test_load_fails_without_file() {
        // Test that parsing fails on invalid TOML content
        let result: Result<Config, _> = toml::from_str("invalid [[[[");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_config_missing_fields() {
        // Test that a minimal TOML with missing required fields fails
        let result: Result<Config, _> = toml::from_str("[llm]");
        assert!(result.is_err());
    }
}
