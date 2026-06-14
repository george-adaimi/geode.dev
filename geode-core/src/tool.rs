use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn schema(&self) -> serde_json::Value;
    fn safety_level(&self) -> SafetyLevel;

    /// Dynamic safety level based on the actual tool arguments.
    /// Defaults to the static safety_level().
    fn args_safety_level(&self, _args: &serde_json::Value) -> SafetyLevel {
        self.safety_level()
    }

    async fn execute(&self, args: serde_json::Value) -> ToolResult;
}

#[derive(Debug, Clone, PartialEq)]
pub enum SafetyLevel {
    Safe,
    Warning,
    Dangerous,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub output: String,
    pub success: bool,
    pub error: Option<String>,
}

impl ToolResult {
    pub fn ok(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            success: true,
            error: None,
        }
    }

    pub fn err(output: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            success: false,
            error: Some(error.into()),
        }
    }
}
