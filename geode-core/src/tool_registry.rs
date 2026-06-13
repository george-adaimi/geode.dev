use crate::tool::Tool;
use std::collections::HashMap;

pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    pub fn list(&self) -> Vec<&dyn Tool> {
        self.tools.values().map(|t| t.as_ref()).collect()
    }

    pub fn to_function_definitions(&self) -> Vec<serde_json::Value> {
        self.tools
            .values()
            .map(|tool| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": tool.name(),
                        "description": tool.description(),
                        "parameters": tool.schema(),
                    }
                })
            })
            .collect()
    }

    /// Build a human-readable list of tools for the system prompt.
    pub fn to_tool_list(&self) -> String {
        self.tools
            .values()
            .map(|tool| format!("- {}: {}", tool.name(), tool.description()))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{SafetyLevel, Tool, ToolResult};
    use async_trait::async_trait;

    struct DummyTool;

    #[async_trait]
    impl Tool for DummyTool {
        fn name(&self) -> &str { "dummy" }
        fn description(&self) -> &str { "A dummy tool" }
        fn schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }
        fn safety_level(&self) -> SafetyLevel { SafetyLevel::Safe }
        async fn execute(&self, _args: serde_json::Value) -> ToolResult {
            ToolResult::ok("done")
        }
    }

    #[test]
    fn test_register_and_get() {
        let mut reg = ToolRegistry::new();
        assert!(reg.get("dummy").is_none());
        reg.register(Box::new(DummyTool));
        let tool = reg.get("dummy").expect("dummy tool should exist");
        assert_eq!(tool.name(), "dummy");
    }

    #[test]
    fn test_list() {
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(DummyTool));
        assert_eq!(reg.list().len(), 1);
    }

    #[test]
    fn test_to_function_definitions() {
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(DummyTool));
        let defs = reg.to_function_definitions();
        assert_eq!(defs.len(), 1);
        let def = &defs[0];
        assert_eq!(def["type"], "function");
        assert_eq!(def["function"]["name"], "dummy");
        assert_eq!(def["function"]["description"], "A dummy tool");
    }
}
