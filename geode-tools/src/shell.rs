use async_trait::async_trait;
use geode_core::{SafetyLevel, Tool, ToolResult};

pub struct ShellTool;

impl ShellTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ShellTool {
    fn default() -> Self {
        Self::new()
    }
}

const MAX_OUTPUT_CHARS: usize = 4096;
const TRUNCATE_HEAD: usize = 2048;
const TRUNCATE_TAIL: usize = 2048;

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Execute shell commands and capture output"
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "args": {
                    "type": "string",
                    "description": "Arguments to the command (optional, appended to command)"
                }
            },
            "required": ["command"]
        })
    }

    fn safety_level(&self) -> SafetyLevel {
        SafetyLevel::Dangerous
    }

    async fn execute(&self, args: serde_json::Value) -> ToolResult {
        let command = match args.get("command").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => return ToolResult::err("", "Missing required 'command' field"),
        };
        let extra_args = args
            .get("args")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let full_cmd = if extra_args.is_empty() {
            command.to_string()
        } else {
            format!("{} {}", command, extra_args)
        };

        let output = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(&full_cmd)
            .output()
            .await;

        match output {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let exit_code = output.status.code().map(|c| c.to_string()).unwrap_or("?".to_string());

                let mut result = stdout.to_string();
                if !stderr.is_empty() {
                    if !result.is_empty() {
                        result.push('\n');
                    }
                    result.push_str(&format!("(stderr: {})", stderr.trim()));
                }

                let truncated = truncate_output(&result);
                let metadata = format!(
                    "[Exit code: {}]\n{}",
                    exit_code, truncated
                );
                ToolResult::ok(metadata)
            }
            Err(e) => ToolResult::err("", format!("Failed to execute command: {}", e)),
        }
    }
}

fn truncate_output(output: &str) -> String {
    if output.len() <= MAX_OUTPUT_CHARS {
        output.to_string()
    } else {
        let head = &output[..TRUNCATE_HEAD.min(output.len())];
        let remaining = output.len() - MAX_OUTPUT_CHARS;
        let tail_start = output.len().saturating_sub(TRUNCATE_TAIL);
        let tail = &output[tail_start..];
        format!(
            "{}\n... [{} more characters not shown] ...\n{}",
            head, remaining, tail
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_short_output() {
        let short = "hello world";
        assert_eq!(truncate_output(short), short);
    }

    #[test]
    fn test_truncate_long_output() {
        let long = "a".repeat(5000);
        let truncated = truncate_output(&long);
        assert!(truncated.len() < long.len());
        assert!(truncated.contains("more characters not shown"));
        assert!(truncated.starts_with("aaaaa"));
        assert!(truncated.ends_with("aaaaa"));
    }

    #[test]
    fn test_truncate_exact_boundary() {
        let exact = "a".repeat(MAX_OUTPUT_CHARS);
        assert_eq!(truncate_output(&exact), exact);
    }
}
