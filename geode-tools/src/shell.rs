use async_trait::async_trait;
use geode_core::{SafetyLevel, Tool, ToolResult, Tokenizer};

pub struct ShellTool {
    max_tokens: usize,
}

impl ShellTool {
    pub fn new() -> Self {
        Self { max_tokens: 1024 }
    }
}

impl Default for ShellTool {
    fn default() -> Self {
        Self::new()
    }
}

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

    fn args_safety_level(&self, args: &serde_json::Value) -> SafetyLevel {
        if let Some(command) = args.get("command").and_then(|v| v.as_str()) {
            let safe_commands = [
                "ls", "cat", "head", "tail", "echo", "pwd", "which",
                "whoami", "env", "date", "uname", "find", "grep",
                "wc", "sort", "diff", "file", "du", "df",
            ];
            let cmd_name = command.split_whitespace().next().unwrap_or("");
            if safe_commands.contains(&cmd_name) {
                return SafetyLevel::Safe;
            }
        }
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

                let truncated = truncate_output(&result, self.max_tokens);
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

fn truncate_output(output: &str, max_tokens: usize) -> String {
    let tok = Tokenizer::new();
    let token_count = tok.count(output);
    if token_count <= max_tokens {
        output.to_string()
    } else {
        // Estimate character positions using token ratio
        let ratio = max_tokens as f64 / token_count as f64;
        let target_chars = (output.len() as f64 * ratio) as usize;
        let head_len = (target_chars / 2).max(50);
        let tail_len = (target_chars / 2).max(50);
        let remaining = token_count - max_tokens;
        let head = &output[..head_len.min(output.len())];
        let tail = &output[output.len().saturating_sub(tail_len)..];
        format!(
            "{}\n... [~{} more tokens not shown] ...\n{}",
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
        assert_eq!(truncate_output(short, 1024), short);
    }

    #[test]
    fn test_truncate_long_output() {
        // ~1250 tokens at 4 chars/token
        let long = "a ".repeat(10000);
        let truncated = truncate_output(&long, 1024);
        assert!(truncated.len() < long.len());
        assert!(truncated.contains("more tokens not shown"));
        assert!(truncated.starts_with("a "));
        assert!(truncated.ends_with("a "));
    }

    #[test]
    fn test_truncate_exact_boundary() {
        // A string that's ~1000 tokens
        let exact = "hello ".repeat(800);
        let truncated = truncate_output(&exact, 2000);
        assert_eq!(truncated, exact);
    }
}
