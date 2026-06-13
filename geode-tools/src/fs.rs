use async_trait::async_trait;
use geode_core::{SafetyLevel, Tool, ToolResult};

pub struct FsTool;

impl FsTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for FsTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for FsTool {
    fn name(&self) -> &str {
        "fs"
    }

    fn description(&self) -> &str {
        "Filesystem operations: read_file, write_file, list_dir, search_files"
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read_file", "write_file", "list_dir", "search_files"],
                    "description": "The filesystem operation to perform"
                },
                "path": {
                    "type": "string",
                    "description": "The file or directory path (required for read_file, write_file, list_dir)"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (required for write_file)"
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern for search_files (glob or regex)"
                },
                "search_dir": {
                    "type": "string",
                    "description": "Directory to search in for search_files (defaults to current directory)"
                }
            },
            "required": ["operation"]
        })
    }

    fn safety_level(&self) -> SafetyLevel {
        SafetyLevel::Safe
    }

    async fn execute(&self, args: serde_json::Value) -> ToolResult {
        let ops = args.get("operation").and_then(|v| v.as_str());
        match ops {
            Some("read_file") => read_file(args).await,
            Some("write_file") => write_file(args).await,
            Some("list_dir") => list_dir(args).await,
            Some("search_files") => search_files(args).await,
            Some(op) => ToolResult::err("", format!("Unknown fs operation: {}", op)),
            None => ToolResult::err("", "Missing required 'operation' field"),
        }
    }
}

async fn read_file(args: serde_json::Value) -> ToolResult {
    let path = match args.get("path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return ToolResult::err("", "Missing required 'path' field"),
    };

    match std::fs::read_to_string(path) {
        Ok(content) => ToolResult::ok(content),
        Err(e) => ToolResult::err("", format!("Failed to read {}: {}", path, e)),
    }
}

async fn write_file(args: serde_json::Value) -> ToolResult {
    let path = match args.get("path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return ToolResult::err("", "Missing required 'path' field"),
    };
    let content = match args.get("content").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return ToolResult::err("", "Missing required 'content' field"),
    };

    match std::fs::write(path, content) {
        Ok(_) => ToolResult::ok(format!("Wrote {} bytes to {}", content.len(), path)),
        Err(e) => ToolResult::err("", format!("Failed to write {}: {}", path, e)),
    }
}

async fn list_dir(args: serde_json::Value) -> ToolResult {
    let path = match args.get("path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return ToolResult::err("", "Missing required 'path' field"),
    };

    let entries = match std::fs::read_dir(path) {
        Ok(rd) => rd,
        Err(e) => {
            return ToolResult::err(
                "",
                format!("Failed to list {}: {}", path, e),
            );
        }
    };

    let mut items = Vec::new();
    for entry in entries {
        match entry {
            Ok(e) => {
                let metadata = e.metadata();
                let is_dir = metadata.as_ref().map(|m| m.is_dir()).unwrap_or(false);
                let prefix = if is_dir { "d" } else { "-" };
                items.push(format!("{} {}", prefix, e.file_name().to_string_lossy()));
            }
            Err(e) => items.push(format!("<???> {}", e)),
        }
    }

    let output = if items.is_empty() {
        "(empty directory)".to_string()
    } else {
        items.join("\n")
    };
    ToolResult::ok(output)
}

async fn search_files(args: serde_json::Value) -> ToolResult {
    let pattern = match args.get("pattern").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return ToolResult::err("", "Missing required 'pattern' field"),
    };
    let search_dir = args
        .get("search_dir")
        .and_then(|v| v.as_str())
        .unwrap_or(".");

    let mut matches = Vec::new();
    let search_path = std::path::Path::new(search_dir);

    for entry in walkdir::WalkDir::new(search_path).max_depth(10) {
        match entry {
            Ok(e) => {
                let file_name = e.file_name().to_string_lossy();
                if file_name.contains(pattern) {
                    matches.push(e.path().to_string_lossy().to_string());
                }
            }
            Err(_) => continue,
        }
    }

    if matches.is_empty() {
        ToolResult::ok(format!("No files matching '{}' in {}", pattern, search_dir))
    } else {
        ToolResult::ok(matches.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_read_file_nonexistent() {
        let tool = FsTool::new();
        let result = tool
            .execute(serde_json::json!({"operation": "read_file", "path": "/nonexistent/file.txt"}))
            .await;
        assert!(!result.success);
    }

    #[tokio::test]
    async fn test_list_dir_empty() {
        let tool = FsTool::new();
        let result = tool
            .execute(serde_json::json!({"operation": "list_dir", "path": "/tmp"}))
            .await;
        assert!(result.success);
        assert!(!result.output.is_empty());
    }

    #[tokio::test]
    async fn test_missing_operation() {
        let tool = FsTool::new();
        let result = tool.execute(serde_json::json!({})).await;
        assert!(!result.success);
    }
}
