use async_trait::async_trait;
use geode_core::{SafetyLevel, Tool, ToolResult};

pub struct WebTool;

impl WebTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for WebTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for WebTool {
    fn name(&self) -> &str {
        "web"
    }

    fn description(&self) -> &str {
        "Web operations: fetch_url - fetch a URL and return readable text"
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "format": "uri",
                    "description": "The URL to fetch"
                }
            },
            "required": ["url"]
        })
    }

    fn safety_level(&self) -> SafetyLevel {
        SafetyLevel::Safe
    }

    async fn execute(&self, args: serde_json::Value) -> ToolResult {
        let url = match args.get("url").and_then(|v| v.as_str()) {
            Some(u) => u,
            None => return ToolResult::err("", "Missing required 'url' field"),
        };

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build();

        let client = match client {
            Ok(c) => c,
            Err(e) => return ToolResult::err("", format!("Failed to build HTTP client: {}", e)),
        };

        let response = match client.get(url).send().await {
            Ok(r) => r,
            Err(e) => {
                return ToolResult::err(
                    "",
                    format!("Failed to fetch {}: {}", url, e),
                );
            }
        };

        if !response.status().is_success() {
            return ToolResult::err(
                "",
                format!("HTTP {}: {}", response.status(), response.status().canonical_reason().unwrap_or("Unknown")),
            );
        }

        let body = match response.text().await {
            Ok(b) => b,
            Err(e) => {
                return ToolResult::err(
                    "",
                    format!("Failed to read response body: {}", e),
                );
            }
        };

        let text = strip_html(&body);
        ToolResult::ok(text)
    }
}

fn strip_html(html: &str) -> String {
    // Remove HTML tags
    let no_tags = HTML_TAG_RE.replace_all(html, "");
    // Decode common HTML entities
    let decoded = html_entity_decode(&no_tags);
    // Collapse whitespace
    decoded
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

static HTML_TAG_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"<[^>]+>").unwrap());

fn html_entity_decode(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
        .replace("&apos;", "'")
        .replace("&ndash;", "-")
        .replace("&mdash;", "--")
        .replace("&laquo;", "<<")
        .replace("&raquo;", ">>")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_html() {
        let html = "<p>Hello <b>world</b>!</p>";
        let text = strip_html(html);
        assert_eq!(text, "Hello world!");
    }

    #[test]
    fn test_strip_html_entities() {
        let html = "&lt;script&gt; &amp; &quot;quotes&quot;";
        let text = strip_html(html);
        assert!(text.contains("<script>"));
        assert!(text.contains("&"));
        assert!(text.contains("quotes"));
    }

    #[test]
    fn test_strip_html_empty() {
        assert_eq!(strip_html(""), "");
    }
}
