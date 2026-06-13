use crate::message::{Message, ToolCall};
use crate::tokenizer::Tokenizer;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::time::Duration;

pub struct LlmClient {
    base_url: String,
    client: reqwest::Client,
    tokenizer: Tokenizer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<ChatChoice>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatChoice {
    pub message: ChatMessage,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ParsedToolCall>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ParsedToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ParsedFunction,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ParsedFunction {
    pub name: String,
    pub arguments: String,
}

impl From<ParsedToolCall> for ToolCall {
    fn from(tc: ParsedToolCall) -> Self {
        Self {
            id: tc.id,
            call_type: tc.call_type,
            function: crate::message::ToolFunction {
                name: tc.function.name,
                arguments: tc.function.arguments,
            },
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct StreamChunk {
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamChoice {
    pub delta: StreamDelta,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamDelta {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<StreamToolCall>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamToolCall {
    pub index: usize,
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub call_type: Option<String>,
    pub function: Option<StreamFunction>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamFunction {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

impl LlmClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("Failed to build HTTP client"),
            tokenizer: Tokenizer::new(),
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Count tokens for a set of messages.
    pub fn count_tokens(&self, messages: &[Message]) -> usize {
        self.tokenizer.count_messages(messages)
    }

    /// Non-streaming chat completion.
    pub async fn chat(&self, request: ChatRequest) -> reqwest::Result<ChatResponse> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?
            .json::<ChatResponse>()
            .await?;
        Ok(response)
    }

    /// Streaming chat completion — returns an async stream of chunks.
    pub async fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> reqwest::Result<impl futures::Stream<Item = reqwest::Result<StreamChunk>>> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let res = self
            .client
            .post(&url)
            .json(&request)
            .header("Accept", "text/event-stream")
            .send()
            .await?;

        Ok(res.bytes_stream().map(|chunk| {
            chunk.map(|bytes| {
                let text = String::from_utf8(bytes.to_vec()).unwrap_or_default();
                // Return first chunk if any parsed, otherwise empty
                let chunks = parse_stream_chunks(&text);
                if chunks.is_empty() {
                    StreamChunk { choices: vec![] }
                } else {
                    chunks.into_iter().next().unwrap_or_default()
                }
            })
        }))
    }
}

/// Parse SSE stream text into individual chunk events.
fn parse_stream_chunks(text: &str) -> Vec<StreamChunk> {
    let mut chunks = Vec::new();
    for line in text.lines() {
        if !line.starts_with("data: ") {
            continue;
        }
        let data = &line[6..];
        if data.trim() == "[DONE]" {
            break;
        }
        if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data.trim()) {
            chunks.push(chunk);
        }
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_stream_chunks() {
        let sse = r#"data: {"choices":[{"delta":{"content":"hello"}}]}
data: {"choices":[{"delta":{"content":" world"}}]}
data: [DONE]
"#;
        let chunks = parse_stream_chunks(sse);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].choices[0].delta.content, Some("hello".to_string()));
        assert_eq!(chunks[1].choices[0].delta.content, Some(" world".to_string()));
    }

    #[test]
    fn test_parse_stream_empty() {
        assert!(parse_stream_chunks("").is_empty());
    }

    #[test]
    fn test_parse_stream_done_only() {
        assert!(parse_stream_chunks("data: [DONE]\n").is_empty());
    }

    #[test]
    fn test_message_serialization() {
        let msg = Message::user("hello");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"user\""));
        assert!(json.contains("hello"));
    }

    #[test]
    fn test_message_tool_result_serialization() {
        let msg = Message::tool_result("call123", "file contents here");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"tool\""));
        assert!(json.contains("call123"));
    }
}
