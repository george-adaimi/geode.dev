use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum Message {
    #[serde(rename = "system")]
    System { content: String },
    #[serde(rename = "user")]
    User { content: String },
    #[serde(rename = "assistant")]
    Assistant {
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
    },
    #[serde(rename = "tool")]
    Tool {
        tool_call_id: String,
        content: String,
    },
    Summary { content: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub arguments: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self::System {
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::User {
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>, tool_calls: Option<Vec<ToolCall>>) -> Self {
        Self::Assistant {
            content: Some(content.into()),
            tool_calls,
        }
    }

    pub fn assistant_text(content: impl Into<String>) -> Self {
        Self::Assistant {
            content: Some(content.into()),
            tool_calls: None,
        }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::Tool {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
        }
    }

    pub fn summary(content: impl Into<String>) -> Self {
        Self::Summary {
            content: content.into(),
        }
    }

    /// Check if this is a summary message (internal use only, not sent to LLM).
    pub fn is_summary(&self) -> bool {
        matches!(self, Message::Summary { .. })
    }

    /// Convert to a system message for API sending (summaries become system messages).
    pub fn to_api_message(&self) -> Option<Message> {
        match self {
            Message::Summary { content } => Some(Message::System {
                content: format!("[Context Summary]\n{}", content),
            }),
            m => Some(m.clone()),
        }
    }
}

impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Message::System { content } => write!(f, "System: {}", content),
            Message::User { content } => write!(f, "User: {}", content),
            Message::Assistant { content, tool_calls } => {
                if let Some(tc) = tool_calls {
                    for call in tc {
                        write!(f, "Assistant ({}): {}", call.function.name, call.function.arguments)?;
                    }
                } else if let Some(c) = content {
                    write!(f, "Assistant: {}", c)?;
                }
                Ok(())
            }
            Message::Tool { content, .. } => write!(f, "Tool: {}", content),
            Message::Summary { content } => write!(f, "[Summary] {}", content),
        }
    }
}
