/// Token counter for estimating context window usage.
///
/// Uses a simple byte-pair encoding approximation:
/// - Each word (whitespace-separated) ≈ 1-2 tokens
/// - Non-ASCII characters ≈ 1 token each
///
/// For production use with specific models, replace with the model's
/// actual tokenizer (e.g., tiktoken, llama-cpp-rs).
pub struct Tokenizer;

impl Tokenizer {
    pub fn new() -> Self {
        Self
    }

    /// Estimate token count for a string.
    /// This is an approximation — actual counts vary by model tokenizer.
    pub fn count(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }

        let mut count = 0;
        for c in text.chars() {
            if c.is_ascii() {
                count += 1;
            } else {
                count += 2;
            }
        }
        // BPE-style: words are typically 1-2 tokens
        // Divide by ~4 to approximate BPE tokenization
        (count / 4).max(1)
    }

    /// Estimate total tokens for a list of messages.
    pub fn count_messages(&self, messages: &[crate::message::Message]) -> usize {
        let mut total = 0;
        // Per-message overhead
        total += 3;
        for msg in messages {
            total += self.count_message(msg);
        }
        // Per-message overhead for formatting
        total += messages.len() * 4;
        total
    }

    fn count_message(&self, msg: &crate::message::Message) -> usize {
        let text = match msg {
            crate::message::Message::System { content } => content,
            crate::message::Message::User { content } => content,
            crate::message::Message::Assistant { content, tool_calls } => {
                let base = content.as_deref().unwrap_or("");
                let mut parts = vec![base];
                if let Some(tc) = tool_calls {
                    for call in tc {
                        parts.push(&call.function.name);
                        parts.push(&call.function.arguments);
                    }
                }
                &parts.join(" ")
            }
            crate::message::Message::Tool { content, .. } => content,
            crate::message::Message::Summary { content } => content,
        };
        self.count(text)
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;

    #[test]
    fn test_count_empty() {
        let tok = Tokenizer::new();
        assert_eq!(tok.count(""), 0);
    }

    #[test]
    fn test_count_non_empty() {
        let tok = Tokenizer::new();
        let count = tok.count("Hello, world! This is a test.");
        assert!(count > 0);
        // A 38-char ASCII string should be a small positive number
        assert!(count <= 50);
    }

    #[test]
    fn test_count_messages() {
        let tok = Tokenizer::new();
        let messages = vec![
            Message::system("You are helpful."),
            Message::user("Hello"),
        ];
        let count = tok.count_messages(&messages);
        assert!(count > 0);
    }

    #[test]
    fn test_count_roughly_proportional() {
        let tok = Tokenizer::new();
        let short = tok.count("hi");
        let long = tok.count("hi there friend how are you doing today");
        assert!(long > short);
    }
}
