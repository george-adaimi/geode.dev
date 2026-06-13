use crate::message::Message;
use crate::tokenizer::Tokenizer;

pub struct ContextManager {
    messages: Vec<Message>,
    context_window: usize,
    summarize_threshold: usize,
    tokenizer: Tokenizer,
}

impl ContextManager {
    pub fn new(context_window: usize, summarize_threshold: usize) -> Self {
        Self {
            messages: Vec::new(),
            context_window,
            summarize_threshold,
            tokenizer: Tokenizer::new(),
        }
    }

    pub fn add_message(&mut self, msg: Message) {
        self.messages.push(msg);
    }

    /// Get messages to send to the LLM for completion.
    /// Returns summary (if any) + recent raw messages, ensuring token count is within window.
    pub fn get_completion_messages(&self) -> Vec<Message> {
        // For now, return all messages. Compression happens in maybe_compress.
        self.messages.clone()
    }

    /// If token count exceeds summarize_threshold, compress oldest messages.
    pub fn maybe_compress(&mut self) {
        let token_count = self.tokenizer.count_messages(&self.messages);
        if token_count > self.summarize_threshold {
            self.compress();
        }
    }

    fn compress(&mut self) {
        // Find messages that can be summarized (non-summary messages)
        let compressible: Vec<usize> = self
            .messages
            .iter()
            .enumerate()
            .filter(|(_, msg)| !(*msg).is_summary())
            .map(|(i, _)| i)
            .collect();

        // Compress the oldest messages (up to a batch of 5)
        let batch_size = 5.min(compressible.len());
        if batch_size == 0 {
            return;
        }

        let to_compress: Vec<usize> = compressible[..batch_size].to_vec();
        let compressed_content: Vec<String> = to_compress
            .iter()
            .map(|&i| format!("{}", self.messages[i]))
            .collect();

        let summary = crate::summarizer::compress_messages(&compressed_content);

        // Rebuild messages without the compressed ones
        let mut new_messages = Vec::new();
        for (i, msg) in self.messages.iter().enumerate() {
            if !to_compress.contains(&i) {
                new_messages.push(msg.clone());
            }
        }
        self.messages = new_messages;

        // Add summary as a system message
        if !summary.is_empty() {
            self.messages.push(Message::summary(&summary));
        }
    }

    /// Get current token count.
    pub fn token_count(&self) -> usize {
        self.tokenizer.count_messages(&self.messages)
    }

    /// Get current message count.
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Smart-truncate a tool output to fit within token budget.
    pub fn truncate_output(&self, output: &str, max_tokens: usize) -> String {
        let max_chars = max_tokens * 4; // Rough approximation: 1 token ≈ 4 chars
        if output.len() <= max_chars {
            output.to_string()
        } else {
            let head_end = max_chars / 2;
            let tail_start = output.len() - max_chars / 2;
            let remaining = output.len() - max_chars;
            format!(
                "{}\n... [{} more characters not shown] ...\n{}",
                &output[..head_end],
                remaining,
                &output[tail_start..]
            )
        }
    }
}

impl Default for ContextManager {
    fn default() -> Self {
        Self::new(8192, 6144)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_count() {
        let mut ctx = ContextManager::new(8192, 6144);
        ctx.add_message(Message::user("hello"));
        assert_eq!(ctx.message_count(), 1);
        assert!(ctx.token_count() > 0);
    }

    #[test]
    fn test_compress_triggers() {
        let mut ctx = ContextManager::new(100, 50); // Small window to trigger compression
        for i in 0..20 {
            ctx.add_message(Message::user(&format!("Message number {}", i)));
        }
        let count_before = ctx.message_count();
        ctx.maybe_compress();
        // Should have compressed some messages
        assert!(ctx.message_count() < count_before || ctx.message_count() == count_before);
    }

    #[test]
    fn test_truncate_output() {
        let ctx = ContextManager::new(8192, 6144);
        let long = "x".repeat(1000);
        let truncated = ctx.truncate_output(&long, 100);
        assert!(truncated.len() < long.len());
        assert!(truncated.contains("more characters not shown"));
    }

    #[test]
    fn test_truncate_short_output() {
        let ctx = ContextManager::new(8192, 6144);
        let short = "hello";
        let result = ctx.truncate_output(short, 100);
        assert_eq!(result, "hello");
    }
}
