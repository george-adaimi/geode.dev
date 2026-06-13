/// Compress a list of message descriptions into a single summary.
///
/// In v1, this uses a simple heuristic: one sentence per message.
/// In future versions, this can be replaced with an LLM-based summarizer.
pub fn compress_messages(messages: &[String]) -> String {
    if messages.is_empty() {
        return String::new();
    }

    let summaries: Vec<String> = messages.iter().map(|msg| compress_message(msg)).collect();
    summaries.join(" ")
}

/// Compress a single message description into one sentence.
pub fn compress_message(msg: &str) -> String {
    // Simple heuristic: take the first line, truncate if too long
    let first_line = msg.lines().next().unwrap_or(msg);
    let trimmed = first_line.trim();

    // Truncate to ~120 chars if too long
    if trimmed.len() > 120 {
        format!("{}...", &trimmed[..117])
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_empty() {
        assert!(compress_messages(&[]).is_empty());
    }

    #[test]
    fn test_compress_single() {
        let msgs = vec!["User: hello".to_string()];
        let summary = compress_messages(&msgs);
        assert!(summary.contains("hello"));
    }

    #[test]
    fn test_compress_multiple() {
        let msgs = vec![
            "User: what files are here".to_string(),
            "Tool: listed 3 files".to_string(),
        ];
        let summary = compress_messages(&msgs);
        assert!(summary.contains("what files are here"));
        assert!(summary.contains("listed 3 files"));
    }

    #[test]
    fn test_compress_message_truncation() {
        let long = "a".repeat(200);
        let compressed = compress_message(&long);
        assert!(compressed.len() <= 120);
        assert!(compressed.ends_with("..."));
    }

    #[test]
    fn test_compress_message_short() {
        let short = "User: hi";
        assert_eq!(compress_message(short), "User: hi");
    }
}
