use geode_core::{Agent, LlmClient, ToolRegistry};
use geode_tools::{FsTool, ShellTool, WebTool};
use geode_core::AgentEvent;
use rustyline::error::ReadlineError;
use std::io::Write;
use rustyline::history::DefaultHistory;
use rustyline::Editor;
use tokio::sync::mpsc;

pub fn emit_event(event: &AgentEvent) {
    match event {
        AgentEvent::TextChunk(text) => {
            print!("{}", text);
            let _ = std::io::stdout().flush();
        }
        AgentEvent::PlanGenerated(plan) => {
            println!("\n{}", plan.format_display());
        }
        AgentEvent::ToolCallAboutToRun { name, args } => {
            println!("\n[Calling: {}({})]", name, truncate(args, 100));
        }
        AgentEvent::ToolCallComplete { name, output, success } => {
            let status = if *success { "OK" } else { "FAIL" };
            println!("[{}] {} → {}", status, name, truncate(output, 200));
        }
        AgentEvent::ApprovalDenied { name, args } => {
            println!("\n[Denied: {}({})]", name, truncate(args, 100));
        }
        AgentEvent::Complete(answer) => {
            println!("\n{}", answer);
        }
        AgentEvent::Failed(err) => {
            eprintln!("\nError: {}", err);
        }
    }
}

pub struct Repl {
    agent: Agent,
}

impl Repl {
    pub fn new(server_url: &str, model_name: &str) -> Self {
        let llm = LlmClient::new(server_url);
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(FsTool::new()));
        registry.register(Box::new(ShellTool::new()));
        registry.register(Box::new(WebTool::new()));

        let system_prompt = build_system_prompt(&registry);

        let agent = Agent::new(llm, registry, system_prompt, Some(approval_callback), model_name);

        Self { agent }
    }

    pub async fn run(&mut self) {
        println!("Geode REPL (type 'exit' or 'quit' to quit)");
        println!();

        let history_path = geode_core::config::config_dir().join("repl_history.txt");
        let mut rl = match Editor::<(), DefaultHistory>::new() {
            Ok(rl) => rl,
            Err(e) => {
                eprintln!("Failed to initialize REPL editor: {}", e);
                return;
            }
        };
        let _ = rl.load_history(&history_path);

        loop {
            let readline = rl.readline("geode > ");
            let line = match readline {
                Ok(line) => line,
                Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
                Err(e) => {
                    eprintln!("Readline error: {}", e);
                    break;
                }
            };

            let trimmed = line.trim().to_string();
            if trimmed.is_empty() {
                continue;
            }

            let _ = rl.add_history_entry(&trimmed);

            match trimmed.as_str() {
                "exit" | "quit" => break,
                "clear" => {
                    println!("\x1b[2J\x1b[1;1H");
                    continue;
                }
                _ => {
                    let (tx, rx) = mpsc::channel(64);
                    let printer = tokio::spawn(async move {
                        let mut rx = rx;
                        let mut events = Vec::new();
                        while let Some(event) = rx.recv().await {
                            emit_event(&event);
                            events.push(event);
                        }
                        events
                    });
                    self.agent.run(&trimmed, &tx).await.unwrap();
                    drop(tx);
                    let events = printer.await.unwrap();
                    let denied = events.iter().any(|e| matches!(e, AgentEvent::ApprovalDenied { .. }));

                    if denied {
                        print!("\nTool call was denied. What would you like to do instead? > ");
                        std::io::stdout().flush().unwrap();
                        let mut input = String::new();
                        if std::io::stdin().read_line(&mut input).is_err() {
                            break;
                        }
                        let input = input.trim().to_string();
                        if !input.is_empty() {
                            let (tx, rx) = mpsc::channel(64);
                            let printer = tokio::spawn(async move {
                                let mut rx = rx;
                                let mut events = Vec::new();
                                while let Some(event) = rx.recv().await {
                                    emit_event(&event);
                                    events.push(event);
                                }
                                events
                            });
                            self.agent.on_user_input(&input, &tx).await.unwrap();
                            drop(tx);
                            let _ = printer.await.unwrap();
                        }
                    }

                    println!();
                }
            }
        }

        let _ = rl.save_history(&history_path);
    }
}

fn approval_callback(tool_name: &str, args: &str) -> bool {
    let display = truncate_middle(args, 120);
    print!("\n[Approval required] Execute `{tool_name}({display})`? (yes/no) > ");
    std::io::stdout().flush().unwrap();

    let mut input = String::new();
    if std::io::stdin().read_line(&mut input).is_err() {
        return false;
    }

    matches!(input.trim(), "y" | "yes" | "Y" | "YES")
}

fn truncate_middle(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let half = max_len / 2;
        format!("{}...{}", &s[..half], &s[s.len() - half..])
    }
}

pub fn load_system_prompt() -> String {
    let path = geode_core::config::system_prompt_path();
    if path.exists() {
        if let Ok(content) = std::fs::read_to_string(&path) {
            if !content.trim().is_empty() {
                return content;
            }
        }
    }

    // Default system prompt template
    "You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
${toolsList}

In addition to the tools above, you may have access to other custom tools depending on the project.

Guidelines:
${guidelines}

After producing the plan, each step will be executed automatically. You will then respond with the final answer.

Current date: ${date}
Current working directory: ${promptCwd}"
        .to_string()
}

pub fn build_system_prompt(registry: &ToolRegistry) -> String {
    let template = load_system_prompt();
    let tools_list = registry.to_tool_list();
    let guidelines = "Use bash for file operations like ls, rg, find.\n\
        Be concise in your responses.\n\
        Show file paths clearly when working with files.";
    let date = chrono::Local::now().format("%Y-%m-%d").to_string();
    let cwd = std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    template
        .replace("${toolsList}", &tools_list)
        .replace("${guidelines}", guidelines)
        .replace("${date}", &date)
        .replace("${promptCwd}", &cwd)
}

pub fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
