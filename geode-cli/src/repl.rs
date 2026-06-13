use geode_core::{Agent, LlmClient, ToolRegistry};
use geode_tools::{FsTool, ShellTool, WebTool};
use std::io::{self, Write};

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

        let prompt_text = String::from("geode > ");
        let mut history: Vec<String> = Vec::new();

        loop {
            print!("{}", prompt_text);
            std::io::stdout().flush().unwrap();

            let mut line = String::new();
            if std::io::stdin().read_line(&mut line).is_err() {
                break;
            }

            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            match line {
                "exit" | "quit" => break,
                "clear" => {
                    println!("\x1b[2J\x1b[1;1H");
                    continue;
                }
                _ => {
                    history.push(line.to_string());
                    let result = self.agent.run(line).await;
                    let mut denied = false;
                    for event in result {
                        match &event {
                            geode_core::AgentEvent::TextChunk(text) => print!("{}", text),
                            geode_core::AgentEvent::PlanGenerated(plan) => {
                                println!("\n{}", plan.format_display());
                            }
                            geode_core::AgentEvent::ToolCallAboutToRun { name, args } => {
                                println!("\n[Calling: {}({})]", name, truncate(args, 100));
                            }
                            geode_core::AgentEvent::ToolCallComplete { name, output, success } => {
                                let status = if *success { "OK" } else { "FAIL" };
                                println!("[{}] {} → {}", status, name, truncate(output, 200));
                            }
                            geode_core::AgentEvent::ApprovalDenied { name, args } => {
                                println!("\n[Denied: {}({})]", name, truncate(args, 100));
                                denied = true;
                            }
                            geode_core::AgentEvent::Complete(answer) => {
                                println!("\n{}", answer);
                            }
                            geode_core::AgentEvent::Failed(err) => {
                                eprintln!("\nError: {}", err);
                            }
                        }
                    }

                    if denied {
                        print!("\nTool call was denied. What would you like to do instead? > ");
                        std::io::stdout().flush().unwrap();
                        let mut input = String::new();
                        if std::io::stdin().read_line(&mut input).is_err() {
                            break;
                        }
                        let input = input.trim();
                        if !input.is_empty() {
                            let result = self.agent.on_user_input(input).await;
                            for event in result {
                                match event {
                                    geode_core::AgentEvent::TextChunk(text) => print!("{}", text),
                                    geode_core::AgentEvent::PlanGenerated(plan) => {
                                        println!("\n{}", plan.format_display());
                                    }
                                    geode_core::AgentEvent::ToolCallAboutToRun { name, args } => {
                                        println!("\n[Calling: {}({})]", name, truncate(args, 100));
                                    }
                                    geode_core::AgentEvent::ToolCallComplete { name, output, success } => {
                                        let status = if success { "OK" } else { "FAIL" };
                                        println!("[{}] {} → {}", status, name, truncate(&output, 200));
                                    }
                                    geode_core::AgentEvent::ApprovalDenied { name, args } => {
                                        println!("\n[Denied: {}({})]", name, truncate(args, 100));
                                    }
                                    geode_core::AgentEvent::Complete(answer) => {
                                        println!("\n{}", answer);
                                    }
                                    geode_core::AgentEvent::Failed(err) => {
                                        eprintln!("\nError: {}", err);
                                    }
                                }
                            }
                        }
                    }

                    println!();
                }
            }
        }
    }
}

fn approval_callback(_tool_name: &str, _args: &str) -> bool {
    print!("\n[Approval required] Execute this tool call? (yes/no) > ");
    std::io::stdout().flush().unwrap();

    let mut input = String::new();
    if std::io::stdin().read_line(&mut input).is_err() {
        return false;
    }

    matches!(input.trim(), "y" | "yes" | "Y" | "YES")
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
