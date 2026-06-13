mod commands;
mod repl;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "geode", about = "Local AI agent framework")]
struct Cli {
    /// Skip safety approvals
    #[arg(long)]
    auto: bool,

    /// Custom config path
    #[arg(long)]
    config: Option<String>,

    /// Override server URL
    #[arg(long)]
    server: Option<String>,

    /// Single-shot prompt (enter REPL if omitted)
    #[arg(trailing_var_arg = true)]
    prompt: Vec<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Model management
    Model(commands::model::ModelSubCmd),
    /// Config management
    #[command(name = "config")]
    Cfg(commands::config::ConfigCmd),
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Model(cmd)) => {
            if let Err(e) = cmd.run().await {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
            return;
        }
        Some(Commands::Cfg(cmd)) => {
            if let Err(e) = cmd.run() {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
            return;
        }
        None => {}
    }

    // Load config
    let config = match geode_core::config::load_config() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("No config found. Run 'geode config new' to create one.");
            std::process::exit(1);
        }
    };

    let server_url = cli.server.unwrap_or(config.llm.server_url);
    let model_name = config.llm.model_path;

    if let Some(prompt) = cli.prompt.first() {
        // Single-shot mode — no approval prompts (auto-approve)
        let mut registry = geode_core::ToolRegistry::new();
        registry.register(Box::new(geode_tools::FsTool::new()));
        registry.register(Box::new(geode_tools::ShellTool::new()));
        registry.register(Box::new(geode_tools::WebTool::new()));

        let llm = LlmClient::new(&server_url);
        let system_prompt = repl::build_system_prompt(&registry);
        let mut agent = Agent::new(llm, registry, system_prompt, None::<fn(&str, &str) -> bool>, &model_name);

        let result = agent.run(prompt).await;
        for event in result {
            match event {
                geode_core::AgentEvent::TextChunk(text) => print!("{}", text),
                geode_core::AgentEvent::PlanGenerated(plan) => {
                    println!("\n{}", plan.format_display());
                }
                geode_core::AgentEvent::ToolCallAboutToRun { name, args } => {
                    println!("\n[Calling: {}({})]", name, repl::truncate(&args, 100));
                }
                geode_core::AgentEvent::ToolCallComplete { name, output, success } => {
                    let status = if success { "OK" } else { "FAIL" };
                    println!("[{}] {} → {}", status, name, repl::truncate(&output, 200));
                }
                geode_core::AgentEvent::ApprovalDenied { name, args } => {
                    println!("\n[Denied: {}({})]", name, repl::truncate(&args, 100));
                }
                geode_core::AgentEvent::Complete(answer) => {
                    println!("\n{}", answer);
                }
                geode_core::AgentEvent::Failed(err) => {
                    eprintln!("\nError: {}", err);
                }
            }
        }
    } else {
        // REPL mode — interactive approval
        let mut repl = repl::Repl::new(&server_url, &model_name);
        repl.run().await;
    }
}

use geode_core::{Agent, LlmClient};
