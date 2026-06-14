mod commands;
mod repl;

use clap::{Parser, Subcommand};
use geode_core::{Agent, LlmClient};
use tokio::sync::mpsc;

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

        let (tx, mut rx) = mpsc::channel(64);
        let prompt = prompt.clone();
        let driver = tokio::spawn(async move {
            agent.run(&prompt, &tx).await
        });
        while let Some(event) = rx.recv().await {
            repl::emit_event(&event);
        }
        let driver_result = driver.await;
        match driver_result {
            Ok(Err(e)) => {
                eprintln!("Agent error: {}", e);
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("Agent driver error: {}", e);
                std::process::exit(1);
            }
            Ok(Ok(())) => {}
        }
    } else {
        // REPL mode — interactive approval
        let mut repl = repl::Repl::new(&server_url, &model_name);
        repl.run().await;
    }
}
