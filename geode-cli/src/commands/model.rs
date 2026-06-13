use anyhow::Result;
use clap::{Parser, Subcommand};
use futures::StreamExt;
use geode_core::model_registry;
use std::fs;
use std::io::Write;

#[derive(Parser)]
pub struct ModelSubCmd {
    #[command(subcommand)]
    action: ModelAction,
}

#[derive(Subcommand)]
enum ModelAction {
    /// Install a model from the registry
    Install(ModelInstallCmd),
    /// List installed models
    List,
}

#[derive(Parser)]
struct ModelInstallCmd {
    /// Model name (e.g., llama3.1-8b)
    name: String,
}

impl ModelSubCmd {
    pub async fn run(&self) -> Result<()> {
        match &self.action {
            ModelAction::Install(cmd) => cmd.run().await,
            ModelAction::List => Self::list_installed(),
        }
    }

    fn list_installed() -> Result<()> {
        let models_dir = geode_core::config::models_dir();
        if !models_dir.exists() {
            println!("No models installed.");
            println!("Available models:");
            for m in model_registry::list_available_models() {
                println!("  {} - {}", m.name, m.description);
            }
            return Ok(());
        }

        let entries = fs::read_dir(&models_dir)?;
        let mut gguf_files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|ext| ext == "gguf").unwrap_or(false))
            .collect();

        if gguf_files.is_empty() {
            println!("No models installed.");
        } else {
            println!("Installed models:");
            for entry in &mut gguf_files {
                let metadata = entry.metadata()?;
                let size = metadata.len();
                println!(
                    "  {} ({:.1} MB)",
                    entry.file_name().to_string_lossy(),
                    size as f64 / 1_000_000.0
                );
            }
        }

        println!("\nAvailable models:");
        for m in model_registry::list_available_models() {
            println!("  {} - {}", m.name, m.description);
        }

        Ok(())
    }
}

impl ModelInstallCmd {
    async fn run(&self) -> Result<()> {
        let model = model_registry::get_model(&self.name)
            .ok_or_else(|| anyhow::anyhow!("Unknown model: {}. Run 'geode model list' for available models.", self.name))?;

        println!("Downloading {}...", model.name);
        println!("  From: {}", model.hf_url);

        let models_dir = geode_core::config::models_dir();
        fs::create_dir_all(&models_dir)?;

        let file_path = models_dir.join(format!("{}.gguf", self.name));

        let client = reqwest::Client::new();
        let response = client.get(&model.hf_url).send().await?;

        if !response.status().is_success() {
            anyhow::bail!(
                "Failed to download model: HTTP {}",
                response.status()
            );
        }

        let total_size = response.content_length().unwrap_or(0);
        let mut file = fs::File::create(&file_path)?;
        let mut downloaded: u64 = 0;
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let len = chunk.len() as u64;
            file.write_all(&chunk)?;
            downloaded += len;
            if total_size > 0 {
                let progress = (downloaded as f64 / total_size as f64) * 100.0;
                print!("\r  Progress: {:.1}% ({:.1} MB / {:.1} MB)",
                    progress,
                    downloaded as f64 / 1_000_000.0,
                    total_size as f64 / 1_000_000.0);
            }
        }
        println!();

        let file_size = file_path.metadata()?.len();
        println!("Downloaded {} ({:.1} MB)", file_path.display(), file_size as f64 / 1_000_000.0);

        Ok(())
    }
}
