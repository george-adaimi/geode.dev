use anyhow::Result;
use clap::{Parser, Subcommand};
use geode_core::config;
use std::fs;

#[derive(Parser)]
pub struct ConfigCmd {
    #[command(subcommand)]
    action: ConfigAction,
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Show current configuration
    Show,
    /// Create a new default config
    New,
}

impl ConfigCmd {
    pub fn run(&self) -> Result<()> {
        match &self.action {
            ConfigAction::Show => ConfigCmd::show(),
            ConfigAction::New => ConfigCmd::new(),
        }
    }

    fn show() -> Result<()> {
        match config::load_config() {
            Ok(cfg) => {
                let toml_str = toml::to_string_pretty(&cfg)?;
                println!("{}", toml_str);
                Ok(())
            }
            Err(_) => {
                println!("No config file found. Run 'geode config new' to create one.");
                println!("\nDefault configuration:");
                let default_cfg = config::default_config();
                let toml_str = toml::to_string_pretty(&default_cfg)?;
                println!("{}", toml_str);
                Ok(())
            }
        }
    }

    fn new() -> Result<()> {
        let cfg_path = config::config_path();
        let parent = cfg_path
            .parent()
            .expect("Config path should have a parent directory");

        if parent.exists() {
            anyhow::bail!("Config directory already exists at {}", parent.display());
        }

        fs::create_dir_all(parent)?;

        let default_cfg = config::default_config();
        let toml_str = toml::to_string_pretty(&default_cfg)?;
        fs::write(&cfg_path, &toml_str)?;

        println!("Created config file at {}", cfg_path.display());
        Ok(())
    }
}
