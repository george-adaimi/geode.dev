<div align="center">
  <img src="assets/geode_logo.png" alt="geode.dev logo" width="500"/>
</div>

# <span style="color: lightblue">geode.dev</span>

A local AI agent framework that runs as a Rust CLI. <span style="color: lightblue">geode.dev</span> connects to a local LLM served via llama.cpp and acts as an autonomous agent — planning tasks, executing tool calls, and iteratively working through multi-step objectives.

## Features

- **Continuous planning**: Generates a plan, executes steps, and replans if needed
- **Tool system**: Filesystem, shell, and web operations with safety approvals
- **Context management**: Automatic summarization to stay within the context window
- **Two modes**: Interactive REPL and single-shot CLI
- **Local-first**: Works with any OpenAI-compatible API server (llama.cpp, Ollama, etc.)

## Prerequisites

- [Rust](https://rustup.rs/) (latest stable)
- A local LLM server (e.g., [llama.cpp](https://github.com/ggerganov/llama.cpp) server) exposing an OpenAI-compatible `/v1/chat/completions` endpoint

## Installation

```bash
# Clone and build
git clone https://github.com/your-org/geode.git
cd geode
cargo install --path geode-cli
```

This installs the `geode` binary to `~/.cargo/bin`.

## Setup

### 1. Create a config file

```bash
geode config new
```

This creates `~/.geode/config.toml` with defaults:

```toml
[llm]
server_url = "http://localhost:8080"
model_path = ""
context_window = 8192
summarize_threshold = 6144

[tools]
enabled = ["fs", "shell", "web"]

[safety]
auto_approve_safe = true
```

Edit `server_url` to point to your LLM server and set `model_path` to the model identifier.

### 2. Install a model (optional)

<span style="color: lightblue">geode.dev</span> ships with a built-in model registry. To install a model:

```bash
geode model install llama3.1-8b
```

Available models:

| Name | Description |
|------|-------------|
| `llama3.1-8b` | Meta Llama 3.1 Instruct 8B, Q4 quantization |
| `llama3.1-70b` | Meta Llama 3.1 Instruct 70B, Q4 quantization |
| `mistral-7b` | Mistral v0.3 Instruct 7B, Q4 quantization |
| `phi3-mini` | Microsoft Phi-3 Mini 4K Instruct, Q4 quantization |
| `gemma-2-2b` | Google Gemma-2 2B Instruct, Q4 quantization |
| `qwen2.5-7b` | Qwen 2.5 Instruct 7B, Q4 quantization |

Installed models are stored in `~/.geode/models/`.

### 3. Customize the system prompt (optional)

Create `~/.geode/SYSTEM.md` to override the default system prompt.

## Usage

### Interactive mode (REPL)

```bash
geode
```

Starts an interactive session where you can ask questions and give instructions across multiple turns. The agent maintains conversation context and supports:

- **Tool approval**: Dangerous operations (shell commands, file writes) prompt for your approval before executing
- **Command history**: Use arrow keys to navigate past inputs
- **Built-in commands**: Type `exit` or `quit` to leave, `clear` to clear the screen

### Single-shot mode

```bash
geode "Read the README.md file and summarize the key features"
```

Runs the prompt, executes any needed tool calls, and prints the result. Use `--auto` to skip approval prompts:

```bash
geode --auto "List all files in /tmp and search for any .log files"
```

### Model management

```bash
# List installed and available models
geode model list

# Install a model
geode model install qwen2.5-7b
```

### Config management

```bash
# Show current config
geode config show

# Create a new default config
geode config new
```

### Override options

```bash
# Override server URL for a single run
geode --server http://192.168.1.100:8080 "What files are in the current directory?"

# Use a custom config file
geode --config /path/to/custom/config.toml "Summarize the main README"
```

## Tools

| Tool | Operations | Safety |
|------|-----------|--------|
| `fs` | `read_file`, `write_file`, `list_dir`, `search_files` | Safe / Warning |
| `shell` | `run_command` | Dangerous |
| `web` | `fetch_url` | Safe |

Safe tools execute immediately. Dangerous tools require explicit approval in interactive mode.

## Architecture

```
geode-cli/          # CLI binary, REPL, config/model commands
geode-core/         # Agent loop, LLM client, planning, context management
geode-tools/        # Tool implementations (fs, shell, web)
```

The agent follows a continuous planning loop:

1. Generate a plan (JSON list of steps)
2. Execute each step — calling the LLM with tool definitions, executing returned tool calls
3. Replan if any steps fail
4. Return a final answer

## Customization

- **System prompt**: Edit `~/.geode/SYSTEM.md`
- **Context window**: Set `context_window` and `summarize_threshold` in config
- **Safety**: Set `auto_approve_safe = false` to require approval for all tools
