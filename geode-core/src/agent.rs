use crate::context::ContextManager;
use crate::llm_client::{ChatChoice, ChatMessage, ChatRequest, ChatResponse, LlmClient, ParsedFunction, ParsedToolCall};
use crate::message::{Message, ToolCall, ToolFunction};
use crate::planning::{Plan, PlanStep};
use crate::tool::{SafetyLevel, ToolResult};
use crate::tool_registry::ToolRegistry;
use anyhow::Result;
use futures::StreamExt;
use tokio::sync::mpsc::Sender;

pub struct Agent {
    llm: LlmClient,
    registry: ToolRegistry,
    context: ContextManager,
    system_prompt: String,
    on_approve: Option<Box<dyn Fn(&str, &str) -> bool + Send + Sync>>,
    model_name: String,
}

#[derive(Debug)]
pub enum AgentEvent {
    TextChunk(String),
    PlanGenerated(Plan),
    ToolCallAboutToRun { name: String, args: String },
    ToolCallComplete { name: String, output: String, success: bool },
    ApprovalDenied { name: String, args: String },
    Complete(String),
    Failed(String),
}

/// Outcome returned from `execute_step` to replace event-buffer reads.
#[derive(Debug, Default)]
pub struct StepOutcome {
    pub denied: Option<(String, String)>,
    pub had_tool_calls: bool,
}

pub type AgentResult = Vec<AgentEvent>;

impl Agent {
    pub fn new<F>(llm: LlmClient, registry: ToolRegistry, system_prompt: String, on_approve: Option<F>, model_name: impl Into<String>) -> Self
    where
        F: Fn(&str, &str) -> bool + Send + Sync + 'static,
    {
        Self {
            llm,
            registry,
            context: ContextManager::new(8192, 6144),
            system_prompt,
            on_approve: on_approve.map(|f| Box::new(f) as Box<dyn Fn(&str, &str) -> bool + Send + Sync>),
            model_name: model_name.into(),
        }
    }

    pub async fn run(&mut self, user_prompt: &str, tx: &Sender<AgentEvent>) -> Result<()> {
        self.context.add_message(Message::user(user_prompt));

        // Generate initial plan
        let plan = match self.build_messages_for_plan() {
            Ok(msgs) => match self.send_to_llm_for_plan(&msgs).await {
                Ok(p) => p,
                Err(_) => {
                    // Plan generation failed — fall back to direct LLM call with tools
                    return self.fallback_answer(user_prompt, tx).await;
                }
            },
            Err(e) => {
                let _ = tx.send(AgentEvent::Failed(format!("Failed to build context: {}", e))).await;
                return Ok(());
            }
        };
        let _ = tx.send(AgentEvent::PlanGenerated(plan.clone())).await;

        let mut current_plan = plan;
        for _round in 0..10 {
            if current_plan.all_complete() {
                break;
            }

            if current_plan.any_failed() {
                let _ = tx.send(AgentEvent::TextChunk("Replanning due to failed step...\n".to_string())).await;
                current_plan = match self.build_messages_for_plan() {
                    Ok(msgs) => match self.send_to_llm_for_plan(&msgs).await {
                        Ok(p) => p,
                        Err(e) => {
                            let _ = tx.send(AgentEvent::Failed(format!("Failed to replan: {}", e))).await;
                            return Ok(());
                        }
                    },
                    Err(e) => {
                        let _ = tx.send(AgentEvent::Failed(format!("Failed to build context: {}", e))).await;
                        return Ok(());
                    }
                };
                continue;
            }

            let step_idx = match current_plan.next_pending_step() {
                Some(idx) => idx,
                None => break,
            };

            current_plan.set_step_running(step_idx);
            let step_desc = current_plan.steps[step_idx].description.clone();
            let _ = tx.send(AgentEvent::TextChunk(format!("[Step {}] {}\n", step_idx + 1, step_desc))).await;

            let step = current_plan.steps[step_idx].clone();
            let outcome = self.execute_step(&step, tx).await;
            match outcome {
                Ok(outcome) => match outcome.denied {
                    Some((name, args)) => {
                        let _ = tx.send(AgentEvent::ApprovalDenied { name, args }).await;
                        return Ok(());
                    }
                    None => {
                        if !outcome.had_tool_calls {
                            current_plan.set_step_complete(step_idx, "Completed via text response");
                        }
                    }
                },
                Err(e) => {
                    let _ = tx.send(AgentEvent::TextChunk(format!("Step failed: {}\n", e))).await;
                    current_plan.set_step_failed(step_idx, e.to_string());
                }
            }
        }

        // Streaming final answer — synthesizes step work into one response.
        if let Err(e) = self.get_final_answer(tx).await {
            let _ = tx.send(AgentEvent::Failed(format!("Failed to get final answer: {}", e))).await;
        }
        let _ = tx.send(AgentEvent::Complete(String::new())).await;
        Ok(())
    }

    fn build_messages_for_plan(&self) -> Result<Vec<Message>> {
        let mut messages = self.context.get_completion_messages();
        messages = messages
            .into_iter()
            .filter_map(|m| m.to_api_message())
            .collect();
        messages.insert(0, Message::system(self.system_prompt.clone()));
        Ok(messages)
    }

    /// Build all messages from context for an LLM call (step execution, final answer, etc.).
    fn build_messages_for_llm(&self) -> Vec<Message> {
        let mut messages = self.context.get_completion_messages();
        messages = messages
            .into_iter()
            .filter_map(|m| m.to_api_message())
            .collect();
        messages.insert(0, Message::system(self.system_prompt.clone()));
        messages
    }

    async fn send_to_llm_for_plan(&self, messages: &[Message]) -> Result<Plan> {
        let request = ChatRequest {
            model: self.model_name.clone(),
            messages: messages.to_vec(),
            tools: None,
            tool_choice: None,
            stream: false,
        };

        let response = self.llm.chat(request).await?;

        if let Some(choice) = response.choices.first() {
            if let Some(content) = &choice.message.content {
                if let Ok(plan) = self.parse_plan_from_text(content) {
                    return Ok(plan);
                }
            }
        }

        anyhow::bail!("Failed to parse plan from LLM response")
    }

    fn parse_plan_from_text(&self, text: &str) -> Result<Plan> {
        let text = text.trim();
        if text.starts_with('[') {
            let steps: Vec<PlanStep> = serde_json::from_str(text)?;
            return Ok(Plan::new(steps));
        }
        if let Some(start) = text.find("```") {
            let rest = &text[start + 3..];
            // Skip language tag like "json\n"
            let rest = if let Some(newline) = rest.find('\n') {
                &rest[newline + 1..]
            } else {
                rest
            };
            if let Some(end) = rest.find("```") {
                let json_str = &rest[..end].trim();
                let steps: Vec<PlanStep> = serde_json::from_str(json_str)?;
                return Ok(Plan::new(steps));
            }
        }
        // Fallback: try to find a JSON array anywhere in the text
        if let Some(start) = text.find('[') {
            if let Some(end) = text.rfind(']') {
                if end > start {
                    if let Ok(steps) = serde_json::from_str(&text[start..=end]) {
                        return Ok(Plan::new(steps));
                    }
                }
            }
        }
        anyhow::bail!("Failed to parse plan from text")
    }

    /// Send a streaming LLM request, emitting TextChunk events as text arrives.
    /// Returns the complete ChatResponse (same contract as a non-streaming call).
    async fn chat_streaming(
        &self,
        mut request: ChatRequest,
        tx: &Sender<AgentEvent>,
    ) -> Result<ChatResponse> {
        request.stream = true;
        let mut stream = self.llm.chat_stream(request).await?;

        let mut full_content = String::new();
        let mut tool_calls: Vec<Option<ParsedToolCall>> = Vec::new();

        while let Some(chunks) = stream.next().await {
            let chunks = chunks?;
            for chunk in chunks {
                for choice in chunk.choices {
                    if let Some(content) = choice.delta.content {
                        if !content.is_empty() {
                            full_content.push_str(&content);
                            let _ = tx.send(AgentEvent::TextChunk(content)).await;
                        }
                    }

                    if let Some(tc_list) = choice.delta.tool_calls {
                        for tc in tc_list {
                            let index = tc.index;
                            while tool_calls.len() <= index {
                                tool_calls.push(None);
                            }
                            let entry = tool_calls[index].get_or_insert_with(|| ParsedToolCall {
                                id: String::new(),
                                call_type: "function".to_string(),
                                function: ParsedFunction {
                                    name: String::new(),
                                    arguments: String::new(),
                                },
                            });
                            if let Some(id) = tc.id {
                                entry.id = id;
                            }
                            if let Some(ct) = tc.call_type {
                                entry.call_type = ct;
                            }
                            if let Some(name) = tc.function.as_ref().and_then(|f| f.name.clone()) {
                                entry.function.name = name;
                            }
                            if let Some(args) = tc.function.as_ref().and_then(|f| f.arguments.clone()) {
                                entry.function.arguments.push_str(&args);
                            }
                        }
                    }
                }
            }
        }

        let final_tool_calls: Option<Vec<ParsedToolCall>> = if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls.into_iter().filter_map(|t| t).collect())
        };

        Ok(ChatResponse {
            choices: vec![ChatChoice {
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: if full_content.is_empty() { None } else { Some(full_content) },
                    tool_calls: final_tool_calls,
                },
            }],
        })
    }

    async fn execute_step(
        &mut self,
        step: &PlanStep,
        tx: &Sender<AgentEvent>,
    ) -> Result<StepOutcome> {
        self.context.add_message(Message::user(format!(
            "Execute this step: '{}'. Use the appropriate tool.",
            step.description
        )));

        let mut outcome = StepOutcome::default();

        // Non-streaming loop — just show tool calls, no text output yet.
        for _ in 0..10 {
            let messages = self.build_messages_for_llm();

            let request = ChatRequest {
                model: self.model_name.clone(),
                messages,
                tools: Some(self.registry.to_function_definitions()),
                tool_choice: Some("auto".to_string()),
                stream: false,
            };

            let response = self.llm.chat(request).await?;

            let choice = response.choices.first().cloned();
            let tc_list = choice.as_ref().and_then(|c| c.message.tool_calls.clone());
            let text_content = choice.as_ref().and_then(|c| c.message.content.clone());

            if let Some(tc_list) = tc_list.as_ref().filter(|l| !l.is_empty()) {
                let assistant_tool_calls: Vec<ToolCall> = tc_list
                    .iter()
                    .map(|tc| ToolCall {
                        id: tc.id.clone(),
                        call_type: tc.call_type.clone(),
                        function: ToolFunction {
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        },
                    })
                    .collect();

                self.context.add_message(Message::assistant(
                    text_content.as_deref().unwrap_or("").to_string(),
                    Some(assistant_tool_calls),
                ));

                for tc in tc_list {
                    outcome.had_tool_calls = true;
                    let _ = tx.send(AgentEvent::ToolCallAboutToRun {
                        name: tc.function.name.clone(),
                        args: tc.function.arguments.clone(),
                    }).await;

                    let tool_call = ToolCall {
                        id: tc.id.clone(),
                        call_type: tc.call_type.clone(),
                        function: ToolFunction {
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        },
                    };

                    let tool_result = self.execute_tool_call(&tool_call).await;
                    let _ = tx.send(AgentEvent::ToolCallComplete {
                        name: tool_call.function.name.clone(),
                        output: tool_result.output.clone(),
                        success: tool_result.success,
                    }).await;

                    if tool_result.output.contains("was denied by user") {
                        outcome.denied = Some((tc.function.name.clone(), tc.function.arguments.clone()));
                        return Ok(outcome);
                    }

                    self.context.add_message(Message::tool_result(&tc.id, &tool_result.output));
                }

                continue;
            }

            // No tool calls — add text to context for final answer synthesis, then break.
            if let Some(text) = text_content.as_ref().filter(|t| !t.trim().is_empty()) {
                self.context.add_message(Message::assistant(text.clone(), None));
            }
            break;
        }

        Ok(outcome)
    }

    async fn execute_tool_call(&self, tool_call: &ToolCall) -> ToolResult {
        let tool_name = &tool_call.function.name;
        let args = &tool_call.function.arguments;

        let parsed_args: serde_json::Value = match serde_json::from_str(args) {
            Ok(v) => v,
            Err(e) => {
                return ToolResult::err(
                    "",
                    format!("Failed to parse tool arguments: {}", e),
                );
            }
        };

        // Check dynamic safety level based on args — ask for approval if needed
        if let Some(tool) = self.registry.get(tool_name) {
            match tool.args_safety_level(&parsed_args) {
                SafetyLevel::Safe => {}
                SafetyLevel::Warning | SafetyLevel::Dangerous => {
                    if let Some(on_approve) = &self.on_approve {
                        let approved = on_approve(tool_name, args);
                        if !approved {
                            return ToolResult::err(
                                "",
                                format!("Tool call to '{}' was denied by user.", tool_name),
                            );
                        }
                    }
                }
            }
        }

        match self.registry.get(tool_name) {
            Some(tool) => tool.execute(parsed_args).await,
            None => ToolResult::err("", format!("Unknown tool: {}", tool_name)),
        }
    }

    /// Streaming final answer after all steps complete.
    async fn get_final_answer(&mut self, tx: &Sender<AgentEvent>) -> Result<()> {
        self.context.add_message(Message::user(
            "All steps are complete. Synthesize a final answer for the user based on what was done.",
        ));

        let messages = self.build_messages_for_llm();

        let request = ChatRequest {
            model: self.model_name.clone(),
            messages,
            tools: Some(self.registry.to_function_definitions()),
            tool_choice: Some("auto".to_string()),
            stream: false,
        };

        self.chat_streaming(request, tx).await?;
        Ok(())
    }

    async fn fallback_answer(&mut self, _user_prompt: &str, tx: &Sender<AgentEvent>) -> Result<()> {
        // Initial LLM call with tools enabled — use streaming so tokens appear incrementally
        let initial_messages = self.build_messages_for_llm();

        let request = ChatRequest {
            model: self.model_name.clone(),
            messages: initial_messages,
            tools: Some(self.registry.to_function_definitions()),
            tool_choice: Some("auto".to_string()),
            stream: false,
        };

        let response = match self.chat_streaming(request, tx).await {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(AgentEvent::Failed(format!("LLM error: {}", e))).await;
                return Ok(());
            }
        };

        let choice = response.choices.first().cloned();
        let tc_list = choice.as_ref().and_then(|c| c.message.tool_calls.clone());
        let text_content = choice.as_ref().and_then(|c| c.message.content.clone());

        if let Some(tc_list) = tc_list.as_ref().filter(|l| !l.is_empty()) {
            // LLM made tool calls — save to context, execute them, then loop for final answer
            let assistant_tool_calls: Vec<ToolCall> = tc_list
                .iter()
                .map(|tc| ToolCall {
                    id: tc.id.clone(),
                    call_type: tc.call_type.clone(),
                    function: ToolFunction {
                        name: tc.function.name.clone(),
                        arguments: tc.function.arguments.clone(),
                    },
                })
                .collect();

            self.context.add_message(Message::assistant(
                text_content.unwrap_or_default(),
                Some(assistant_tool_calls),
            ));

            for tc in tc_list {
                let _ = tx.send(AgentEvent::ToolCallAboutToRun {
                    name: tc.function.name.clone(),
                    args: tc.function.arguments.clone(),
                }).await;
                let tool_call = ToolCall {
                    id: tc.id.clone(),
                    call_type: tc.call_type.clone(),
                    function: ToolFunction {
                        name: tc.function.name.clone(),
                        arguments: tc.function.arguments.clone(),
                    },
                };
                let tool_result = self.execute_tool_call(&tool_call).await;
                let _ = tx.send(AgentEvent::ToolCallComplete {
                    name: tool_call.function.name.clone(),
                    output: tool_result.output.clone(),
                    success: tool_result.success,
                }).await;
                self.context.add_message(Message::tool_result(&tc.id, &tool_result.output));
            }

            // Loop to let the LLM make more tool calls or produce a final answer
            let mut last_had_tool_calls = false;

            for _ in 0..10 {
                let messages = self.build_messages_for_llm();

                let request = ChatRequest {
                    model: self.model_name.clone(),
                    messages,
                    tools: Some(self.registry.to_function_definitions()),
                    tool_choice: Some("auto".to_string()),
                    stream: false,
                };

                let response = match self.chat_streaming(request, tx).await {
                    Ok(r) => r,
                    Err(_) => break,
                };

                let choice = response.choices.first().cloned();
                let tc_list = choice.as_ref().and_then(|c| c.message.tool_calls.clone());
                let text_content = choice.as_ref().and_then(|c| c.message.content.clone());

                if let Some(tc_list) = tc_list.as_ref().filter(|l| !l.is_empty()) {
                    last_had_tool_calls = true;

                    let assistant_tool_calls: Vec<ToolCall> = tc_list
                        .iter()
                        .map(|tc| ToolCall {
                            id: tc.id.clone(),
                            call_type: tc.call_type.clone(),
                            function: ToolFunction {
                                name: tc.function.name.clone(),
                                arguments: tc.function.arguments.clone(),
                            },
                        })
                        .collect();

                    self.context.add_message(Message::assistant(
                        text_content.unwrap_or_default(),
                        Some(assistant_tool_calls),
                    ));

                    for tc in tc_list {
                        let _ = tx.send(AgentEvent::ToolCallAboutToRun {
                            name: tc.function.name.clone(),
                            args: tc.function.arguments.clone(),
                        }).await;
                        let tool_call = ToolCall {
                            id: tc.id.clone(),
                            call_type: tc.call_type.clone(),
                            function: ToolFunction {
                                name: tc.function.name.clone(),
                                arguments: tc.function.arguments.clone(),
                            },
                        };
                        let tool_result = self.execute_tool_call(&tool_call).await;
                        let _ = tx.send(AgentEvent::ToolCallComplete {
                            name: tool_call.function.name.clone(),
                            output: tool_result.output.clone(),
                            success: tool_result.success,
                        }).await;
                        self.context.add_message(Message::tool_result(&tc.id, &tool_result.output));
                    }
                    continue;
                }

                // No tool calls — if we have text content, break (text was already streamed)
                if text_content.as_ref().map(|t| !t.trim().is_empty()).unwrap_or(false) {
                    if last_had_tool_calls {
                        last_had_tool_calls = false;
                        continue;
                    }
                    break;
                }

                last_had_tool_calls = false;
            }
        } else if text_content.as_ref().filter(|t| !t.trim().is_empty()).is_some() {
            // No tool calls — LLM directly answered. Text was already streamed via chat_streaming.
        }

        Ok(())
    }

    /// Accept user input after a tool call was denied.
    /// Returns events continuing execution from the user's input.
    pub async fn on_user_input(&mut self, input: &str, tx: &Sender<AgentEvent>) -> Result<()> {
        self.context.add_message(Message::user(input));

        // Try to generate a new plan from the user's input
        let plan = match self.build_messages_for_plan() {
            Ok(msgs) => match self.send_to_llm_for_plan(&msgs).await {
                Ok(p) => p,
                Err(_) => {
                    // Plan failed, fall back to direct answer
                    return self.fallback_answer(input, tx).await;
                }
            },
            Err(e) => {
                let _ = tx.send(AgentEvent::Failed(format!("Failed to build context: {}", e))).await;
                return Ok(());
            }
        };
        let _ = tx.send(AgentEvent::PlanGenerated(plan.clone())).await;

        let mut current_plan = plan;
        for _round in 0..10 {
            if current_plan.all_complete() {
                break;
            }

            if current_plan.any_failed() {
                let _ = tx.send(AgentEvent::TextChunk("Replanning due to failed step...\n".to_string())).await;
                current_plan = match self.build_messages_for_plan() {
                    Ok(msgs) => match self.send_to_llm_for_plan(&msgs).await {
                        Ok(p) => p,
                        Err(e) => {
                            let _ = tx.send(AgentEvent::Failed(format!("Failed to replan: {}", e))).await;
                            return Ok(());
                        }
                    },
                    Err(e) => {
                        let _ = tx.send(AgentEvent::Failed(format!("Failed to build context: {}", e))).await;
                        return Ok(());
                    }
                };
                continue;
            }

            let step_idx = match current_plan.next_pending_step() {
                Some(idx) => idx,
                None => break,
            };

            current_plan.set_step_running(step_idx);
            let step_desc = current_plan.steps[step_idx].description.clone();
            let _ = tx.send(AgentEvent::TextChunk(format!("[Step {}] {}\n", step_idx + 1, step_desc))).await;

            let step = current_plan.steps[step_idx].clone();
            let outcome = self.execute_step(&step, tx).await;
            match outcome {
                Ok(outcome) => match outcome.denied {
                    Some((name, args)) => {
                        let _ = tx.send(AgentEvent::ApprovalDenied { name, args }).await;
                        return Ok(());
                    }
                    None => {
                        if !outcome.had_tool_calls {
                            current_plan.set_step_complete(step_idx, "Completed via text response");
                        }
                    }
                },
                Err(e) => {
                    let _ = tx.send(AgentEvent::TextChunk(format!("Step failed: {}\n", e))).await;
                    current_plan.set_step_failed(step_idx, e.to_string());
                }
            }
        }

        // Streaming final answer — synthesizes step work into one response.
        if let Err(e) = self.get_final_answer(tx).await {
            let _ = tx.send(AgentEvent::Failed(format!("Failed to get final answer: {}", e))).await;
        }
        let _ = tx.send(AgentEvent::Complete(String::new())).await;
        Ok(())
    }
}
