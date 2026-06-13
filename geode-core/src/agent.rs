use crate::context::ContextManager;
use crate::llm_client::{ChatRequest, LlmClient};
use crate::message::{Message, ToolCall, ToolFunction};
use crate::planning::{Plan, PlanStep};
use crate::tool::{SafetyLevel, ToolResult};
use crate::tool_registry::ToolRegistry;
use anyhow::Result;

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

    pub async fn run(&mut self, user_prompt: &str) -> AgentResult {
        let mut events = Vec::new();

        self.context.add_message(Message::user(user_prompt));

        // Generate initial plan
        let plan = match self.build_messages_for_plan() {
            Ok(msgs) => match self.send_to_llm_for_plan(&msgs).await {
                Ok(p) => p,
                Err(_) => {
                    // Plan generation failed — fall back to direct LLM call with tools
                    return self.fallback_answer(user_prompt).await;
                }
            },
            Err(e) => {
                events.push(AgentEvent::Failed(format!("Failed to build context: {}", e)));
                return events;
            }
        };
        events.push(AgentEvent::PlanGenerated(plan.clone()));

        let mut current_plan = plan;
        for _round in 0..10 {
            if current_plan.all_complete() {
                break;
            }

            if current_plan.any_failed() {
                events.push(AgentEvent::TextChunk("Replanning due to failed step...\n".to_string()));
                current_plan = match self.build_messages_for_plan() {
                    Ok(msgs) => match self.send_to_llm_for_plan(&msgs).await {
                        Ok(p) => p,
                        Err(e) => {
                            events.push(AgentEvent::Failed(format!("Failed to replan: {}", e)));
                            return events;
                        }
                    },
                    Err(e) => {
                        events.push(AgentEvent::Failed(format!("Failed to build context: {}", e)));
                        return events;
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
            events.push(AgentEvent::TextChunk(format!("[Step {}] {}\n", step_idx + 1, step_desc)));

            let step = current_plan.steps[step_idx].clone();
            let result = self.execute_step(&step, &mut events).await;
            match result {
                Ok(Some((name, args))) => {
                    // Tool call was denied by user
                    events.push(AgentEvent::ApprovalDenied { name, args });
                    return events;
                }
                Ok(None) => {
                    // Normal completion (with or without tool calls)
                    // Check if the last ToolCallComplete was a denial
                    if let Some(AgentEvent::ToolCallComplete { output, .. }) = events.last() {
                        if output.contains("was denied by user") {
                            // Shouldn't happen since we return early, but handle it
                            continue;
                        }
                    }
                    // If no tool calls were made, mark complete
                    let had_tool_calls = events.iter().any(|e| matches!(e, AgentEvent::ToolCallAboutToRun { .. }));
                    if !had_tool_calls {
                        current_plan.set_step_complete(step_idx, "Completed via text response");
                    }
                }
                Err(e) => {
                    current_plan.set_step_failed(step_idx, e.to_string());
                    events.push(AgentEvent::TextChunk(format!("Step failed: {}\n", e)));
                }
            }
        }

        // Final answer
        self.context.add_message(Message::user(
            "All steps complete. Please provide a final answer summarizing what was done.",
        ));
        match self.get_final_answer().await {
            Ok(answer) => events.push(AgentEvent::Complete(answer)),
            Err(e) => events.push(AgentEvent::Failed(format!("Failed to get final answer: {}", e))),
        }

        events
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

    async fn send_to_llm_for_plan(&self, messages: &[Message]) -> Result<Plan> {
        let request = ChatRequest {
            model: self.model_name.clone(),
            messages: messages.to_vec(),
            tools: None,
            tool_choice: None,
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

    async fn execute_step(
        &mut self,
        step: &PlanStep,
        events: &mut Vec<AgentEvent>,
    ) -> Result<Option<(String, String)>> {
        let prompt = format!(
            "Execute this step: '{}'. Use the appropriate tool.",
            step.description
        );

        let mut messages = vec![Message::system(self.system_prompt.clone())];
        messages.push(Message::user(prompt));
        let request = ChatRequest {
            model: self.model_name.clone(),
            messages,
            tools: Some(self.registry.to_function_definitions()),
            tool_choice: Some("auto".to_string()),
        };

        let response = self.llm.chat(request).await?;
        let mut has_tool_calls = false;

        if let Some(choice) = response.choices.first() {
            if let Some(tc_list) = &choice.message.tool_calls {
                for tc in tc_list {
                    has_tool_calls = true;
                    let tool_call = ToolCall {
                        id: tc.id.clone(),
                        call_type: tc.call_type.clone(),
                        function: ToolFunction {
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        },
                    };

                    events.push(AgentEvent::ToolCallAboutToRun {
                        name: tc.function.name.clone(),
                        args: tc.function.arguments.clone(),
                    });

                    let tool_result = self.execute_tool_call(&tool_call).await;
                    events.push(AgentEvent::ToolCallComplete {
                        name: tool_call.function.name.clone(),
                        output: tool_result.output.clone(),
                        success: tool_result.success,
                    });

                    // If tool was denied (not just failed), signal early exit with tool info
                    if tool_result.output.contains("was denied by user") {
                        return Ok(Some((tc.function.name.clone(), tc.function.arguments.clone())));
                    }

                    // Add to context for next iteration
                    self.context.add_message(Message::tool_result(&tc.id, &tool_result.output));
                }
            }
        }

        if has_tool_calls {
            Ok(None)
        } else {
            Ok(None)
        }
    }

    async fn execute_tool_call(&self, tool_call: &ToolCall) -> ToolResult {
        let tool_name = &tool_call.function.name;
        let args = &tool_call.function.arguments;

        // Check safety level — ask for approval if needed
        if let Some(tool) = self.registry.get(tool_name) {
            match tool.safety_level() {
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

        let parsed_args: serde_json::Value = match serde_json::from_str(args) {
            Ok(v) => v,
            Err(e) => {
                return ToolResult::err(
                    "",
                    format!("Failed to parse tool arguments: {}", e),
                );
            }
        };

        match self.registry.get(tool_name) {
            Some(tool) => tool.execute(parsed_args).await,
            None => ToolResult::err("", format!("Unknown tool: {}", tool_name)),
        }
    }

    async fn get_final_answer(&mut self) -> Result<String> {
        let mut messages = self
            .context
            .get_completion_messages()
            .into_iter()
            .filter_map(|m| m.to_api_message())
            .collect::<Vec<_>>();
        messages.insert(0, Message::system(self.system_prompt.clone()));

        let request = ChatRequest {
            model: self.model_name.clone(),
            messages,
            tools: None,
            tool_choice: None,
        };

        let response = self.llm.chat(request).await?;

        if let Some(choice) = response.choices.first() {
            Ok(choice.message.content.clone().unwrap_or_default())
        } else {
            anyhow::bail!("No response from LLM")
        }
    }

    async fn fallback_answer(&mut self, _user_prompt: &str) -> AgentResult {
        let mut events = Vec::new();
        let mut messages = self
            .context
            .get_completion_messages()
            .into_iter()
            .filter_map(|m| m.to_api_message())
            .collect::<Vec<_>>();
        messages.insert(0, Message::system(self.system_prompt.clone()));

        let request = ChatRequest {
            model: self.model_name.clone(),
            messages,
            tools: Some(self.registry.to_function_definitions()),
            tool_choice: Some("auto".to_string()),
        };

        let response = match self.llm.chat(request).await {
            Ok(r) => r,
            Err(e) => {
                events.push(AgentEvent::Failed(format!("LLM error: {}", e)));
                return events;
            }
        };

        if let Some(choice) = response.choices.first() {
            if let Some(tc_list) = &choice.message.tool_calls {
                for tc in tc_list {
                    events.push(AgentEvent::ToolCallAboutToRun {
                        name: tc.function.name.clone(),
                        args: tc.function.arguments.clone(),
                    });
                    let tool_call = ToolCall {
                        id: tc.id.clone(),
                        call_type: tc.call_type.clone(),
                        function: ToolFunction {
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        },
                    };
                    let tool_result = self.execute_tool_call(&tool_call).await;
                    events.push(AgentEvent::ToolCallComplete {
                        name: tool_call.function.name.clone(),
                        output: tool_result.output.clone(),
                        success: tool_result.success,
                    });
                    self.context.add_message(Message::tool_result(&tc.id, &tool_result.output));
                }
            }
            if let Some(text) = &choice.message.content {
                if !text.is_empty() {
                    events.push(AgentEvent::TextChunk(text.clone()));
                }
            }
        }

        // Get final answer after executing any tool calls
        let final_messages = self
            .context
            .get_completion_messages()
            .into_iter()
            .filter_map(|m| m.to_api_message())
            .collect::<Vec<_>>();
        let final_messages_len = final_messages.len();
        let final_answer = match self.llm.chat(ChatRequest {
            model: self.model_name.clone(),
            messages: final_messages,
            tools: None,
            tool_choice: None,
        }).await {
            Ok(resp) => resp.choices.first().and_then(|c| c.message.content.clone()).unwrap_or_default(),
            Err(_) => String::new(),
        };

        if !final_answer.is_empty() {
            events.push(AgentEvent::Complete(final_answer));
        } else if final_messages_len == 0 {
            events.push(AgentEvent::Failed("No response from LLM".to_string()));
        }

        events
    }

    /// Accept user input after a tool call was denied.
    /// Returns events continuing execution from the user's input.
    pub async fn on_user_input(&mut self, input: &str) -> AgentResult {
        let mut events = Vec::new();
        self.context.add_message(Message::user(input));

        // Try to generate a new plan from the user's input
        let plan = match self.build_messages_for_plan() {
            Ok(msgs) => match self.send_to_llm_for_plan(&msgs).await {
                Ok(p) => p,
                Err(_) => {
                    // Plan failed, fall back to direct answer
                    return match self.get_final_answer().await {
                        Ok(answer) => vec![AgentEvent::Complete(answer)],
                        Err(e) => vec![AgentEvent::Failed(format!("Failed to get answer: {}", e))],
                    };
                }
            },
            Err(e) => {
                return vec![AgentEvent::Failed(format!("Failed to build context: {}", e))];
            }
        };
        events.push(AgentEvent::PlanGenerated(plan.clone()));

        let mut current_plan = plan;
        for _round in 0..10 {
            if current_plan.all_complete() {
                break;
            }

            if current_plan.any_failed() {
                events.push(AgentEvent::TextChunk("Replanning due to failed step...\n".to_string()));
                current_plan = match self.build_messages_for_plan() {
                    Ok(msgs) => match self.send_to_llm_for_plan(&msgs).await {
                        Ok(p) => p,
                        Err(e) => {
                            events.push(AgentEvent::Failed(format!("Failed to replan: {}", e)));
                            return events;
                        }
                    },
                    Err(e) => {
                        events.push(AgentEvent::Failed(format!("Failed to build context: {}", e)));
                        return events;
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
            events.push(AgentEvent::TextChunk(format!("[Step {}] {}\n", step_idx + 1, step_desc)));

            let step = current_plan.steps[step_idx].clone();
            let result = self.execute_step(&step, &mut events).await;
            match result {
                Ok(Some((name, args))) => {
                    events.push(AgentEvent::ApprovalDenied { name, args });
                    return events;
                }
                Ok(None) => {
                    let had_tool_calls = events.iter().any(|e| matches!(e, AgentEvent::ToolCallAboutToRun { .. }));
                    if !had_tool_calls {
                        current_plan.set_step_complete(step_idx, "Completed via text response");
                    }
                }
                Err(e) => {
                    current_plan.set_step_failed(step_idx, e.to_string());
                    events.push(AgentEvent::TextChunk(format!("Step failed: {}\n", e)));
                }
            }
        }

        // Final answer
        self.context.add_message(Message::user(
            "All steps complete. Please provide a final answer summarizing what was done.",
        ));
        match self.get_final_answer().await {
            Ok(answer) => events.push(AgentEvent::Complete(answer)),
            Err(e) => events.push(AgentEvent::Failed(format!("Failed to get final answer: {}", e))),
        }

        events
    }
}
