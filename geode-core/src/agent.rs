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

        // Loop: if the LLM returns text without tool calls, ask it to continue
        // with actual tool executions rather than just describing what it would do.
        for _loop_iter in 0..10 {
            let request = ChatRequest {
                model: self.model_name.clone(),
                messages: messages.clone(),
                tools: Some(self.registry.to_function_definitions()),
                tool_choice: Some("auto".to_string()),
            };

            let response = self.llm.chat(request).await?;

            // Snapshot the choice so we can use it after the loop iteration.
            // The response.choices is consumed in the match below, so we
            // extract what we need up front.
            let choice = response.choices.first().cloned();
            let tc_list = choice.as_ref().and_then(|c| c.message.tool_calls.clone());
            let text_content = choice.as_ref().and_then(|c| c.message.content.clone());

            if let Some(tc_list) = tc_list.as_ref().filter(|l| !l.is_empty()) {
                // Build assistant message with tool calls
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

                // Add assistant message with tool calls to messages and context
                let assistant_content = text_content.as_deref().unwrap_or("").to_string();
                messages.push(Message::assistant(
                    assistant_content.clone(),
                    Some(assistant_tool_calls.clone()),
                ));
                self.context.add_message(Message::assistant(
                    assistant_content,
                    Some(assistant_tool_calls),
                ));

                // Execute each tool call and add results
                for tc in tc_list {
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

                    if tool_result.output.contains("was denied by user") {
                        return Ok(Some((tc.function.name.clone(), tc.function.arguments.clone())));
                    }

                    // Add tool result to both messages and context
                    let result_msg = Message::tool_result(&tc.id, &tool_result.output);
                    messages.push(result_msg.clone());
                    self.context.add_message(result_msg);
                }

                continue; // keep looping — LLM may have more tool calls
            }

            // No tool calls — LLM returned text only
            if let Some(text) = text_content.as_ref().filter(|t| !t.trim().is_empty()) {
                messages.push(Message::assistant(text.clone(), None));
                self.context.add_message(Message::assistant(text.clone(), None));
                events.push(AgentEvent::TextChunk(text.clone()));
                continue; // loop back — ask the LLM to continue
            }

            break;
        }

        Ok(None)
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

        let mut accumulated_text = String::new();
        let mut last_had_tool_calls = false;

        // Loop: check if the LLM suggests more tool calls before giving a final answer.
        // If it returns text without tool calls and doesn't suggest more actions,
        // return the text. Otherwise, execute any tool calls and continue.
        for _ in 0..10 {
            let request = ChatRequest {
                model: self.model_name.clone(),
                messages: messages.clone(),
                tools: Some(self.registry.to_function_definitions()),
                tool_choice: Some("auto".to_string()),
            };

            let response = self.llm.chat(request).await?;

            let choice = response.choices.first().cloned();
            let tc_list = choice.as_ref().and_then(|c| c.message.tool_calls.clone());
            let text_content = choice.as_ref().and_then(|c| c.message.content.clone());

            if let Some(text) = text_content.as_ref().filter(|t| !t.trim().is_empty()) {
                accumulated_text = text.clone();
            }

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

                let assistant_content = text_content.unwrap_or_default();
                messages.push(Message::assistant(
                    assistant_content.clone(),
                    Some(assistant_tool_calls.clone()),
                ));
                self.context.add_message(Message::assistant(
                    assistant_content,
                    Some(assistant_tool_calls),
                ));

                for tc in tc_list {
                    let tool_call = ToolCall {
                        id: tc.id.clone(),
                        call_type: tc.call_type.clone(),
                        function: ToolFunction {
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        },
                    };
                    let tool_result = self.execute_tool_call(&tool_call).await;
                    let result_msg = Message::tool_result(&tc.id, &tool_result.output);
                    messages.push(result_msg.clone());
                    self.context.add_message(result_msg);
                }
                continue;
            }

            // No tool calls — text only
            if !accumulated_text.is_empty() {
                messages.push(Message::assistant(accumulated_text.clone(), None));
                if last_had_tool_calls {
                    last_had_tool_calls = false;
                    continue;
                }
                return Ok(accumulated_text);
            }

            last_had_tool_calls = false;
        }

        Ok(accumulated_text)
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
            messages: messages.clone(),
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

        // Process initial response — add assistant + tool results to messages
        let choice = response.choices.first().cloned();
        let tc_list = choice.as_ref().and_then(|c| c.message.tool_calls.clone());
        let text_content = choice.as_ref().and_then(|c| c.message.content.clone());

        if let Some(text) = text_content.as_ref().filter(|t| !t.trim().is_empty()) {
            events.push(AgentEvent::TextChunk(text.clone()));
        }

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

            messages.push(Message::assistant(
                text_content.unwrap_or_default(),
                Some(assistant_tool_calls.clone()),
            ));
            self.context.add_message(Message::assistant(
                String::new(),
                Some(assistant_tool_calls),
            ));

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
                let result_msg = Message::tool_result(&tc.id, &tool_result.output);
                messages.push(result_msg.clone());
                self.context.add_message(result_msg);
            }
        }

        // Final answer loop with tools enabled
        let mut final_messages = messages.clone();
        final_messages.insert(0, Message::system(self.system_prompt.clone()));

        let mut accumulated_text = String::new();
        let mut last_had_tool_calls = false;

        for _ in 0..10 {
            let request = ChatRequest {
                model: self.model_name.clone(),
                messages: final_messages.clone(),
                tools: Some(self.registry.to_function_definitions()),
                tool_choice: Some("auto".to_string()),
            };

            let response = match self.llm.chat(request).await {
                Ok(r) => r,
                Err(_) => break,
            };

            let choice = response.choices.first().cloned();
            let tc_list = choice.as_ref().and_then(|c| c.message.tool_calls.clone());
            let text_content = choice.as_ref().and_then(|c| c.message.content.clone());

            if let Some(text) = text_content.as_ref().filter(|t| !t.trim().is_empty()) {
                accumulated_text = text.clone();
            }

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

                final_messages.push(Message::assistant(
                    text_content.unwrap_or_default(),
                    Some(assistant_tool_calls.clone()),
                ));
                self.context.add_message(Message::assistant(
                    String::new(),
                    Some(assistant_tool_calls),
                ));

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
                    let result_msg = Message::tool_result(&tc.id, &tool_result.output);
                    final_messages.push(result_msg.clone());
                    self.context.add_message(result_msg);
                }
                continue;
            }

            // No tool calls — text only
            if !accumulated_text.is_empty() {
                final_messages.push(Message::assistant(accumulated_text.clone(), None));
                if last_had_tool_calls {
                    last_had_tool_calls = false;
                    continue;
                }
                break;
            }

            last_had_tool_calls = false;
        }

        if !accumulated_text.is_empty() {
            events.push(AgentEvent::Complete(accumulated_text));
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
