use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub steps: Vec<PlanStep>,
    pub status: PlanStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    #[serde(default)]
    pub id: usize,
    pub description: String,
    #[serde(default)]
    pub status: StepStatus,
    #[serde(default)]
    pub result: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum PlanStatus {
    Planning,
    Running,
    Complete,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum StepStatus {
    #[default]
    Pending,
    Running,
    Complete,
    Failed,
}

impl Plan {
    pub fn new(steps: Vec<PlanStep>) -> Self {
        Self {
            steps,
            status: PlanStatus::Planning,
        }
    }

    pub fn next_pending_step(&self) -> Option<usize> {
        self.steps
            .iter()
            .position(|s| s.status == StepStatus::Pending)
    }

    pub fn all_complete(&self) -> bool {
        self.steps.iter().all(|s| s.status == StepStatus::Complete)
    }

    pub fn any_failed(&self) -> bool {
        self.steps.iter().any(|s| s.status == StepStatus::Failed)
    }

    pub fn set_step_running(&mut self, step_id: usize) {
        if let Some(step) = self.steps.get_mut(step_id) {
            step.status = StepStatus::Running;
        }
    }

    pub fn set_step_complete(&mut self, step_id: usize, result: impl Into<String>) {
        if let Some(step) = self.steps.get_mut(step_id) {
            step.status = StepStatus::Complete;
            step.result = Some(result.into());
        }
    }

    pub fn set_step_failed(&mut self, step_id: usize, reason: impl Into<String>) {
        if let Some(step) = self.steps.get_mut(step_id) {
            step.status = StepStatus::Failed;
            step.result = Some(format!("Failed: {}", reason.into()));
        }
    }

    pub fn format_display(&self) -> String {
        let mut lines: Vec<String> = Vec::new();
        lines.push("Plan:".to_string());
        for step in &self.steps {
            let status_str = match step.status {
                StepStatus::Pending => "[ ]".to_string(),
                StepStatus::Running => "[~]".to_string(),
                StepStatus::Complete => "[x]".to_string(),
                StepStatus::Failed => "[!]".to_string(),
            };
            lines.push(format!("  {} {} {}", status_str, step.id, step.description));
            if let Some(ref result) = step.result {
                let first_line = result.lines().next().unwrap_or(result);
                lines.push(format!("      {}", truncate(first_line, 80)));
            }
        }
        lines.push(format!("Status: {:?}", self.status));
        lines.join("\n")
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_plan() {
        let steps = vec![
            PlanStep { id: 1, description: "Step 1".to_string(), status: StepStatus::Pending, result: None },
            PlanStep { id: 2, description: "Step 2".to_string(), status: StepStatus::Pending, result: None },
        ];
        let plan = Plan::new(steps);
        assert_eq!(plan.status, PlanStatus::Planning);
        assert_eq!(plan.next_pending_step(), Some(0));
    }

    #[test]
    fn test_step_lifecycle() {
        let steps = vec![
            PlanStep { id: 1, description: "Do thing".to_string(), status: StepStatus::Pending, result: None },
        ];
        let mut plan = Plan::new(steps);
        assert_eq!(plan.next_pending_step(), Some(0));

        plan.set_step_running(0);
        assert_eq!(plan.steps[0].status, StepStatus::Running);

        plan.set_step_complete(0, "Done");
        assert_eq!(plan.steps[0].status, StepStatus::Complete);
        assert!(plan.all_complete());
    }

    #[test]
    fn test_format_display() {
        let steps = vec![
            PlanStep { id: 1, description: "Read file".to_string(), status: StepStatus::Complete, result: Some("contents".to_string()) },
            PlanStep { id: 2, description: "Write file".to_string(), status: StepStatus::Pending, result: None },
        ];
        let plan = Plan::new(steps);
        let display = plan.format_display();
        assert!(display.contains("[x]"));
        assert!(display.contains("[ ]"));
    }
}
