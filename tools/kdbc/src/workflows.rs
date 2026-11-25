use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Subcommand;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::client::Platform;
use crate::watch::{watch, WatchOptions, WatchTarget};

#[derive(Subcommand)]
pub enum Workflows {
    /// List saved workflows and their IDs.
    List {
        #[arg(long, default_value_t = 50, value_parser = clap::value_parser!(u32).range(1..=100))]
        limit: u32,
        #[arg(long, default_value_t = 0)]
        offset: u32,
    },
    /// Queue the saved workflow and return its execution ID.
    Run {
        workflow_id: Uuid,
        /// Read initial workflow inputs from a JSON object in this file.
        #[arg(long, value_name = "FILE")]
        inputs: Option<PathBuf>,
    },
    /// Read one execution by ID.
    Get { execution_id: Uuid },
    /// Wait for an execution to finish and print its final state.
    Watch {
        execution_id: Uuid,
        #[command(flatten)]
        options: WatchOptions,
    },
}

impl Workflows {
    pub fn execute(self, platform: &Platform) -> Result<Value> {
        match self {
            Self::List { limit, offset } => platform.get(
                "workflows",
                &[("limit", limit.to_string()), ("offset", offset.to_string())],
            ),
            Self::Run {
                workflow_id,
                inputs,
            } => {
                let inputs = match inputs {
                    Some(path) => {
                        let file = File::open(&path)
                            .with_context(|| format!("Could not open {}", path.display()))?;
                        let value: Value = serde_json::from_reader(BufReader::new(file))
                            .context("Workflow inputs must be valid JSON")?;
                        if !value.is_object() {
                            bail!("Workflow inputs must be a JSON object");
                        }
                        value
                    }
                    None => json!({}),
                };
                let result = platform.post(
                    &format!("workflows/{workflow_id}/execute/async"),
                    Some(&json!({"trigger_type": "manual", "trigger_data": {}, "inputs": inputs})),
                )?;
                result["execution_id"]
                    .as_str()
                    .and_then(|value| Uuid::parse_str(value).ok())
                    .context("Queue response did not contain a valid execution ID")?;
                Ok(result)
            }
            Self::Get { execution_id } => {
                platform.get(&format!("workflows/executions/{execution_id}"), &[])
            }
            Self::Watch {
                execution_id,
                options,
            } => watch(platform, WatchTarget::Workflow, execution_id, options),
        }
    }
}
