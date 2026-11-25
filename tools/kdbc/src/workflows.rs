use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::Subcommand;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::{client::Platform, output_json};

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
        /// Seconds between status checks.
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u64).range(1..=60))]
        interval: u64,
        /// Maximum number of seconds to watch.
        #[arg(long, default_value_t = 900, value_parser = clap::value_parser!(u64).range(1..=86400))]
        timeout: u64,
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
                interval,
                timeout,
            } => watch_execution(
                platform,
                execution_id,
                Duration::from_secs(interval),
                Duration::from_secs(timeout),
            ),
        }
    }
}

fn watch_execution(
    platform: &Platform,
    execution_id: Uuid,
    interval: Duration,
    timeout: Duration,
) -> Result<Value> {
    let path = format!("workflows/executions/{execution_id}");
    let deadline = Instant::now() + timeout;
    let mut previous_status = String::new();
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            bail!("Stopped waiting for execution {execution_id}. Run workflows watch again to keep tracking it.");
        }
        let execution = platform
            .get_with_timeout(&path, &[], remaining.min(Duration::from_secs(60)))
            .with_context(|| format!("Could not check execution {execution_id}"))?;
        let status = execution["status"]
            .as_str()
            .context("Execution response has no status")?;
        match status {
            "completed" => return Ok(execution),
            "failed" | "cancelled" => {
                output_json(&execution)?;
                bail!("Workflow execution {execution_id} {status}");
            }
            "pending" | "running" => {
                if previous_status != status {
                    eprintln!("Execution {execution_id} is {status}");
                    previous_status = status.to_owned();
                }
            }
            _ => bail!("Workflow execution {execution_id} returned unknown status {status:?}"),
        }
        thread::sleep(interval.min(deadline.saturating_duration_since(Instant::now())));
    }
}
