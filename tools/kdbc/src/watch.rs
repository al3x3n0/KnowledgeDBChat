use std::thread;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::Args;
use serde_json::Value;
use uuid::Uuid;

use crate::{client::Platform, output_json};

#[derive(Args)]
pub struct WatchOptions {
    /// Seconds between status checks.
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u64).range(1..=60))]
    interval: u64,
    /// Maximum number of seconds to watch.
    #[arg(long, default_value_t = 900, value_parser = clap::value_parser!(u64).range(1..=86400))]
    timeout: u64,
}

pub enum WatchTarget {
    Workflow,
    Synthesis,
}

pub fn watch(
    platform: &Platform,
    target: WatchTarget,
    id: Uuid,
    options: WatchOptions,
) -> Result<Value> {
    let (path, label, command, active_statuses): (_, _, _, &[_]) = match target {
        WatchTarget::Workflow => (
            format!("workflows/executions/{id}"),
            "Workflow execution",
            "workflows watch",
            &["pending", "running"],
        ),
        WatchTarget::Synthesis => (
            format!("synthesis/{id}"),
            "Synthesis job",
            "synthesis watch",
            &["pending", "analyzing", "synthesizing", "generating"],
        ),
    };
    let interval = Duration::from_secs(options.interval);
    let deadline = Instant::now() + Duration::from_secs(options.timeout);
    let mut previous_status = String::new();
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            bail!("Watch timed out for {id}. Run kdbc {command} {id} to keep tracking it.");
        }
        let job = platform
            .get_with_timeout(&path, &[], remaining.min(Duration::from_secs(60)))
            .with_context(|| format!("Could not check {id}"))?;
        let status = job["status"].as_str().context("Response has no status")?;
        match status {
            "completed" => return Ok(job),
            "failed" | "cancelled" => {
                output_json(&job)?;
                bail!("{label} {id} {status}");
            }
            status if active_statuses.contains(&status) => {
                if previous_status != status {
                    eprintln!("{label} {id} is {status}");
                    previous_status = status.to_owned();
                }
            }
            _ => bail!("{label} {id} returned unknown status {status:?}"),
        }
        thread::sleep(interval.min(deadline.saturating_duration_since(Instant::now())));
    }
}
