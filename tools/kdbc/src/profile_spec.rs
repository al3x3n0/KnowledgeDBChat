use std::fs::File;
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Args;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

#[derive(Args)]
pub struct ProfileCreate {
    /// Read a JSON profile definition from a file or from stdin with -.
    #[arg(long, value_name = "FILE", conflicts_with_all = ["name", "program", "tool", "note", "source", "all_sources"])]
    file: Option<PathBuf>,
    #[arg(long, required_unless_present = "file")]
    name: Option<String>,
    #[arg(long)]
    program: Vec<String>,
    #[arg(long)]
    tool: Vec<String>,
    #[arg(long)]
    note: Vec<Uuid>,
    #[arg(long, conflicts_with = "all_sources")]
    source: Vec<Uuid>,
    /// Give the selected document tools access to all sources.
    #[arg(long, requires = "tool")]
    all_sources: bool,
    /// Print the request payload without creating a profile.
    #[arg(long)]
    pub dry_run: bool,
}

impl ProfileCreate {
    pub fn payload(self) -> Result<Value> {
        let definition = match self.file {
            Some(path) => ProfileDefinition::read(&path)?,
            None => ProfileDefinition {
                name: self.name.context("Supply --name or --file")?,
                programs: self.program,
                tools: self.tool,
                note_ids: self.note,
                source_ids: (!self.source.is_empty()).then_some(self.source),
                all_sources: self.all_sources,
                prompts: Vec::new(),
            },
        };
        definition.payload()
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProfileDefinition {
    name: String,
    #[serde(default)]
    programs: Vec<String>,
    #[serde(default)]
    tools: Vec<String>,
    #[serde(default)]
    note_ids: Vec<Uuid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_ids: Option<Vec<Uuid>>,
    #[serde(default, skip_serializing)]
    all_sources: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    prompts: Vec<PromptDefinition>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PromptDefinition {
    name: String,
    title: String,
    instructions: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    include_resources: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    use_inputs: Option<bool>,
}

impl ProfileDefinition {
    fn read(path: &Path) -> Result<Self> {
        if path == Path::new("-") {
            return serde_json::from_reader(io::stdin().lock())
                .context("Could not parse a JSON profile definition from stdin");
        }
        let file =
            File::open(path).with_context(|| format!("Could not open {}", path.display()))?;
        serde_json::from_reader(BufReader::new(file)).with_context(|| {
            format!(
                "Could not parse the profile definition in {}",
                path.display()
            )
        })
    }

    fn payload(mut self) -> Result<Value> {
        self.name = self.name.trim().to_owned();
        if self.name.is_empty() || self.name.chars().count() > 100 {
            bail!("The profile name must be between 1 and 100 characters");
        }
        if self.programs.is_empty()
            && self.tools.is_empty()
            && self.note_ids.is_empty()
            && self.prompts.is_empty()
        {
            bail!("Select at least one program, tool, research note, or prompt");
        }
        if self.all_sources && self.source_ids.is_some() {
            bail!("Choose either explicit source IDs or all sources");
        }
        if !self.tools.is_empty() && self.source_ids.is_none() && !self.all_sources {
            bail!("Select document sources (--source or source_ids), or explicitly pass --all-sources or set all_sources to true");
        }
        Ok(serde_json::to_value(self)?)
    }
}
