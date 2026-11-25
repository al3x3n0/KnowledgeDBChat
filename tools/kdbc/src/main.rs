mod client;
mod profile_spec;
mod workflows;

use std::io::{self, IsTerminal, Read, Write};
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde_json::{json, Value};
use uuid::Uuid;

use client::Platform;
use profile_spec::ProfileCreate;
use workflows::Workflows;

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Platform URL, including any deployment path prefix.
    #[arg(
        long,
        env = "KDBC_URL",
        default_value = "http://localhost:23000",
        global = true
    )]
    url: String,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Obtain or check an account token.
    Auth {
        #[command(subcommand)]
        command: Auth,
    },
    /// Search library documents.
    Search {
        query: String,
        #[arg(long, value_enum, default_value = "smart")]
        mode: SearchMode,
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
        page: u32,
        #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u32).range(1..=100))]
        limit: u32,
        #[arg(long)]
        source: Option<Uuid>,
    },
    /// List document sources and their IDs.
    Sources,
    /// List and read research campaigns.
    Campaigns {
        #[command(subcommand)]
        command: Campaigns,
    },
    /// Inspect a retrieval trace and its Context Pack.
    Context {
        /// Retrieval trace ID.
        trace_id: Uuid,
        /// Print the retrieved Context Pack text instead of the trace JSON.
        #[arg(long)]
        text: bool,
    },
    /// Queue and track saved workflows.
    Workflows {
        #[command(subcommand)]
        command: Workflows,
    },
    /// List, read, and save research notes.
    Notes {
        #[command(subcommand)]
        command: Notes,
    },
    /// Show the review inbox, including incomplete sections.
    Review,
    /// Create and connect to scoped MCP profiles.
    Mcp {
        #[command(subcommand)]
        command: Profiles,
    },
}

#[derive(Subcommand)]
enum Auth {
    /// Read a password from stdin and print an account token.
    Login {
        #[arg(long, short)]
        username: String,
    },
    /// Show the account associated with KDBC_TOKEN.
    Whoami,
}

#[derive(Clone, ValueEnum)]
enum SearchMode {
    Smart,
    Keyword,
    Exact,
}

#[derive(Args)]
struct Pagination {
    #[arg(long, default_value_t = 50, value_parser = clap::value_parser!(u32).range(1..=200))]
    limit: u32,
    #[arg(long, default_value_t = 0)]
    offset: u32,
}

impl Pagination {
    fn query(&self) -> Vec<(&'static str, String)> {
        vec![
            ("limit", self.limit.to_string()),
            ("offset", self.offset.to_string()),
        ]
    }
}

#[derive(Subcommand)]
enum Campaigns {
    /// List research campaigns.
    List {
        #[command(flatten)]
        pagination: Pagination,
        #[arg(long)]
        status: Option<String>,
    },
    /// Read a research campaign by ID.
    Get { id: Uuid },
}

#[derive(Subcommand)]
enum Notes {
    /// List research notes.
    List {
        #[command(flatten)]
        pagination: Pagination,
        #[arg(long)]
        query: Option<String>,
    },
    /// Read a research note by ID.
    Get {
        id: Uuid,
        /// Print the note's Markdown instead of JSON.
        #[arg(long)]
        markdown: bool,
    },
    /// Save an owned Context Pack as a research note.
    SaveContext { trace_id: Uuid },
}

#[derive(Subcommand)]
enum Profiles {
    /// Discover programs and tools available to profiles using KDBC_TOKEN.
    Options,
    /// List MCP profiles using KDBC_TOKEN.
    List,
    /// Show a profile's configured tools, sources, notes, and prompts.
    Inspect { id: Uuid },
    /// Create a scoped profile and print its one-time key as JSON.
    Create(ProfileCreate),
    /// Connect using KDBC_MCP_KEY and list available capabilities.
    Check,
    /// Read a published note through MCP using KDBC_MCP_KEY.
    ReadNote {
        id: Uuid,
        /// Print the resource text instead of the MCP JSON result.
        #[arg(long)]
        markdown: bool,
    },
}

fn run(cli: Cli) -> Result<()> {
    let platform = Platform::new(&cli.url)?;
    let result = match cli.command {
        Command::Auth {
            command: Auth::Login { username },
        } => {
            if io::stdin().is_terminal() {
                bail!("Pipe the password to stdin. Read it without echo in your shell.");
            }
            let mut password = String::new();
            io::stdin()
                .read_to_string(&mut password)
                .context("Could not read password from stdin")?;
            let password = password
                .strip_suffix("\r\n")
                .or_else(|| password.strip_suffix('\n'))
                .unwrap_or(&password);
            if password.is_empty() {
                bail!("Supply the password on stdin");
            }
            return output_text(&platform.login(&username, password)?);
        }
        Command::Auth {
            command: Auth::Whoami,
        } => platform.get("auth/me", &[])?,
        Command::Search {
            query,
            mode,
            page,
            limit,
            source,
        } => {
            let mode = mode.to_possible_value().context("Invalid search mode")?;
            let mut params = vec![
                ("q", query),
                ("mode", mode.get_name().to_owned()),
                ("page", page.to_string()),
                ("page_size", limit.to_string()),
            ];
            if let Some(id) = source {
                params.push(("source_id", id.to_string()));
            }
            platform.get("documents/search", &params)?
        }
        Command::Sources => platform.get("documents/sources/", &[])?,
        Command::Campaigns {
            command: Campaigns::List { pagination, status },
        } => {
            let mut params = pagination.query();
            if let Some(status) = status {
                params.push(("status", status));
            }
            platform.get("research/campaigns", &params)?
        }
        Command::Campaigns {
            command: Campaigns::Get { id },
        } => platform.get(&format!("research/campaigns/{id}"), &[])?,
        Command::Context { trace_id, text } => {
            let trace = platform.get(&format!("retrieval-traces/{trace_id}"), &[])?;
            if text {
                return output_text(
                    trace["trace"]["kg_context_pack"]["kg_context"]
                        .as_str()
                        .filter(|context| !context.trim().is_empty())
                        .context("This retrieval trace has no Context Pack text")?,
                );
            }
            trace
        }
        Command::Workflows { command } => command.execute(&platform)?,
        Command::Notes {
            command: Notes::List { pagination, query },
        } => {
            let mut params = pagination.query();
            if let Some(query) = query {
                params.push(("q", query));
            }
            platform.get("research-notes", &params)?
        }
        Command::Notes {
            command: Notes::Get { id, markdown },
        } => {
            let note = platform.get(&format!("research-notes/{id}"), &[])?;
            if markdown {
                return output_text(
                    note["content_markdown"]
                        .as_str()
                        .context("Note has no Markdown")?,
                );
            }
            note
        }
        Command::Notes {
            command: Notes::SaveContext { trace_id },
        } => platform.post(&format!("research-notes/context-packs/{trace_id}"), None)?,
        Command::Review => platform.get("review", &[])?,
        Command::Mcp { command } => match command {
            Profiles::Options => platform.get("mcp-config/profile-options", &[])?,
            Profiles::List => {
                let result = platform.get("api-keys/", &[])?;
                let profiles = result["api_keys"]
                    .as_array()
                    .context("Expected an API key list")?
                    .iter()
                    .filter(|key| key["is_mcp_profile"] == true)
                    .collect::<Vec<_>>();
                json!({ "profiles": profiles, "total": profiles.len() })
            }
            Profiles::Inspect { id } => {
                let config = platform.get(&format!("mcp-config/keys/{id}/config"), &[])?;
                if !config["mcp_programs"].is_array() {
                    bail!("Key {id} is not an MCP profile");
                }
                config
            }
            Profiles::Create(selection) => {
                let dry_run = selection.dry_run;
                let payload = selection.payload()?;
                if dry_run {
                    payload
                } else {
                    platform.post("mcp-config/profiles", Some(&payload))?
                }
            }
            Profiles::Check => platform.mcp()?.check()?,
            Profiles::ReadNote { id, markdown } => {
                let result = platform.mcp()?.call(
                    "resources/read",
                    json!({
                        "uri": format!("kdbc://research-notes/{id}")
                    }),
                )?;
                if markdown {
                    return output_text(
                        result["contents"][0]["text"]
                            .as_str()
                            .context("Resource has no Markdown")?,
                    );
                }
                result
            }
        },
    };
    output_json(&result)
}

fn output_json(result: &Value) -> Result<()> {
    let mut stdout = io::stdout().lock();
    serde_json::to_writer_pretty(&mut stdout, result)?;
    writeln!(stdout)?;
    Ok(())
}

fn output_text(text: &str) -> Result<()> {
    let mut stdout = io::stdout().lock();
    write!(stdout, "{text}")?;
    if !text.ends_with('\n') {
        writeln!(stdout)?;
    }
    Ok(())
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("kdbc: {error:#}");
            ExitCode::FAILURE
        }
    }
}
