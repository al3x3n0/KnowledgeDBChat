# KnowledgeOps Lab CLI

Install `kdbc` from the repository root with Rust and Cargo.

```sh
cargo install --path tools/kdbc --locked
```

Set `KDBC_URL` to your deployment URL. It defaults to the local gateway at
`http://localhost:23000`. You can also pass `--url` for a single invocation.

## Sign in and find context

Read your password without echo and pipe it to `auth login`. The command
returns an account token on stdout. Keep it in the current shell session.

```sh
printf 'Password: ' >&2
read -rs password
printf '\n' >&2
token="$(printf '%s\n' "$password" | kdbc auth login --username alice)"
unset password
export KDBC_TOKEN="$token"
unset token

kdbc auth whoami
kdbc search "cache behavior" --mode keyword
kdbc sources
kdbc campaigns list
kdbc review
kdbc context "$TRACE_ID" --text
kdbc notes list --query "cache"
kdbc notes get "$NOTE_ID" --markdown
```

## Queue and track workflows

List saved workflows, then queue one by its ID. Supply `--inputs inputs.json`
when the workflow needs initial inputs. The file must contain a JSON object.
Use the returned `execution_id` to inspect or watch that run.

```sh
kdbc workflows list
kdbc workflows run "$WORKFLOW_ID" --inputs inputs.json
kdbc workflows get "$EXECUTION_ID"
kdbc workflows watch "$EXECUTION_ID" --timeout 600
```

The watch command prints status changes to stderr and the final execution as
JSON. Failed or cancelled executions also return a nonzero exit code. A watch
timeout leaves the execution running so you can resume watching it by ID.

## Publish and verify an MCP profile

Find program and tool names with `kdbc mcp options`. If you have a Context
Pack, `notes save-context` accepts its retrieval trace ID and returns the saved
note with its citations. Use that note ID when creating a profile.

```sh
kdbc notes save-context "$TRACE_ID"
kdbc mcp options
kdbc mcp create --name "Cache research" --note "$NOTE_ID"
kdbc mcp list
```

The create response contains the profile ID and the one-time `api_key`.
Capture the response in a secure location. Set `KDBC_MCP_KEY` to that key to
test the client-facing profile. The account token is used for managing profiles.

```sh
kdbc mcp check
kdbc mcp read-note "$NOTE_ID" --markdown
```

Repeat `--note`, `--program`, `--tool`, or `--source` to select multiple
items. When selecting document tools, choose document sources with `--source`,
or use `--all-sources` to grant all-source access deliberately.

```sh
kdbc mcp create --name "Compiler library" --program "$PROGRAM_NAME" \
  --tool search --source "$SOURCE_ID"
```

### Reuse a profile definition

Put the selections in a JSON file. This also lets you publish custom prompts.
Replace the sample IDs with IDs from your deployment.

```json
{
  "name": "Compiler research",
  "tools": ["search"],
  "source_ids": ["6159cd56-4995-43b3-bbc9-fcfece28c29f"],
  "note_ids": ["5f89821a-4c01-4130-8267-c7d7a31c331e"],
  "prompts": [
    {
      "name": "benchmark_summary",
      "title": "Benchmark summary",
      "instructions": "Summarize the findings about {{topic}} and cite the sources.",
      "use_inputs": true
    }
  ]
}
```

```sh
kdbc mcp create --file profile.json --dry-run
kdbc mcp create --file profile.json
kdbc mcp inspect "$PROFILE_ID"
```

Use `--file -` to read from stdin. For document tools, use `source_ids` or
set `all_sources` to `true`. Preview prints the request locally. The server
checks the selected programs, sources, notes, and prompts when you create
the profile. Each create issues a new profile and key.

Commands return JSON by default. `auth login` returns the token as text,
`context --text` returns the Context Pack text, and `--markdown` returns note
Markdown. Errors go to stderr with a nonzero exit
status. The CLI does not store credentials on disk.
