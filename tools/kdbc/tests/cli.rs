use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Ipv4Addr, TcpListener};
use std::process::{Command, Output, Stdio};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use serde_json::{json, Value};

const TRACE: &str = "7c4194e4-656e-4c09-bffc-f1ce2ebff16d";
const NOTE: &str = "5f89821a-4c01-4130-8267-c7d7a31c331e";
const SOURCE: &str = "6159cd56-4995-43b3-bbc9-fcfece28c29f";
const WORKFLOW: &str = "745f36df-7f3c-444a-a3a3-b18493442c3e";
const EXECUTION: &str = "4cc6b44f-5d14-4fc6-802e-aef8480a9bb5";

struct Exchange {
    request: String,
    token: Option<&'static str>,
    body: Value,
    status: &'static str,
    response: Value,
}

fn serve(exchanges: Vec<Exchange>) -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let url = format!("http://{}/lab", listener.local_addr().unwrap());
    let worker = thread::spawn(move || {
        for exchange in exchanges {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut reader = BufReader::new(&mut stream);
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            assert_eq!(line.trim_end(), format!("{} HTTP/1.1", exchange.request));
            let mut authorization = None;
            let mut length = 0;
            loop {
                line.clear();
                assert_ne!(reader.read_line(&mut line).unwrap(), 0);
                let Some((name, value)) = line.trim_end().split_once(':') else {
                    break;
                };
                match name.to_ascii_lowercase().as_str() {
                    "authorization" => authorization = Some(value.trim().to_owned()),
                    "content-length" => length = value.trim().parse().unwrap(),
                    _ => {}
                }
            }
            assert_eq!(
                authorization,
                exchange.token.map(|token| format!("Bearer {token}"))
            );
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            let body: Value = if body.is_empty() {
                Value::Null
            } else {
                serde_json::from_slice(&body).unwrap()
            };
            assert_eq!(body, exchange.body);
            let response = if exchange.response.is_null() {
                String::new()
            } else {
                exchange.response.to_string()
            };
            write!(
                stream,
                "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                exchange.status,
                response.len(),
                response,
            ).unwrap();
        }
    });
    (url, worker)
}

fn run(url: &str, args: &[&str], input: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_kdbc"))
        .args(args)
        .env("KDBC_URL", url)
        .env("KDBC_TOKEN", "account-token")
        .env("KDBC_MCP_KEY", "profile-key")
        .env("NO_PROXY", "*")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    child.wait_with_output().unwrap()
}

fn output(result: Output) -> Value {
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    serde_json::from_slice(&result.stdout).unwrap()
}

#[test]
fn saves_context_and_creates_a_scoped_profile() {
    let (url, worker) = serve(vec![
        Exchange {
            request: "POST /lab/api/v1/auth/login".into(),
            token: None,
            body: json!({"username":"researcher", "password":"fake password"}),
            status: "200 OK",
            response: json!({"access_token":"account-token"}),
        },
        Exchange {
            request: format!("POST /lab/api/v1/research-notes/context-packs/{TRACE}"),
            token: Some("account-token"),
            body: Value::Null,
            status: "200 OK",
            response: json!({"id":NOTE, "source_document_ids":[SOURCE]}),
        },
        Exchange {
            request: "POST /lab/api/v1/mcp-config/profiles".into(),
            token: Some("account-token"),
            body: json!({
                "name":"Experiment",
                "programs":["lab"],
                "tools":["search"],
                "note_ids":[NOTE],
                "source_ids":[SOURCE]
            }),
            status: "201 Created",
            response: json!({"id":SOURCE, "api_key":"one-time-key"}),
        },
    ]);
    let login = run(
        &url,
        &["auth", "login", "--username", "researcher"],
        b"fake password\n",
    );
    assert!(
        login.status.success(),
        "{}",
        String::from_utf8_lossy(&login.stderr)
    );
    assert_eq!(login.stdout, b"account-token\n");
    let saved = output(run(&url, &["notes", "save-context", TRACE], b""));
    assert_eq!(saved["id"], NOTE);
    let profile = output(run(
        &url,
        &[
            "mcp",
            "create",
            "--name",
            "Experiment",
            "--note",
            NOTE,
            "--program",
            "lab",
            "--tool",
            "search",
            "--source",
            SOURCE,
        ],
        b"",
    ));
    assert_eq!(profile["api_key"], "one-time-key");
    worker.join().unwrap();
}

#[test]
fn reads_published_markdown_with_the_profile_key() {
    let rpc = |body, status, response| Exchange {
        request: "POST /lab/api/v1/mcp".into(),
        token: Some("profile-key"),
        body,
        status,
        response,
    };
    let (url, worker) = serve(vec![
        rpc(
            json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{
                "protocolVersion":"2025-11-25","capabilities":{},
                "clientInfo":{"name":"kdbc","version":env!("CARGO_PKG_VERSION")}
            }}),
            "200 OK",
            json!({"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{"resources":{}}}}),
        ),
        rpc(
            json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
            "202 Accepted",
            Value::Null,
        ),
        rpc(
            json!({"jsonrpc":"2.0","id":2,"method":"resources/read","params":{
                "uri":format!("kdbc://research-notes/{NOTE}")
            }}),
            "200 OK",
            json!({"jsonrpc":"2.0","id":2,"result":{"contents":[{"text":"Finding [[S1]]"}]}}),
        ),
    ]);
    let result = run(&url, &["mcp", "read-note", NOTE, "--markdown"], b"");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(result.stdout, b"Finding [[S1]]\n");
    worker.join().unwrap();
}

#[test]
fn rejects_implicit_access_to_all_sources() {
    let result = run(
        "http://localhost:23000",
        &["mcp", "create", "--name", "Lab", "--tool", "search"],
        b"",
    );
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8_lossy(&result.stderr).contains("explicitly pass --all-sources"));
}

#[test]
fn inspects_context_and_watches_the_execution_returned_by_queue() {
    let execution_path = format!("GET /lab/api/v1/workflows/executions/{EXECUTION}");
    let (url, worker) = serve(vec![
        Exchange {
            request: format!("GET /lab/api/v1/retrieval-traces/{TRACE}"),
            token: Some("account-token"),
            body: Value::Null,
            status: "200 OK",
            response: json!({"trace":{"kg_context_pack":{"kg_context":"Recorded evidence"}}}),
        },
        Exchange {
            request: format!("POST /lab/api/v1/workflows/{WORKFLOW}/execute/async"),
            token: Some("account-token"),
            body: json!({"trigger_type":"manual","trigger_data":{},"inputs":{}}),
            status: "202 Accepted",
            response: json!({"execution_id":EXECUTION,"status":"pending"}),
        },
        Exchange {
            request: execution_path.clone(),
            token: Some("account-token"),
            body: Value::Null,
            status: "200 OK",
            response: json!({"id":EXECUTION,"status":"running"}),
        },
        Exchange {
            request: execution_path,
            token: Some("account-token"),
            body: Value::Null,
            status: "200 OK",
            response: json!({"id":EXECUTION,"status":"completed","progress":100}),
        },
    ]);
    let context = run(&url, &["context", TRACE, "--text"], b"");
    assert!(
        context.status.success(),
        "{}",
        String::from_utf8_lossy(&context.stderr)
    );
    assert_eq!(context.stdout, b"Recorded evidence\n");
    let queued = output(run(&url, &["workflows", "run", WORKFLOW], b""));
    assert_eq!(queued["execution_id"], EXECUTION);
    let watched = output(run(
        &url,
        &[
            "workflows",
            "watch",
            queued["execution_id"].as_str().unwrap(),
            "--interval",
            "1",
        ],
        b"",
    ));
    assert_eq!(watched["progress"], 100);
    worker.join().unwrap();
}

#[test]
fn failed_workflow_watch_exits_nonzero_and_still_returns_json() {
    let failed = json!({"id":EXECUTION,"status":"failed","error":"Step failed"});
    let (url, worker) = serve(vec![Exchange {
        request: format!("GET /lab/api/v1/workflows/executions/{EXECUTION}"),
        token: Some("account-token"),
        body: Value::Null,
        status: "200 OK",
        response: failed.clone(),
    }]);
    let result = run(&url, &["workflows", "watch", EXECUTION], b"");
    assert!(!result.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&result.stdout).unwrap(),
        failed
    );
    assert!(String::from_utf8_lossy(&result.stderr).contains(EXECUTION));
    worker.join().unwrap();
}

#[test]
fn previews_and_publishes_a_profile_definition_with_a_prompt() {
    let definition = json!({
        "name": "Benchmark research",
        "tools": ["search"],
        "all_sources": true,
        "note_ids": [NOTE],
        "prompts": [{
            "name": "benchmark_summary",
            "title": "Benchmark summary",
            "instructions": "Summarize {{topic}}.",
            "use_inputs": true,
        }],
    });
    let input = definition.to_string();
    let preview = output(run(
        "http://localhost:23000",
        &["mcp", "create", "--file", "-", "--dry-run"],
        input.as_bytes(),
    ));
    assert!(preview.get("all_sources").is_none());
    assert!(preview.get("source_ids").is_none());
    assert_eq!(preview["prompts"], definition["prompts"]);
    assert_eq!(preview["note_ids"], definition["note_ids"]);

    let config = json!({"api_key_id":SOURCE,"mcp_programs":[],"source_access_mode":"all"});
    let (url, worker) = serve(vec![
        Exchange {
            request: "POST /lab/api/v1/mcp-config/profiles".into(),
            token: Some("account-token"),
            body: preview,
            status: "201 Created",
            response: json!({"id":SOURCE,"api_key":"one-time-key"}),
        },
        Exchange {
            request: format!("GET /lab/api/v1/mcp-config/keys/{SOURCE}/config"),
            token: Some("account-token"),
            body: Value::Null,
            status: "200 OK",
            response: config.clone(),
        },
    ]);
    let created = output(run(
        &url,
        &["mcp", "create", "--file", "-"],
        input.as_bytes(),
    ));
    assert_eq!(created["api_key"], "one-time-key");
    let inspected = output(run(
        &url,
        &["mcp", "inspect", created["id"].as_str().unwrap()],
        b"",
    ));
    assert_eq!(inspected, config);
    worker.join().unwrap();
}

#[test]
fn rejects_a_misspelled_source_field_in_a_profile_definition() {
    let result = run(
        "http://localhost:23000",
        &["mcp", "create", "--file", "-", "--dry-run"],
        br#"{"name":"Lab","tools":["search"],"sources_ids":[]}"#,
    );
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8_lossy(&result.stderr).contains("unknown field `sources_ids`"));
}
