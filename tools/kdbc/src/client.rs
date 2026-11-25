use std::env;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use reqwest::blocking::{Client, RequestBuilder, Response};
use reqwest::header::ACCEPT;
use reqwest::{Method, StatusCode, Url};
use serde_json::{json, Value};

pub struct Platform {
    base: Url,
    http: Client,
}

impl Platform {
    pub fn new(address: &str) -> Result<Self> {
        let mut base = Url::parse(address).context("Invalid platform URL")?;
        if !matches!(base.scheme(), "http" | "https")
            || base.host_str().is_none()
            || !base.username().is_empty()
            || base.password().is_some()
            || base.query().is_some()
            || base.fragment().is_some()
        {
            bail!("Use an HTTP(S) platform URL without credentials, a query, or a fragment");
        }
        if !base.path().ends_with('/') {
            base.path_segments_mut()
                .map_err(|_| anyhow!("Invalid platform URL path"))?
                .push("");
        }
        let http = Client::builder()
            .timeout(Duration::from_secs(60))
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(Self { base, http })
    }

    pub fn get(&self, path: &str, query: &[(&str, String)]) -> Result<Value> {
        self.get_with_timeout(path, query, Duration::from_secs(60))
    }

    pub fn get_with_timeout(
        &self,
        path: &str,
        query: &[(&str, String)],
        timeout: Duration,
    ) -> Result<Value> {
        let token = credential("KDBC_TOKEN", "Run kdbc auth login to get an account token.")?;
        let request = self
            .request(Method::GET, path)?
            .bearer_auth(token)
            .timeout(timeout)
            .query(query);
        decode(send(request)?)
    }

    pub fn post(&self, path: &str, body: Option<&Value>) -> Result<Value> {
        let token = credential("KDBC_TOKEN", "Run kdbc auth login to get an account token.")?;
        let mut request = self.request(Method::POST, path)?.bearer_auth(token);
        if let Some(body) = body {
            request = request.json(body);
        }
        decode(send(request)?)
    }

    pub fn login(&self, username: &str, password: &str) -> Result<String> {
        let request = self
            .request(Method::POST, "auth/login")?
            .json(&json!({ "username": username, "password": password }));
        let result = decode(send(request)?)?;
        result["access_token"]
            .as_str()
            .filter(|token| !token.is_empty())
            .map(str::to_owned)
            .context("Login response did not contain an access token")
    }

    pub fn mcp(&self) -> Result<Mcp<'_>> {
        let mut client = Mcp {
            platform: self,
            key: credential(
                "KDBC_MCP_KEY",
                "Set KDBC_MCP_KEY to the profile's connection key.",
            )?,
            protocol: None,
            next_id: 1,
            capabilities: Value::Null,
        };
        let response = client.call(
            "initialize",
            json!({
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": { "name": "kdbc", "version": env!("CARGO_PKG_VERSION") }
            }),
        )?;
        client.protocol = Some(
            response["protocolVersion"]
                .as_str()
                .filter(|version| !version.is_empty())
                .context("MCP server did not negotiate a protocol version")?
                .to_owned(),
        );
        client.capabilities = response["capabilities"].clone();
        let initialized =
            client.send(json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }))?;
        if initialized.status() != StatusCode::ACCEPTED {
            bail!(
                "Expected MCP initialization acknowledgment, got HTTP {}",
                initialized.status()
            );
        }
        Ok(client)
    }

    fn request(&self, method: Method, path: &str) -> Result<RequestBuilder> {
        let url = self.base.join(&format!("api/v1/{path}"))?;
        Ok(self.http.request(method, url))
    }
}

pub struct Mcp<'a> {
    platform: &'a Platform,
    key: String,
    protocol: Option<String>,
    next_id: u64,
    capabilities: Value,
}

impl Mcp<'_> {
    pub fn call(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        let mut message = decode(self.send(json!({
            "jsonrpc": "2.0", "id": id, "method": method, "params": params
        }))?)?;
        if message["jsonrpc"] != "2.0" || message["id"] != id {
            bail!("Unexpected JSON-RPC response for {method}");
        }
        if let Some(error) = message.get("error") {
            let code = error["code"].as_i64().context("Invalid JSON-RPC error")?;
            let detail = error["message"].as_str().unwrap_or("Request failed");
            bail!("MCP {method} failed ({code}): {detail}");
        }
        let result = message["result"].take();
        if !result.is_object() {
            bail!("MCP {method} did not return a result object");
        }
        Ok(result)
    }

    pub fn check(&mut self) -> Result<Value> {
        let mut result =
            json!({ "protocol_version": self.protocol, "tools": self.discover("tools")? });
        for feature in ["resources", "prompts"] {
            if self.capabilities[feature].is_object() {
                result[feature] = self.discover(feature)?;
            }
        }
        Ok(result)
    }

    fn discover(&mut self, feature: &str) -> Result<Value> {
        let mut response = self.call(&format!("{feature}/list"), json!({}))?;
        if !response[feature].is_array() {
            bail!("MCP server did not return a {feature} list");
        }
        Ok(response[feature].take())
    }

    fn send(&self, message: Value) -> Result<Response> {
        let mut request = self
            .platform
            .request(Method::POST, "mcp")?
            .bearer_auth(&self.key)
            .header(ACCEPT, "application/json, text/event-stream")
            .json(&message);
        if let Some(protocol) = &self.protocol {
            request = request.header("MCP-Protocol-Version", protocol);
        }
        send(request)
    }
}

fn credential(name: &str, hint: &str) -> Result<String> {
    let value = env::var(name).unwrap_or_default();
    if value.trim().is_empty() {
        bail!("Set {name}. {hint}");
    }
    Ok(value)
}

fn send(request: RequestBuilder) -> Result<Response> {
    let response = request
        .send()
        .map_err(reqwest::Error::without_url)
        .context("Could not reach the platform")?;
    let status = response.status();
    if !status.is_success() {
        let detail = response
            .json::<Value>()
            .ok()
            .and_then(|body| body["detail"].as_str().map(str::to_owned));
        match detail {
            Some(detail) => bail!("HTTP {status}: {detail}"),
            None => bail!("HTTP {status}"),
        }
    }
    Ok(response)
}

fn decode(response: Response) -> Result<Value> {
    response
        .json()
        .map_err(reqwest::Error::without_url)
        .context("Expected a JSON response")
}
