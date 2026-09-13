// Mermaid to SVG/PNG, over the protocol the backend already speaks.
//
// This replaces yuzutech/kroki-mermaid, which is 1.54 GB: a headless Chromium
// (292 MB) plus 204 MB of mesa and libLLVM that a browser rendering an SVG off
// screen never opens, wrapped in a Vert.x service. What the backend actually
// uses is two routes -- POST /svg and POST /png with the raw diagram as the
// body -- because mermaid_renderer.py already talks to the companion rather
// than the Kroki gateway (KROKI_LOCAL_IS_COMPANION).
//
// One browser is launched at startup and kept. mermaid-cli's own CLI starts a
// browser per invocation, which is about two seconds of process spawn on every
// diagram in a document being exported.

import http from "node:http";

import { createRenderer } from "./render.js";

const PORT = Number(process.env.PORT || 8002);
const MAX_BODY_BYTES = Number(process.env.MAX_BODY_BYTES || 256 * 1024);

const renderer = await createRenderer();

const readBody = (req) =>
  new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new Error(`diagram exceeds ${MAX_BODY_BYTES} bytes`));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });

const server = http.createServer(async (req, res) => {
  const route = (req.url || "/").split("?")[0].replace(/\/+$/, "") || "/";

  if (req.method === "GET" && (route === "/health" || route === "/")) {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ status: "ok", formats: ["svg", "png"] }));
    return;
  }

  const format = { "/svg": "svg", "/png": "png" }[route];
  if (req.method !== "POST" || !format) {
    res.writeHead(404, { "Content-Type": "text/plain" });
    res.end("POST the diagram source to /svg or /png\n");
    return;
  }

  try {
    const source = (await readBody(req)).trim();
    if (!source) {
      res.writeHead(400, { "Content-Type": "text/plain" });
      res.end("empty diagram\n");
      return;
    }
    const data = await renderer.render(source, format);
    res.writeHead(200, {
      "Content-Type": format === "svg" ? "image/svg+xml" : "image/png",
      "Content-Length": data.length,
    });
    res.end(data);
  } catch (error) {
    // The backend reads the first 200 characters of this into its own error,
    // so say what was wrong with the diagram rather than logging it here only.
    const message = String(error?.message || error);
    console.error(`render failed: ${message}`);
    res.writeHead(400, { "Content-Type": "text/plain" });
    res.end(`${message}\n`);
  }
});

server.listen(PORT, () => console.log(`mermaid renderer listening on ${PORT}`));

for (const signal of ["SIGTERM", "SIGINT"]) {
  process.on(signal, async () => {
    server.close();
    await renderer.close();
    process.exit(0);
  });
}
