// One browser, kept, rendering Mermaid to SVG or PNG.
//
// Separated from the server so that the image can prove at build time that a
// diagram renders, without starting a listener that would never exit.

import puppeteer from "puppeteer";
import { renderMermaid } from "@mermaid-js/mermaid-cli";

const RENDER_TIMEOUT_MS = Number(process.env.RENDER_TIMEOUT_MS || 30_000);

const withTimeout = (promise, ms) =>
  Promise.race([
    promise,
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error(`render timed out after ${ms}ms`)), ms),
    ),
  ]);

export async function createRenderer() {
  // --no-sandbox: the process is already confined to this container, and
  // Chromium's own sandbox needs privileges the container does not have.
  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });

  return {
    async render(source, format) {
      const { data } = await withTimeout(
        renderMermaid(browser, source, format, {
          backgroundColor: process.env.BACKGROUND_COLOR || "white",
        }),
        RENDER_TIMEOUT_MS,
      );
      return data;
    },
    close: () => browser.close(),
  };
}
