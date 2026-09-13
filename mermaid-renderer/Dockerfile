# syntax=docker/dockerfile:1

# The Mermaid renderer, without the browser's graphics stack.
#
# yuzutech/kroki-mermaid is 1.54 GB. Any Mermaid renderer needs a browser --
# Mermaid is JavaScript that lays out a DOM -- so most of that is irreducible.
# What is not irreducible: a JVM service to front it, npm to build it, and
# ~190 MB of mesa and libLLVM, the software OpenGL stack that a headless
# Chromium rasterising an SVG never opens.
#
# What the backend calls is two routes: POST /svg and POST /png with the raw
# diagram as the body. mermaid_renderer.py already speaks that (the companion
# protocol, KROKI_LOCAL_IS_COMPANION) rather than the Kroki gateway's encoded
# URLs, so nothing on the backend changes.

# Builder: npm and its 100 MB stay here.
FROM node:20-alpine AS deps
ENV PUPPETEER_SKIP_DOWNLOAD=true
WORKDIR /app
# `npm ci` against a committed lock file, not `npm install`: the two direct
# dependencies are pinned exactly, but a floating transitive is how a renderer
# that built last week stops rendering this week.
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund


FROM alpine:3.23

# `nodejs` rather than the node image: this needs the runtime, not the
# toolchain, and node:20-alpine carries npm and a build environment that only
# the stage above uses.
#
# Chromium comes from the distro rather than the ~170 MB puppeteer downloads.
# The font packages are not optional -- without them every label in every
# diagram renders as an empty box, which reads as a broken diagram rather than
# a missing font.
RUN apk add --no-cache \
        nodejs \
        chromium \
        font-noto \
        font-noto-emoji \
        ttf-dejavu \
    && rm -rf /var/cache/apk/* \
    # mesa's software GL and the libLLVM that JITs its shaders are pulled in as
    # chromium dependencies and come to ~190 MB. Headless Chromium rasterises
    # through its own bundled SwiftShader and never opens them: verified by
    # deleting exactly these files and re-rendering, which returned the same
    # SVG, byte count included. Deleted rather than `apk del`, which would take
    # chromium with them; this image is replaced, never upgraded in place.
    && rm -f /usr/lib/libLLVM* /usr/lib/libgallium* \
    && chromium-browser --headless --version

ENV PUPPETEER_SKIP_DOWNLOAD=true \
    PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser \
    NODE_ENV=production

# Chromium refuses to run as root without --no-sandbox, and running a browser
# that parses agent-written diagram source as root is worth avoiding anyway.
RUN addgroup -S renderer && adduser -S -G renderer renderer

WORKDIR /app
# --chown on the COPY, not a `chown -R` afterwards: chown rewrites every file's
# metadata, and a layer that touches 129 MB of node_modules stores all 129 MB
# again. Measured on this image: 1.24 GB against 1.11 GB.
COPY --from=deps --chown=renderer:renderer /app/node_modules ./node_modules
COPY --chown=renderer:renderer render.js server.js check.js package.json ./

USER renderer

EXPOSE 8002

# Asserted at build time: a browser that starts, and a real diagram that comes
# back as real SVG and a real PNG. An image whose renderer only fails on the
# first request is a failure this repo has already had once, from a different
# service.
RUN node check.js

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD wget --quiet --tries=1 --output-document=/dev/null http://localhost:8002/health || exit 1

CMD ["node", "server.js"]
