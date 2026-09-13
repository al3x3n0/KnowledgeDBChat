# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies.
# --legacy-peer-deps: react-scripts@5.0.1 pins a peer of typescript@^3||^4 while
# the project is on typescript@^5.4.5, so a plain `npm ci` fails ERESOLVE. npm 6
# resolution matches what CRA actually builds with; drop the flag once
# react-scripts is replaced or upgraded.
RUN npm ci --legacy-peer-deps

# Copy source code
COPY . .

# CRA inlines process.env.REACT_APP_* at build time, so the API origin has to be
# known here rather than at container start. Leave unset and the app falls back
# to http://localhost:8000 (see src/services/api.ts); pass the public origin of
# whatever fronts the API instead:
#   docker build --build-arg REACT_APP_API_URL=https://kdbc.example.com ...
ARG REACT_APP_API_URL
ARG REACT_APP_WS_URL
ENV REACT_APP_API_URL=${REACT_APP_API_URL}
ENV REACT_APP_WS_URL=${REACT_APP_WS_URL}

# Video playback: getDocumentDownloadUrl otherwise infers the streamer route by
# testing the API base URL for port 3000, which only matches docker-compose.
# Setting this explicitly (to <origin>/video) takes the first branch and works
# behind any reverse proxy.
ARG REACT_APP_VIDEO_STREAM_URL
ENV REACT_APP_VIDEO_STREAM_URL=${REACT_APP_VIDEO_STREAM_URL}

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built application
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]









