# Build stage
FROM golang:1.23-alpine AS builder

WORKDIR /app

# Install git
RUN apk add --no-cache git

# Copy go mod files (go.sum may not exist initially)
COPY go.mod ./
COPY go.sum* ./

# Set build variables related to self-signed certs
ARG GIT_SSL_NO_VERIFY=false 
ARG GOPROXY=proxy.golang.org 
ARG GOSUMDB=on
ARG GOINSECURE=""

ENV GOPROXY=${GOPROXY} \
    GOSUMDB=${GOSUMDB} \
    GOINSECURE=${GOINSECURE} \
    GIT_SSL_NO_VERIFY=${GIT_SSL_NO_VERIFY}

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o video-streamer .

# Final stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates

WORKDIR /root/

# Copy the binary from builder
COPY --from=builder /app/video-streamer .

EXPOSE 8080

CMD ["./video-streamer"]

