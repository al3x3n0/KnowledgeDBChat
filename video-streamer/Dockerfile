# Build stage
FROM golang:1.23-alpine AS builder

WORKDIR /app

# Copy go mod files (go.sum may not exist initially)
COPY go.mod ./
COPY go.sum* ./

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

