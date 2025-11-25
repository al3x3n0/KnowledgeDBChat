package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/stdlib"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

var (
	minioClient *minio.Client
	bucketName  string
	jwtSecret   string
	db          *sql.DB
)

func init() {
	// Load configuration from environment
	minioEndpoint := getEnv("MINIO_ENDPOINT", "minio:9000")
	minioAccessKey := getEnv("MINIO_ACCESS_KEY", "minioadmin")
	minioSecretKey := getEnv("MINIO_SECRET_KEY", "minioadmin")
	bucketName = getEnv("MINIO_BUCKET_NAME", "documents")
	useSSL := getEnv("MINIO_USE_SSL", "false") == "true"
	jwtSecret = getEnv("JWT_SECRET", "your-secret-key-change-in-production")

	// Initialize MinIO client
	var err error
	minioClient, err = minio.New(minioEndpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(minioAccessKey, minioSecretKey, ""),
		Secure: useSSL,
	})
	if err != nil {
		log.Fatalf("Failed to initialize MinIO client: %v", err)
	}

	log.Printf("Video streamer initialized - MinIO: %s, Bucket: %s", minioEndpoint, bucketName)
	
	// Initialize database connection
	databaseURL := getEnv("DATABASE_URL", "postgresql://user:password@postgres:5432/knowledge_db")
	// Convert postgresql:// to postgres:// for pgx
	databaseURL = strings.Replace(databaseURL, "postgresql://", "postgres://", 1)
	
	config, err := pgx.ParseConfig(databaseURL)
	if err != nil {
		log.Fatalf("Failed to parse database URL: %v", err)
	}
	
	db = stdlib.OpenDB(*config)
	db.SetMaxOpenConns(10)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(5 * time.Minute)
	
	// Test connection
	if err := db.Ping(); err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}
	
	log.Printf("Database connection established")
}

func main() {
	port := getEnv("PORT", "8080")
	corsOrigin := getEnv("CORS_ORIGIN", "*")

	// Set Gin mode
	if getEnv("GIN_MODE", "release") == "release" {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.Default()

	// CORS middleware
	router.Use(corsMiddleware(corsOrigin))

	// Health check
	router.GET("/health", healthCheck)

	// Video streaming endpoints
	router.OPTIONS("/stream/:document_id", handleOptions)
	router.HEAD("/stream/:document_id", handleHead)
	router.GET("/stream/:document_id", handleStream)

	log.Printf("Video streamer starting on port %s", port)
	if err := http.ListenAndServe(":"+port, router); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

func corsMiddleware(origin string) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", origin)
		c.Writer.Header().Set("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Range, Authorization, Content-Type")
		c.Writer.Header().Set("Access-Control-Expose-Headers", "Content-Range, Content-Length, Accept-Ranges, Content-Type")
		c.Writer.Header().Set("Access-Control-Max-Age", "86400")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	}
}

func healthCheck(c *gin.Context) {
	c.JSON(200, gin.H{"status": "ok", "service": "video-streamer"})
}

func handleOptions(c *gin.Context) {
	c.Status(204)
}

func handleHead(c *gin.Context) {
	documentID := c.Param("document_id")
	token := c.Query("token")
	authHeader := c.GetHeader("Authorization")

	log.Printf("HEAD request for document: %s, token present: %v, auth header present: %v", documentID, token != "", authHeader != "")

	// Authenticate
	if !authenticate(token, c.GetHeader("Authorization")) {
		log.Printf("Authentication failed for document: %s", documentID)
		c.JSON(401, gin.H{"error": "Authentication required"})
		return
	}

	// Get file path from database directly
	filePath, err := getFilePathFromDatabase(documentID)
	if err != nil {
		log.Printf("Failed to get file path for document %s: %v", documentID, err)
		c.JSON(404, gin.H{"error": "File not found"})
		return
	}

	log.Printf("Resolved file path for document %s: %s", documentID, filePath)

	// Get object info - try the resolved path first
	var objInfo minio.ObjectInfo
	var found bool
	
	// Try the resolved path first (if it's not just a directory prefix)
	if filePath != "" && !strings.HasSuffix(filePath, "/") {
		objInfo, err = minioClient.StatObject(context.Background(), bucketName, filePath, minio.StatObjectOptions{})
		if err == nil {
			found = true
			log.Printf("Found file at resolved path: %s", filePath)
		} else {
			log.Printf("MinIO StatObject failed for path %s: %v", filePath, err)
		}
	}
	
	// If not found, list objects with document ID prefix
	if !found {
		log.Printf("Listing objects in MinIO with prefix: %s/", documentID)
		ctx := context.Background()
		objectCh := minioClient.ListObjects(ctx, bucketName, minio.ListObjectsOptions{
			Prefix:    documentID + "/",
			Recursive: true,  // List recursively to find files in subdirectories
		})
		
		for object := range objectCh {
			if object.Err != nil {
				log.Printf("Error listing objects: %v", object.Err)
				continue
			}
			// Found an object with this document ID prefix
			filePath = object.Key
			objInfo, err = minioClient.StatObject(ctx, bucketName, filePath, minio.StatObjectOptions{})
			if err == nil {
				found = true
				log.Printf("Found file at: %s (size: %d, content-type: %s)", filePath, objInfo.Size, objInfo.ContentType)
				break
			} else {
				log.Printf("Failed to stat object %s: %v", filePath, err)
			}
		}
	}
	
	if !found {
		log.Printf("File not found in MinIO for document %s at any path", documentID)
		c.JSON(404, gin.H{"error": "File not found in storage"})
		return
	}

    // Set headers - ensure proper content type for media files
    contentType := objInfo.ContentType
    if contentType == "" || contentType == "application/octet-stream" {
        // Try to detect from filename
        filename := strings.ToLower(objInfo.Key)
        switch {
        case strings.HasSuffix(filename, ".mkv"):
            contentType = "video/x-matroska"
        case strings.HasSuffix(filename, ".mp4"):
            contentType = "video/mp4"
        case strings.HasSuffix(filename, ".webm"):
            contentType = "video/webm"
        case strings.HasSuffix(filename, ".avi"):
            contentType = "video/x-msvideo"
        case strings.HasSuffix(filename, ".mov"):
            contentType = "video/quicktime"
        case strings.HasSuffix(filename, ".mp3"):
            contentType = "audio/mpeg"
        case strings.HasSuffix(filename, ".wav"):
            contentType = "audio/wav"
        case strings.HasSuffix(filename, ".m4a"):
            contentType = "audio/mp4"
        case strings.HasSuffix(filename, ".flac"):
            contentType = "audio/flac"
        case strings.HasSuffix(filename, ".ogg"):
            contentType = "audio/ogg"
        case strings.HasSuffix(filename, ".aac"):
            contentType = "audio/aac"
        default:
            contentType = "application/octet-stream"
        }
    }
	
	c.Header("Content-Type", contentType)
	c.Header("Content-Length", strconv.FormatInt(objInfo.Size, 10))
	c.Header("Accept-Ranges", "bytes")
	c.Header("Content-Disposition", fmt.Sprintf(`inline; filename="%s"`, objInfo.Key))
	c.Status(200)
}

func handleStream(c *gin.Context) {
	documentID := c.Param("document_id")
	token := c.Query("token")
	authHeader := c.GetHeader("Authorization")
	rangeHeader := c.GetHeader("Range")

	log.Printf("GET request for document: %s, Range: %s, token present: %v, auth header present: %v", documentID, rangeHeader, token != "", authHeader != "")

	// Authenticate
	if !authenticate(token, c.GetHeader("Authorization")) {
		log.Printf("Authentication failed for document: %s", documentID)
		c.JSON(401, gin.H{"error": "Authentication required"})
		return
	}

	// Get file path from database directly
	filePath, err := getFilePathFromDatabase(documentID)
	if err != nil {
		log.Printf("Failed to get file path for document %s: %v", documentID, err)
		c.JSON(404, gin.H{"error": "File not found"})
		return
	}

	log.Printf("Resolved file path for document %s: %s", documentID, filePath)

	// Get object info - try the resolved path first
	var objInfo minio.ObjectInfo
	var found bool
	
	// Try the resolved path first (if it's not just a directory prefix)
	if filePath != "" && !strings.HasSuffix(filePath, "/") {
		objInfo, err = minioClient.StatObject(context.Background(), bucketName, filePath, minio.StatObjectOptions{})
		if err == nil {
			found = true
			log.Printf("Found file at resolved path: %s", filePath)
		} else {
			log.Printf("MinIO StatObject failed for path %s: %v", filePath, err)
		}
	}
	
	// If not found, list objects with document ID prefix
	if !found {
		log.Printf("Listing objects in MinIO with prefix: %s/", documentID)
		ctx := context.Background()
		objectCh := minioClient.ListObjects(ctx, bucketName, minio.ListObjectsOptions{
			Prefix:    documentID + "/",
			Recursive: true,  // List recursively to find files in subdirectories
		})
		
		for object := range objectCh {
			if object.Err != nil {
				log.Printf("Error listing objects: %v", object.Err)
				continue
			}
			// Found an object with this document ID prefix
			filePath = object.Key
			objInfo, err = minioClient.StatObject(ctx, bucketName, filePath, minio.StatObjectOptions{})
			if err == nil {
				found = true
				log.Printf("Found file at: %s (size: %d, content-type: %s)", filePath, objInfo.Size, objInfo.ContentType)
				break
			} else {
				log.Printf("Failed to stat object %s: %v", filePath, err)
			}
		}
	}
	
	if !found {
		log.Printf("File not found in MinIO for document %s at any path", documentID)
		c.JSON(404, gin.H{"error": "File not found in storage"})
		return
	}

	fileSize := objInfo.Size
	contentType := objInfo.ContentType
	
    // Ensure proper content type for media files
    if contentType == "" || contentType == "application/octet-stream" {
        // Try to detect from filename
        filename := strings.ToLower(objInfo.Key)
        switch {
        case strings.HasSuffix(filename, ".mkv"):
            contentType = "video/x-matroska"
        case strings.HasSuffix(filename, ".mp4"):
            contentType = "video/mp4"
        case strings.HasSuffix(filename, ".webm"):
            contentType = "video/webm"
        case strings.HasSuffix(filename, ".avi"):
            contentType = "video/x-msvideo"
        case strings.HasSuffix(filename, ".mov"):
            contentType = "video/quicktime"
        case strings.HasSuffix(filename, ".mp3"):
            contentType = "audio/mpeg"
        case strings.HasSuffix(filename, ".wav"):
            contentType = "audio/wav"
        case strings.HasSuffix(filename, ".m4a"):
            contentType = "audio/mp4"
        case strings.HasSuffix(filename, ".flac"):
            contentType = "audio/flac"
        case strings.HasSuffix(filename, ".ogg"):
            contentType = "audio/ogg"
        case strings.HasSuffix(filename, ".aac"):
            contentType = "audio/aac"
        default:
            contentType = "application/octet-stream"
        }
    }

	// Handle range request
	if rangeHeader != "" {
		start, end, err := parseRange(rangeHeader, fileSize)
		if err != nil {
			log.Printf("Invalid range header '%s' for document %s: %v", rangeHeader, documentID, err)
			c.Header("Content-Range", fmt.Sprintf("bytes */%d", fileSize))
			c.Status(416)
			return
		}

		log.Printf("Range request for document %s: bytes %d-%d of %d", documentID, start, end, fileSize)

		// Get object with range
		opts := minio.GetObjectOptions{}
		opts.SetRange(start, end)
		object, err := minioClient.GetObject(context.Background(), bucketName, filePath, opts)
		if err != nil {
			log.Printf("Failed to get object with range for document %s: %v", documentID, err)
			c.JSON(500, gin.H{"error": "Failed to stream file"})
			return
		}
		defer object.Close()

		// Set headers for partial content
		contentLength := end - start + 1
		c.Header("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, fileSize))
		c.Header("Content-Length", strconv.FormatInt(contentLength, 10))
		c.Header("Content-Type", contentType)
		c.Header("Accept-Ranges", "bytes")
		c.Header("Content-Disposition", fmt.Sprintf(`inline; filename="%s"`, objInfo.Key))
		c.Status(206)

		log.Printf("Streaming range %d-%d for document %s", start, end, documentID)
		// Stream the data
		c.DataFromReader(206, contentLength, contentType, object, nil)
		return
	}

	// Full file request
	log.Printf("Full file request for document %s (size: %d)", documentID, fileSize)
	object, err := minioClient.GetObject(context.Background(), bucketName, filePath, minio.GetObjectOptions{})
	if err != nil {
		log.Printf("Failed to get object for document %s: %v", documentID, err)
		c.JSON(500, gin.H{"error": "Failed to stream file"})
		return
	}
	defer object.Close()

	// Set headers
	c.Header("Content-Type", contentType)
	c.Header("Content-Length", strconv.FormatInt(fileSize, 10))
	c.Header("Accept-Ranges", "bytes")
	c.Header("Content-Disposition", fmt.Sprintf(`inline; filename="%s"`, objInfo.Key))
	c.Status(200)

	// Stream the data - use DataFromReader for proper streaming with Gin
	c.DataFromReader(200, fileSize, contentType, object, nil)
}

func parseRange(rangeHeader string, fileSize int64) (int64, int64, error) {
	// Parse "bytes=start-end"
	if !strings.HasPrefix(rangeHeader, "bytes=") {
		return 0, 0, fmt.Errorf("invalid range header")
	}

	rangeSpec := strings.TrimPrefix(rangeHeader, "bytes=")
	parts := strings.Split(rangeSpec, "-")
	if len(parts) != 2 {
		return 0, 0, fmt.Errorf("invalid range format")
	}

	var start, end int64
	var err error

	if parts[0] != "" {
		start, err = strconv.ParseInt(parts[0], 10, 64)
		if err != nil {
			return 0, 0, err
		}
	} else {
		start = 0
	}

	if parts[1] != "" {
		end, err = strconv.ParseInt(parts[1], 10, 64)
		if err != nil {
			return 0, 0, err
		}
	} else {
		end = fileSize - 1
	}

	// Validate range
	if start < 0 || end >= fileSize || start > end {
		return 0, 0, fmt.Errorf("invalid range")
	}

	return start, end, nil
}

func authenticate(tokenQuery, authHeader string) bool {
	if jwtSecret == "" {
		log.Printf("JWT_SECRET is not set!")
		return false
	}
	
	var token string

	// Prefer Authorization header over query parameter (more secure)
	if authHeader != "" && strings.HasPrefix(authHeader, "Bearer ") {
		token = strings.TrimPrefix(authHeader, "Bearer ")
		tokenPreviewLen := 20
		if len(token) < tokenPreviewLen {
			tokenPreviewLen = len(token)
		}
		log.Printf("Using token from Authorization header (length: %d, first 20 chars: %s)", len(token), token[:tokenPreviewLen])
	} else if tokenQuery != "" {
		token = tokenQuery
		tokenPreviewLen := 20
		if len(token) < tokenPreviewLen {
			tokenPreviewLen = len(token)
		}
		log.Printf("Using token from query parameter (length: %d, first 20 chars: %s)", len(token), token[:tokenPreviewLen])
	} else {
		log.Printf("No token found - tokenQuery empty: %v, authHeader present: %v", tokenQuery == "", authHeader != "")
		return false
	}

	if token == "" {
		log.Printf("Token is empty after extraction")
		return false
	}

	// Parse and validate JWT
	parsedToken, err := jwt.Parse(token, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		log.Printf("JWT signing method: %v, JWT_SECRET length: %d", token.Header["alg"], len(jwtSecret))
		return []byte(jwtSecret), nil
	})

	if err != nil {
		log.Printf("JWT parsing error: %v", err)
		return false
	}

	if !parsedToken.Valid {
		log.Printf("JWT token is not valid")
		return false
	}

	log.Printf("JWT token validated successfully")
	return true
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getFilePathFromDatabase(documentID string) (string, error) {
	// Query database directly to get file path
	query := `SELECT file_path FROM documents WHERE id = $1`
	
	var filePath sql.NullString
	err := db.QueryRow(query, documentID).Scan(&filePath)
	if err != nil {
		if err == sql.ErrNoRows {
			log.Printf("Document %s not found in database", documentID)
			return "", fmt.Errorf("document not found")
		}
		log.Printf("Database query error: %v", err)
		return "", err
	}
	
	if !filePath.Valid || filePath.String == "" {
		log.Printf("Document %s has no file_path in database", documentID)
		return fmt.Sprintf("%s/", documentID), nil
	}
	
	path := filePath.String
	log.Printf("Got file_path from database for document %s: %s", documentID, path)
	
	// Normalize file path (remove 'documents/' prefix if present)
	if strings.HasPrefix(path, "documents/") {
		path = strings.TrimPrefix(path, "documents/")
	}
	if strings.HasPrefix(path, "./") {
		path = strings.TrimPrefix(path, "./")
	}
	
	log.Printf("Normalized file path: %s", path)
	return path, nil
}

// Legacy function - kept for reference but not used
type DocumentResponse struct {
	ID       string `json:"id"`
	FilePath string `json:"file_path"`
	Title    string `json:"title"`
}

func getFilePathFromBackend(documentID, token string) (string, error) {
	// Query backend API to get file path
	backendURL := getEnv("BACKEND_API_URL", "http://backend:8000/api/v1/documents")
	// BACKEND_API_URL should be like "http://backend:8000/api/v1/documents"
	// So we need to construct the full URL
	url := fmt.Sprintf("%s/%s", backendURL, documentID)
	
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return "", err
	}
	
	if token != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
	}
	
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		// Fallback: try common path pattern
		log.Printf("Failed to query backend at %s, error: %v, using fallback path", url, err)
		return fmt.Sprintf("%s/", documentID), nil
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != 200 {
		// Read response body for debugging
		bodyBytes, _ := io.ReadAll(resp.Body)
		log.Printf("Backend returned %d for URL %s, response: %s, using fallback path", resp.StatusCode, url, string(bodyBytes))
		return fmt.Sprintf("%s/", documentID), nil
	}
	
	// Parse JSON response
	var doc DocumentResponse
	if err := json.NewDecoder(resp.Body).Decode(&doc); err != nil {
		log.Printf("Failed to parse response, using fallback: %v", err)
		// Fallback: try to list objects in MinIO with document ID prefix
		return fmt.Sprintf("%s/", documentID), nil
	}
	
	log.Printf("Got document from backend: ID=%s, FilePath=%s", doc.ID, doc.FilePath)
	
	// Normalize file path (remove 'documents/' prefix if present)
	filePath := doc.FilePath
	if strings.HasPrefix(filePath, "documents/") {
		filePath = strings.TrimPrefix(filePath, "documents/")
	}
	if strings.HasPrefix(filePath, "./") {
		filePath = strings.TrimPrefix(filePath, "./")
	}
	
	// If file path is empty or just document ID, try to find the actual file
	if filePath == "" || filePath == documentID || filePath == fmt.Sprintf("%s/", documentID) {
		log.Printf("File path is empty or just document ID, will try to find file in MinIO")
		// Return pattern that will be tried in handleHead/handleStream
		return fmt.Sprintf("%s/", documentID), nil
	}
	
	log.Printf("Normalized file path: %s", filePath)
	return filePath, nil
}
