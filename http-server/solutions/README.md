# Solutions

This directory contains complete implementations for the HTTP server project.

## Files

- **http_server.c** - Complete HTTP/1.1 server implementation

## Building and Running

```bash
gcc -o http_server http_server.c
./http_server [port]
```

Default port is 8080.

## Features

1. **TCP Socket Server**: Handles multiple client connections sequentially
2. **HTTP/1.1 Protocol**: Parses HTTP requests and generates responses
3. **GET Requests**: Serves static files from the `./public` directory
4. **Content Types**: Automatically detects file types (HTML, CSS, JS, images, etc.)
5. **Error Responses**: Returns 404 for missing files, 400 for bad requests
6. **Security**: Basic path traversal protection

## Testing

Create a `public` directory with test files:

```bash
mkdir public
echo "<h1>Hello World</h1>" > public/index.html
./http_server 8080
```

Test with:
```bash
curl http://localhost:8080/
curl http://localhost:8080/index.html
```

Or open `http://localhost:8080` in a web browser.

## Learning Points

- Socket programming with socket(), bind(), listen(), accept()
- HTTP protocol structure (request line, headers, body)
- Parsing HTTP requests
- Building HTTP responses with proper headers
- File I/O and MIME type detection
- Error handling and status codes

## Enhancements to Explore

- Concurrent connection handling (threads or select/epoll)
- POST request handling with form data parsing
- HTTPS support with OpenSSL/TLS
- HTTP/2 protocol features
- Virtual hosting
- CGI script support
- Access logging
- Configuration files
