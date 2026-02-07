# HTTP Server From Scratch

This directory contains a from-scratch implementation of an HTTP server in C.

## Overview

Building an HTTP server is one of the best ways to understand how the web works at a fundamental level. This project will teach you about networking, protocols, file I/O, and how web servers like Apache and Nginx operate under the hood.

## What You'll Learn

- **TCP/IP Networking**: Socket programming, bind, listen, accept
- **HTTP Protocol**: Request/response structure, status codes, headers
- **Systems Programming**: File I/O, memory management, error handling
- **Web Architecture**: How web servers serve static content
- **Security Basics**: Path traversal prevention, input validation

## Project Structure

```
http-server/
├── README.md              # This file - project overview and guide
├── http_server.c          # Template with TODOs for implementation
├── Makefile              # Build configuration
├── public/               # Directory for serving static files
│   ├── index.html        # Test HTML page
│   ├── style.css         # Test CSS file
│   └── script.js         # Test JavaScript file
└── solutions/            # Complete working implementations
    ├── README.md         # Solution documentation and explanations
    └── http_server.c     # Fully implemented HTTP server
```

## Quick Start

1. **Build the template** (won't work yet - it's for you to implement):
   ```bash
   make
   ```

2. **Or build the solution** to see a working example:
   ```bash
   cd solutions/
   gcc -o http_server http_server.c
   ./http_server 8080
   ```

3. **Test it** by opening http://localhost:8080 in your browser

## Learning Path

This project is designed to be implemented incrementally. Follow these steps:

### Phase 1: Basic TCP Server (2-3 hours)
**Goal**: Create a server that accepts connections

1. **Socket Creation**
   - Use `socket()` to create a TCP socket
   - Set `SO_REUSEADDR` option for easier development
   
2. **Binding and Listening**
   - Bind socket to a port with `bind()`
   - Start listening with `listen()`
   
3. **Accepting Connections**
   - Accept incoming connections with `accept()`
   - Test with `telnet localhost 8080`

**Key Concepts**: File descriptors, socket addresses, network byte order

### Phase 2: HTTP Request Parsing (3-4 hours)
**Goal**: Parse incoming HTTP requests

1. **Read Request**
   - Use `recv()` to read data from client socket
   - HTTP requests are text-based
   
2. **Parse Request Line**
   - Extract: Method, Path, HTTP Version
   - Example: `GET /index.html HTTP/1.1`
   
3. **Parse Headers** (optional for basic version)
   - Headers follow the request line
   - Format: `Header-Name: value\r\n`

**Key Concepts**: String parsing, HTTP protocol structure, request format

### Phase 3: HTTP Response Generation (2-3 hours)
**Goal**: Send properly formatted HTTP responses

1. **Status Line**
   - Format: `HTTP/1.1 200 OK\r\n`
   - Common codes: 200 (OK), 404 (Not Found), 500 (Error)
   
2. **Response Headers**
   - `Content-Type`: MIME type of response
   - `Content-Length`: Size of response body in bytes
   - `Connection: close`: Close connection after response
   
3. **Response Body**
   - The actual content (HTML, CSS, JSON, etc.)
   - Sent after headers and blank line `\r\n\r\n`

**Key Concepts**: HTTP response structure, status codes, MIME types

### Phase 4: Static File Serving (3-4 hours)
**Goal**: Serve files from the filesystem

1. **File Reading**
   - Use `open()`, `fstat()`, `read()` to load files
   - Get file size for Content-Length header
   
2. **Path Mapping**
   - Map URL path to filesystem path
   - Example: `/index.html` → `./public/index.html`
   - Default to `index.html` for `/` requests
   
3. **Content Type Detection**
   - Determine MIME type from file extension
   - `.html` → `text/html`
   - `.css` → `text/css`
   - `.js` → `application/javascript`
   
4. **Error Handling**
   - Return 404 if file doesn't exist
   - Handle permission errors

**Key Concepts**: File I/O, MIME types, filesystem navigation

### Phase 5: Security and Error Handling (2-3 hours)
**Goal**: Make the server robust and secure

1. **Path Traversal Prevention**
   - Don't allow `../` in paths
   - Use `realpath()` to resolve paths safely
   
2. **Error Responses**
   - 400 Bad Request for malformed requests
   - 404 Not Found for missing files
   - 500 Internal Server Error for server issues
   
3. **Resource Cleanup**
   - Close file descriptors
   - Free allocated memory
   - Handle signals gracefully (SIGINT)

**Key Concepts**: Security vulnerabilities, error handling, resource management

### Phase 6 (Advanced): Concurrency (4-6 hours)
**Goal**: Handle multiple clients simultaneously

Options:
- **Fork**: Create child processes for each connection
- **Threads**: Use pthread library
- **select/poll**: Multiplexed I/O
- **epoll**: Linux high-performance event notification

**Key Concepts**: Concurrency models, race conditions, synchronization

### Phase 7 (Advanced): HTTP Features (varies)

Additional features to explore:
- POST request handling and body parsing
- Query parameter parsing (`/search?q=test`)
- Cookies and session management
- Range requests for resumable downloads
- Compression (gzip)
- Keep-alive connections
- Virtual hosting

## Testing Your Server

### Using curl
```bash
# Basic GET request
curl http://localhost:8080/

# Verbose output (see headers)
curl -v http://localhost:8080/

# Test 404
curl http://localhost:8080/nonexistent.html

# Save response to file
curl -o output.html http://localhost:8080/index.html
```

### Using a Browser
1. Start your server
2. Open http://localhost:8080 in Chrome/Firefox
3. Check browser's Network tab (F12) to see requests/responses

### Using netcat
```bash
# Send raw HTTP request
echo -e "GET / HTTP/1.1\r\nHost: localhost\r\n\r\n" | nc localhost 8080
```

### Load Testing
```bash
# Apache Bench
ab -n 1000 -c 10 http://localhost:8080/

# wrk (modern alternative)
wrk -t4 -c100 -d30s http://localhost:8080/
```

## Common Pitfalls

1. **Forgetting `\r\n` in HTTP**
   - HTTP uses CRLF (`\r\n`), not just LF (`\n`)
   - Headers end with `\r\n\r\n`

2. **Not Setting SO_REUSEADDR**
   - Without this, you can't restart server quickly after stopping
   - You'll get "Address already in use" errors

3. **Buffer Overflow**
   - Always check buffer sizes when reading
   - Use `strncpy()` instead of `strcpy()`

4. **Memory Leaks**
   - Free all allocated memory
   - Close all file descriptors
   - Use valgrind to check: `valgrind ./http_server`

5. **Path Traversal Security**
   - Never trust user input in file paths
   - Check for `..` sequences
   - Use `realpath()` to validate paths

6. **Binary Files**
   - Don't treat all files as text
   - Be careful with string operations on binary data

## Building and Running

```bash
# Build
make

# Run on default port (8080)
make run

# Run on custom port
./http_server 3000

# Clean build artifacts
make clean
```

## Features to Implement

Core features (in order of difficulty):
- ✅ Socket creation and binding
- ✅ Accepting connections  
- ✅ Reading HTTP requests
- ✅ Parsing request line (method, path)
- ✅ Sending HTTP responses
- ✅ Serving static files (HTML, CSS, JS)
- ✅ Content-Type detection
- ✅ 404 error handling
- ⬜ POST request handling
- ⬜ Request header parsing
- ⬜ Query parameter parsing
- ⬜ Concurrent connections

Advanced features:
- ⬜ Keep-alive connections
- ⬜ Chunked transfer encoding
- ⬜ HTTPS with OpenSSL
- ⬜ HTTP/2 protocol
- ⬜ WebSocket support
- ⬜ CGI script execution
- ⬜ Reverse proxy functionality

## Resources

### HTTP Protocol
- [RFC 7230: HTTP/1.1 Message Syntax and Routing](https://tools.ietf.org/html/rfc7230)
- [HTTP Status Codes](https://httpstatuses.com/)
- [MIME Types Reference](https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types)

### Socket Programming
- Beej's Guide to Network Programming
- Unix Network Programming by Stevens

### Similar Projects
- How nginx works
- How Apache HTTP Server works
- Lightweight servers: lighttpd, thttpd

## Troubleshooting

**Server won't start**
- Check if port is already in use: `lsof -i :8080`
- Try a different port
- Make sure you have permission to bind to the port

**Can't access from browser**
- Check firewall settings
- Verify server is listening: `netstat -an | grep 8080`
- Try `curl http://localhost:8080` first

**Files not found**
- Check that `public/` directory exists
- Verify file permissions
- Look at server logs for the path it's trying to access

**Connection refused**
- Server might not be running
- Check the port number
- Look for error messages in server output

## Next Steps

After completing this project:

1. **Add more features** from the advanced list above
2. **Optimize performance** with benchmarking tools
3. **Compare with nginx source** to see production patterns
4. **Build a web framework** on top of your server
5. **Try other languages** - implement in Rust, Go, Python for comparison

## License

Educational project for learning purposes.
