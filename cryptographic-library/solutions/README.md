# Solutions

This directory contains complete implementations of cryptographic algorithms.

## Files

- **sha256.c** - Complete SHA-256 hash function implementation

## Building and Running

```bash
gcc -o sha256 sha256.c
./sha256
```

## SHA-256 Implementation

The solution includes:
- Complete SHA-256 hashing according to FIPS 180-4
- Proper message padding
- Block transformation with compression function
- All logical functions (CH, MAJ, EP0, EP1, SIG0, SIG1)
- Test vectors for verification

### Features

1. **Context Management**: Initialize, update, and finalize hash computation
2. **Streaming API**: Process data in chunks of any size
3. **Test Vectors**: Validates against official NIST test vectors

### Usage

```c
uint8_t hash[SHA256_DIGEST_SIZE];
sha256((uint8_t*)"hello", 5, hash);
// hash now contains the SHA-256 digest
```

Or with streaming:
```c
SHA256_CTX ctx;
sha256_init(&ctx);
sha256_update(&ctx, data1, len1);
sha256_update(&ctx, data2, len2);
sha256_final(&ctx, hash);
```

## Learning Points

- Cryptographic hash function design
- Bitwise operations and rotations
- Message padding and length encoding
- Big-endian/little-endian handling
- Block cipher modes
- Merkle–Damgård construction

## Security Note

⚠️ **This implementation is for educational purposes only.**

For production use:
- Use established libraries (OpenSSL, libsodium)
- This code has not been audited for security
- Timing attacks and side-channel vulnerabilities not addressed
- No hardware acceleration

## Test Vectors

The implementation passes official NIST test vectors:

```
Input: "abc"
SHA-256: ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad

Input: "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
SHA-256: 248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1
```

## Further Reading

- FIPS 180-4 Specification
- "Applied Cryptography" by Bruce Schneier
- https://www.di-mgt.com.au/sha_testvectors.html
