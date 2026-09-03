# learn-stuff-from-scratch

A collection of from-scratch implementations of various systems and projects for learning purposes.

## Start here

| | |
|---|---|
| **[PHILOSOPHY.md](PHILOSOPHY.md)** | What this repo is for, and the three principles every directory follows: design choices named as problem-solving decisions, MVP-then-complicate driven by limit cases, and verification you can run. **Read this first.** |
| **[curriculum/](curriculum/)** | Six course-and-book blocks (CS229, CS224N, CS230, CS336, CME295, CS329A), with an honest assessment of each pairing and the order to actually use — which is not 1 → 6. |
| **[curriculum/REPO-MAP.md](curriculum/REPO-MAP.md)** | All 34 project directories mapped onto those six blocks: what to build after what, and which half of this repository no block covers. |
| **[docs/archive/](docs/archive/)** | Session reports from earlier work, moved out of the root. Not documentation. |

**If you want somewhere concrete to start:** pick a directory that ships `check.py`. Those
give graded feedback that names the invariant you broke, which is worth more than three
directories that do not. Currently: `aws-from-scratch/` (18 checks), `dynamo-paper/` (17),
`context-caching/` (16), `contextcite/` (14), `compiler-and-vgpu/` (12),
`deploy-and-debug/` (12).

## Directory Structure

### Low-Level Systems (C/C++)
- **[c-compiler/](c-compiler/)** - C compiler implementation in C
- **[quantum-computing-lang/](quantum-computing-lang/)** - Quantum computing language and simulator (like Qiskit) in C
- **[cryptographic-library/](cryptographic-library/)** - Cryptographic primitives (SHA-256, ECDSA, etc.) in C
- **[bash-from-scratch/](bash-from-scratch/)** - Unix shell/terminal implementation
- **[http-server/](http-server/)** - HTTP server implementation
- **[dns-server/](dns-server/)** - DNS server implementation with UDP networking and protocol parsing
- **[firewall-from-scratch/](firewall-from-scratch/)** - Packet filtering firewall with raw sockets, protocol parsing, and rule-based filtering
- **[communication-protocols/](communication-protocols/)** - Serial & parallel communication protocol implementations: UART/USART, SPI, I2C, CAN bus, RS-232/RS-485 (with Linux spidev/i2c-dev/SocketCAN hardware support)

### GPU Programming & Parallel Computing
- **[cuda-from-scratch/](cuda-from-scratch/)** - CUDA parallel programming from basics to neural networks on GPU
- **[compiler-and-vgpu/](compiler-and-vgpu/)** - A compiler and a virtual GPU sharing one instruction set: 32-bit ISA, two-pass assembler, scalar CPU, recursive-descent front end, code generation with linear-scan register allocation and spilling, and a SIMT warp with divergence, mask stacks and barrier deadlock detection (12 graded checks via `python3 check.py`)

### Functional Programming & Formal Verification
- **[haskell-projects/](haskell-projects/)** - Various projects to learn Haskell
- **[lean-proofs/](lean-proofs/)** - Mathematical proofs in Lean, progressing toward Galois theorem
- **[curriculum/](curriculum/)** - Six course-and-book blocks with an honest assessment of each pairing: CS229 + Geron's Hands-On ML, CS224N + NLP with Transformers, CS230 + Goodfellow, CS336 + Raschka's Build a Large Language Model, CME295 + the Hugging Face LLM course, and CS329A + Hur & Song's Build an AI Agent. Includes what to skip given a mathematics background, why the CS230/Goodfellow pairing is weak (the book predates transformers by a year), why blocks 2, 4 and 5 teach attention three times, and [REPO-MAP.md](curriculum/REPO-MAP.md) mapping all 34 project directories onto the blocks

### Machine Learning & MLOps
- **[distributed-training/](distributed-training/)** - Distributed training systems (data parallelism, model parallelism, multi-node training)
- **[ml-in-production/](ml-in-production/)** - Production ML systems (model serving, monitoring, A/B testing)
- **[mlops/](mlops/)** - MLOps pipelines (experiment tracking, CI/CD, feature stores)
- **[ml-inference/](ml-inference/)** - High-performance inference (optimization, quantization, edge deployment)

### Generative AI & Deep Learning
- **[diffusion-models/](diffusion-models/)** - Diffusion models from scratch (DDPM, DDIM, U-Net, image generation like Stable Diffusion)
- **[deepfake-creation/](deepfake-creation/)** - Deepfake generation techniques (face swapping, reenactment, First Order Motion Model, Wav2Lip)
- **[deepfake-detection/](deepfake-detection/)** - Deepfake detection methods (CNN-based, temporal analysis, frequency domain, biological signals)

### ML Infrastructure & Serving
- **[sgl-lang/](sgl-lang/)** - Structured Generation Language (SGL) for LLMs - constrained generation, grammar enforcement, compilation
- **[tensorrt-inference/](tensorrt-inference/)** - TensorRT-style inference engine - graph optimization, quantization, kernel auto-tuning
- **[vllm-engine/](vllm-engine/)** - vLLM serving engine - PagedAttention, continuous batching, high-throughput LLM serving
- **[context-caching/](context-caching/)** - LLM context caching from scratch on a tiny pure-Python transformer: KV cache, block-hash and radix-tree prefix caching, paged KV blocks with copy-on-write, semantic response caching, and cache-aware request routing (16 graded checks via `python3 check.py`)
- **[contextcite/](contextcite/)** - ContextCite (NeurIPS 2024) replicated from scratch: context attribution by ablating sources and fitting a sparse LASSO surrogate - source partitioning, logit-probability scoring, coordinate-descent LASSO, held-out LDS evaluation, and the paper's three applications (14 graded checks via `python3 check.py`)

### System Design & Distributed Systems
- **[system-design/](system-design/)** - Core distributed systems patterns: caching (LRU, cache-aside, stampede), async queues (retries, backoff, DLQ, idempotency), reliability (circuit breaker, bulkhead, backpressure), consistent hashing, leaderboards, URL shortener, rate limiter, and capacity math
- **[dynamo-paper/](dynamo-paper/)** - Amazon's Dynamo paper (SOSP 2007) implemented directly: consistent hashing with preference lists, vector clocks, N/R/W quorums, sloppy quorum with hinted handoff, Merkle-tree anti-entropy, and gossip membership (17 graded checks via `python3 check.py`)
- **[aws-from-scratch/](aws-from-scratch/)** - Learn AWS by implementing toy versions of its core services: IAM policy evaluation, S3 with versioning and delete markers, SQS visibility timeouts, DynamoDB hot partitions, Lambda concurrency and cold starts, SNS filter policies and EventBridge patterns, KMS envelope encryption, VPC stateful-vs-stateless networking, plus a capstone pipeline wiring them together - and a map of which remaining AWS services are variations of which mechanism (18 graded checks via `python3 check.py`)

### Operations & Reliability
- **[deploy-and-debug/](deploy-and-debug/)** - Running the systems in this repo and debugging them when they break: capacity math (KV cache sizing, N/R/W failure tolerance), percentiles/queueing/error budgets, root-cause diagnosis of 11 injected faults from metrics alone, and safe rollout (liveness vs readiness, canary analysis, budget-based auto-rollback) - plus a runbook of the real vllm/nodetool/nvidia-smi/k8s commands (12 graded checks via `python3 check.py`)

### Data Engineering & Analytics
- **[sas-lineage-tool/](sas-lineage-tool/)** - SAS field lineage parser for tracking data transformations and dependencies
- **[web-scraping/](web-scraping/)** - Industrial web scraping/crawler library (Python/C, CUDA acceleration, CAPTCHA bypass, distributed architecture)

### Quantitative Finance & Trading
- **[quantitative-trading/](quantitative-trading/)** - Algorithmic trading systems (statistical arbitrage, ML strategies, RL agents, market microstructure)

## Philosophy

The full statement is in [PHILOSOPHY.md](PHILOSOPHY.md). In short: learn from the
implementation itself, name every design decision along with its cost, build the MVP and
then complicate it in response to a concrete limit case, and ship verification you can run.

Most directories contain:
- **Template files** with TODO comments and implementation guidelines
- **Step-by-step instructions** for gradual implementation
- **Reference solutions** in a `solutions/` folder
- A clear learning path from basics to advanced topics

*Most*, not all. `curriculum/` and `docs/` are documentation. `mlops/` has no solutions
yet. The `compression-lower-bounds/` track, on its own branch, deliberately ships no
solutions at all — the material there is short enough that reading one costs the whole
exercise, and its `check.py` names the broken invariant instead.

## Structure

Each project directory contains:

```
project-name/
├── README.md              # Project overview and learning path
├── template-files         # Empty templates with TODOs and guidelines
├── check.py               # Graded checks against YOUR code (in six directories so far)
├── Makefile               # Build configuration (for C projects)
└── solutions/             # Reference implementations (absent by design in some tracks)
    ├── README.md
    └── solution-files
```

## Getting Started

1. **Choose a project** that interests you
2. **Read the README** in that directory to understand the goals
3. **Start with the template files** - they have TODOs and guidelines
4. **Implement gradually** - follow the TODO comments step by step
5. **Test frequently** - build and test as you implement each section
6. **Run `check.py`** if the directory has one — it tests your code and says which
   invariant broke, which is more useful than a solution
7. **Check solutions** when stuck, and only then
8. **Learn and iterate** - understand each step before moving forward

## Implementation Approach

The templates are designed to be:
- ✅ **Gradual**: Start simple, add complexity incrementally
- ✅ **Guided**: Clear TODO comments explain what to implement
- ✅ **Balanced**: Not too easy (no hand-holding), not too hard (reasonable steps)
- ✅ **Educational**: Focus on understanding concepts, not just copying code

## Building Projects

Most C projects include a Makefile:

```bash
cd project-name/
make          # Build the project
make run      # Run the program
make test     # Run tests (if available)
make clean    # Clean build artifacts
```

## Video Courses & Learning Resources

> For a **structured** path through the machine-learning half of this repository — six
> course-and-book pairings with prerequisites, ordering and per-block builds — see
> [`curriculum/`](curriculum/). The list below is broader and unstructured, and it is the
> better resource for the systems, networking, cryptography and formal-methods
> directories, which no curriculum block covers.

To complement the hands-on projects in this repository, we've curated relevant video courses from universities and online platforms. These courses provide theoretical foundations and different perspectives on the topics covered here.

**Note**: This curated list is based on the excellent [cs-video-courses](https://github.com/Developer-Y/cs-video-courses) repository by Developer-Y, which maintains a comprehensive collection of Computer Science courses with video lectures.

### General Computer Science
- [CS 50 - Introduction to Computer Science, Harvard University](https://online-learning.harvard.edu/course/cs50-introduction-computer-science)
- [6.0001 - Introduction to Computer Science and Programming in Python - MIT OCW](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/video_galleries/lecture-videos/)

### Systems Programming & Operating Systems
*Relevant for: bash-from-scratch, http-server, dns-server, c-compiler*
- [15-213 Introduction to Computer Systems - CMU](https://scs.hosted.panopto.com/Panopto/Pages/Sessions/List.aspx#folderID=%22b96d90ae-9871-4fae-91e2-b1627b43e25e%22&maxResults=150)
- [CS 162 Operating Systems - UC Berkeley](https://archive.org/details/ucberkeley-webcast-PL-XXv-cvA_iBDyz-ba4yDskqMDY6A1w_c?sort=titleSorter)
- [6.824 - Distributed Systems - MIT](https://pdos.csail.mit.edu/6.824/schedule.html)

### Compiler Design & Programming Languages
*Relevant for: c-compiler, quantum-computing-lang*
- [CS143 - Compilers - Stanford](https://web.stanford.edu/class/cs143/)
- [Theoretical CS and Programming Languages courses](https://github.com/Developer-Y/cs-video-courses#theoretical-cs-and-programming-languages)

### Cryptography & Security
*Relevant for: cryptographic-library*
- [Security Courses - Various Universities](https://github.com/Developer-Y/cs-video-courses#security)

### Parallel Computing & GPU Programming
*Relevant for: cuda-from-scratch*
- [Parallel Computing and GPU Programming courses](https://github.com/Developer-Y/cs-video-courses#computer-organization-and-architecture)

### Machine Learning & Deep Learning
*Relevant for: distributed-training, ml-in-production, mlops, ml-inference, diffusion-models, deepfake-creation, deepfake-detection, world-models*
- [CS229 - Machine Learning - Stanford](http://cs229.stanford.edu/)
- [6.S191 - Introduction to Deep Learning - MIT](http://introtodeeplearning.com/)
- [Deep Learning Specialization - Various Universities](https://github.com/Developer-Y/cs-video-courses#deep-learning)
- [Computer Vision Courses](https://github.com/Developer-Y/cs-video-courses#computer-vision)
- [Generative AI and LLMs](https://github.com/Developer-Y/cs-video-courses#generative-ai-and-llms)

### MLOps & Production ML
*Relevant for: sgl-lang, tensorrt-inference, vllm-engine*
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/)
- [Machine Learning Systems Design](https://github.com/Developer-Y/cs-video-courses#machine-learning)

### Functional Programming
*Relevant for: haskell-projects*
- [FP 101x - Introduction to Functional Programming - TU Delft](https://ocw.tudelft.nl/courses/introduction-to-functional-programming/)
- [Functional Programming courses](https://github.com/Developer-Y/cs-video-courses#theoretical-cs-and-programming-languages)

### Formal Verification
*Relevant for: lean-proofs*
- [Formal Methods and Verification courses](https://github.com/Developer-Y/cs-video-courses#theoretical-cs-and-programming-languages)

### Quantum Computing
*Relevant for: quantum-computing-lang*
- [Quantum Computing Courses](https://github.com/Developer-Y/cs-video-courses#quantum-computing)

### Computational Finance
*Relevant for: quantitative-trading*
- [Computational Finance Courses](https://github.com/Developer-Y/cs-video-courses#computational-finance)

### System Design & Distributed Systems
*Relevant for: system-design*
- [CS 75 - Building Dynamic Websites - Harvard](https://cs75.tv/2012/summer/)
- [6.824 - Distributed Systems - MIT](https://pdos.csail.mit.edu/6.824/schedule.html)
- [CMU 15-445 Database Systems](https://15445.courses.cs.cmu.edu/fall2022/)
- [Database Systems Courses](https://github.com/Developer-Y/cs-video-courses#database-systems)
- [Distributed Systems Courses](https://github.com/Developer-Y/cs-video-courses#distributed-systems)

### Data Engineering
*Relevant for: sas-lineage-tool, web-scraping*
- [Database Systems Courses](https://github.com/Developer-Y/cs-video-courses#database-systems)

### Computer Networks
*Relevant for: http-server, dns-server, communication-protocols*
- [Computer Networks Courses](https://github.com/Developer-Y/cs-video-courses#computer-networks)

**For a complete list of courses across all CS topics**, visit the [Developer-Y/cs-video-courses](https://github.com/Developer-Y/cs-video-courses) repository.

## Note

These implementations are for educational purposes. They prioritize clarity and understanding over production-ready features or performance. Use the solutions as reference, but try to implement yourself first for maximum learning!