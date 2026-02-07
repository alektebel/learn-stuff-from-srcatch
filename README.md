# learn-stuff-from-scratch

A collection of from-scratch implementations of various systems and projects for learning purposes.

## Directory Structure

### Low-Level Systems (C/C++)
- **[c-compiler/](c-compiler/)** - C compiler implementation in C
- **[quantum-computing-lang/](quantum-computing-lang/)** - Quantum computing language and simulator (like Qiskit) in C
- **[cryptographic-library/](cryptographic-library/)** - Cryptographic primitives (SHA-256, ECDSA, etc.) in C
- **[bash-from-scratch/](bash-from-scratch/)** - Unix shell/terminal implementation
- **[http-server/](http-server/)** - HTTP server implementation

### GPU Programming & Parallel Computing
- **[cuda-from-scratch/](cuda-from-scratch/)** - CUDA parallel programming from basics to neural networks on GPU

### Functional Programming & Formal Verification
- **[haskell-projects/](haskell-projects/)** - Various projects to learn Haskell
- **[lean-proofs/](lean-proofs/)** - Mathematical proofs in Lean, progressing toward Galois theorem

### Machine Learning & MLOps
- **[distributed-training/](distributed-training/)** - Distributed training systems (data parallelism, model parallelism, multi-node training)
- **[ml-in-production/](ml-in-production/)** - Production ML systems (model serving, monitoring, A/B testing)
- **[mlops/](mlops/)** - MLOps pipelines (experiment tracking, CI/CD, feature stores)
- **[ml-inference/](ml-inference/)** - High-performance inference (optimization, quantization, edge deployment)

### Data Engineering & Analytics (Python)
- **[sas-lineage-tool/](sas-lineage-tool/)** - SAS field lineage parser for tracking data transformations and dependencies

## Philosophy

This repository is dedicated to learning by building things from scratch. Each directory contains:
- **Template files** with TODO comments and implementation guidelines
- **Step-by-step instructions** for gradual implementation
- **Complete solutions** in the `solutions/` folder for reference
- A clear learning path from basics to advanced topics

## Structure

Each project directory contains:

```
project-name/
├── README.md              # Project overview and learning path
├── template-files         # Empty templates with TODOs and guidelines
├── Makefile              # Build configuration (for C projects)
└── solutions/            # Complete working implementations
    ├── README.md         # Solution documentation
    └── solution-files    # Fully implemented code
```

## Getting Started

1. **Choose a project** that interests you
2. **Read the README** in that directory to understand the goals
3. **Start with the template files** - they have TODOs and guidelines
4. **Implement gradually** - follow the TODO comments step by step
5. **Test frequently** - build and test as you implement each section
6. **Check solutions** when stuck or to verify your approach
7. **Learn and iterate** - understand each step before moving forward

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

## Note

These implementations are for educational purposes. They prioritize clarity and understanding over production-ready features or performance. Use the solutions as reference, but try to implement yourself first for maximum learning!