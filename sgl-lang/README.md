# SGL (Structured Generation Language) from Scratch

A from-scratch implementation of a structured generation language for LLMs. Learn how modern LLM frameworks like SGLang manage constrained text generation, grammar-based outputs, and efficient compilation.

## Goal

Build a deep understanding of LLM generation frameworks by implementing:
- **Frontend compiler**: Parse structured generation programs and compile to execution plans
- **Runtime engine**: Execute generation programs with constraints
- **Grammar-based generation**: Enforce JSON schemas, regex patterns, and custom grammars
- **Optimization**: Prefix caching, KV-cache reuse, speculative decoding
- **Integration**: Connect with various LLM backends (OpenAI, HuggingFace, vLLM)

## What is SGL?

SGL (Structured Generation Language) is a domain-specific language for programming LLM generation with constraints. It allows you to:
- Define structured outputs (JSON, XML, code)
- Enforce grammar constraints during generation
- Compose complex multi-step LLM workflows
- Optimize generation with caching and batching

Example SGL program:
```python
@sgl_function
def generate_character(s, name):
    s += f"Generate a character profile for {name}:\n"
    s += "Name: " + s.gen("name", max_tokens=20)
    s += "\nAge: " + s.gen("age", regex=r"\d{1,2}")
    s += "\nPersonality: " + s.gen("personality", 
                                    json_schema=personality_schema)
```

## Project Structure

```
sgl-lang/
├── README.md                          # This file
├── IMPLEMENTATION_GUIDE.md            # Detailed implementation guide
├── core/
│   ├── compiler.py                   # SGL program compiler
│   ├── runtime.py                    # Execution engine
│   ├── grammar.py                    # Grammar constraint parser
│   └── optimizer.py                  # Optimization passes
├── backends/
│   ├── base.py                       # Backend interface
│   ├── openai_backend.py            # OpenAI API backend
│   ├── huggingface_backend.py       # HuggingFace transformers
│   └── vllm_backend.py              # vLLM integration
├── constraints/
│   ├── json_schema.py               # JSON schema constraints
│   ├── regex.py                     # Regex-based generation
│   ├── cfg.py                       # Context-free grammar
│   └── custom.py                    # Custom constraint system
├── optimizations/
│   ├── prefix_cache.py              # Prefix caching system
│   ├── kv_reuse.py                  # KV-cache reuse
│   ├── batching.py                  # Request batching
│   └── speculative.py               # Speculative decoding
├── tests/
│   ├── test_compiler.py
│   ├── test_runtime.py
│   ├── test_constraints.py
│   └── benchmarks/
│       ├── generation_speed.py
│       ├── constraint_accuracy.py
│       └── cache_efficiency.py
└── solutions/                        # Complete reference implementations
    ├── README.md
    └── ...
```

## Learning Path

### Phase 1: Basic Compiler and Runtime (8-10 hours)

**1.1 Language Design and Parser**

Understand the SGL language structure:
- Function decorators (`@sgl_function`)
- Generation operators (`gen()`, `select()`)
- Variable binding and state management
- Control flow (loops, conditions)

**Implementation tasks**:
- Design AST (Abstract Syntax Tree) for SGL programs
- Implement lexer for SGL syntax
- Build parser to create AST from source
- Add type checking for SGL constructs

**2.2 Basic Runtime Engine**

Build the execution engine:
- State management (tracking prompts, variables)
- Operator execution (gen, select, etc.)
- Backend integration interface
- Token streaming support

**Implementation tasks**:
- Create `GenerationState` class
- Implement `gen()` operator
- Implement `select()` for choices
- Add simple backend adapter

**Skills learned**:
- DSL design and implementation
- Compiler frontend architecture
- Runtime execution models
- State machines for generation

---

### Phase 2: Constraint Systems (10-12 hours)

**2.1 JSON Schema Constraints**

Implement JSON schema validation during generation:
- Parse JSON schemas
- Guide generation token-by-token
- Validate partial JSON structures
- Handle nested objects and arrays

**Implementation tasks**:
- Implement JSON schema parser
- Build token-level validation
- Add schema-guided sampling
- Test with complex schemas

**2.2 Regex Constraints**

Build regex-based generation:
- Parse regular expressions
- Convert to finite state automaton (FSA)
- Guide generation with FSA
- Optimize for common patterns

**Implementation tasks**:
- Implement regex to FSA conversion
- Build FSA-guided sampling
- Add backtracking for invalid paths
- Optimize for performance

**2.3 Context-Free Grammars**

Implement CFG-based generation:
- Parse BNF/EBNF grammars
- Build pushdown automaton
- Guide generation with grammar rules
- Handle ambiguous grammars

**Implementation tasks**:
- Implement grammar parser (EBNF)
- Build PDA for CFG validation
- Add grammar-guided sampling
- Test with code generation grammars

**Skills learned**:
- Formal language theory application
- Finite automata and pushdown automata
- Constrained sampling techniques
- Real-time validation systems

---

### Phase 3: Advanced Optimizations (12-15 hours)

**3.1 Prefix Caching**

Implement intelligent prompt caching:
- Identify common prefixes across requests
- Cache KV values for reuse
- Implement cache eviction policies (LRU, LFU)
- Measure cache hit rates

**Implementation tasks**:
- Design cache data structure
- Implement prefix matching algorithm
- Add cache management (insertion, eviction)
- Optimize for memory efficiency

**3.2 KV-Cache Reuse**

Advanced KV-cache management:
- Track which KV pairs are reusable
- Implement efficient cache indexing
- Handle dynamic batching with cache
- Optimize memory layout

**Implementation tasks**:
- Design KV-cache data structure
- Implement cache key generation
- Add reference counting for sharing
- Measure memory savings

**3.3 Continuous Batching**

Implement dynamic request batching:
- Batch requests with different states
- Handle variable-length sequences
- Dynamic batch size adjustment
- Minimize padding overhead

**Implementation tasks**:
- Design batching scheduler
- Implement request queue management
- Add dynamic batch formation
- Optimize for throughput

**3.4 Speculative Decoding**

Add speculative decoding support:
- Draft model integration
- Verification with target model
- Tree-based speculation
- Adaptive speculation strategies

**Implementation tasks**:
- Implement draft model runner
- Build verification logic
- Add tree attention for speculation
- Benchmark speedup

**Skills learned**:
- LLM inference optimization
- Cache systems design
- Batch processing strategies
- Speculative execution

---

### Phase 4: Production Features (8-10 hours)

**4.1 Multiple Backend Support**

Integrate with various LLM backends:
- OpenAI API
- HuggingFace Transformers
- vLLM engine
- Local models (llama.cpp)

**Implementation tasks**:
- Define unified backend interface
- Implement OpenAI adapter
- Implement HuggingFace adapter
- Add vLLM integration
- Handle API differences gracefully

**4.2 Error Handling and Recovery**

Build robust error handling:
- Constraint violation recovery
- Backend failure handling
- Timeout management
- Partial result recovery

**Implementation tasks**:
- Add comprehensive error types
- Implement retry logic
- Add fallback strategies
- Test edge cases

**4.3 Monitoring and Debugging**

Add observability:
- Generation traces
- Performance metrics
- Token usage tracking
- Constraint validation logs

**Implementation tasks**:
- Implement trace collection
- Add metrics aggregation
- Build debug visualization
- Create performance dashboard

**Skills learned**:
- Multi-backend architecture
- Production error handling
- System observability
- Performance monitoring

---

**Total Time**: ~40-50 hours for complete implementation

## Features to Implement

### Core Language Features
- [x] Function definitions with `@sgl_function`
- [x] Generation operators: `gen()`, `select()`, `choose()`
- [x] Variable binding and state
- [x] Control flow (if/else, loops)
- [x] Function composition

### Constraint Types
- [x] JSON schema validation
- [x] Regex patterns
- [x] Context-free grammars
- [x] Custom validators
- [x] Type constraints

### Optimizations
- [x] Prefix caching
- [x] KV-cache reuse
- [x] Continuous batching
- [x] Speculative decoding
- [x] Memory optimization

### Backend Integration
- [x] OpenAI API
- [x] HuggingFace Transformers
- [x] vLLM
- [x] Local models

## Testing Your Implementation

### Unit Tests

Test individual components:
```bash
# Test compiler
python -m pytest tests/test_compiler.py -v

# Test runtime
python -m pytest tests/test_runtime.py -v

# Test constraints
python -m pytest tests/test_constraints.py -v

# Test optimizations
python -m pytest tests/test_optimizations.py -v
```

### Integration Tests

Test complete workflows:
```bash
# Test JSON generation
python tests/integration/test_json_generation.py

# Test code generation
python tests/integration/test_code_generation.py

# Test multi-step workflows
python tests/integration/test_workflows.py
```

### Benchmarks

Measure performance:
```bash
# Generation speed
python tests/benchmarks/generation_speed.py

# Cache efficiency
python tests/benchmarks/cache_efficiency.py

# Constraint accuracy
python tests/benchmarks/constraint_accuracy.py

# Throughput with batching
python tests/benchmarks/batching_throughput.py
```

## Example Programs to Test

### 1. JSON Generation
```python
@sgl_function
def generate_user_profile(s):
    s += "Generate a user profile:\n"
    s += s.gen("profile", json_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0, "maximum": 120},
            "email": {"type": "string", "format": "email"}
        }
    })
```

### 2. Code Generation
```python
@sgl_function
def generate_python_function(s, function_name):
    s += f"def {function_name}("
    s += s.gen("params", regex=r"[a-z_][a-z0-9_]*(, [a-z_][a-z0-9_]*)*")
    s += "):\n"
    s += s.gen("body", grammar=python_grammar, max_tokens=200)
```

### 3. Multi-Step Reasoning
```python
@sgl_function
def solve_math_problem(s, problem):
    s += f"Problem: {problem}\n"
    s += "Let's solve this step by step:\n"
    
    for i in range(3):
        s += f"Step {i+1}: "
        s += s.gen(f"step_{i}", stop="\n", max_tokens=50)
        s += "\n"
    
    s += "Final answer: "
    s += s.gen("answer", regex=r"-?\d+(\.\d+)?")
```

## Performance Goals

Your implementation should achieve:

### Correctness
- ✅ 100% constraint satisfaction (no invalid outputs)
- ✅ Proper grammar enforcement
- ✅ Accurate JSON schema validation
- ✅ Correct regex matching

### Performance
- ✅ <10ms compiler overhead per program
- ✅ <1ms runtime overhead per token
- ✅ >80% cache hit rate on repeated prefixes
- ✅ >2x throughput improvement with batching
- ✅ >1.5x speedup with speculative decoding

### Scalability
- ✅ Handle 100+ concurrent requests
- ✅ Support constraints with 1000+ rules
- ✅ Manage 10GB+ KV-cache efficiently
- ✅ Process 1000+ tokens/sec with batching

## Resources

### Papers
- **SGLang**: "SGLang: Structured Generation Language for LLMs" (2024)
- **Constrained Decoding**: "Fast Structured Decoding via Tokenization" (2023)
- **Grammar Constraints**: "Grammar-based Decoding for Language Models" (2022)
- **Speculative Decoding**: "Fast Inference from Transformers via Speculative Decoding" (2023)

### Related Projects
- [SGLang](https://github.com/sgl-project/sglang) - Original SGLang implementation
- [Outlines](https://github.com/outlines-dev/outlines) - Structured text generation
- [Guidance](https://github.com/guidance-ai/guidance) - Programming LLMs
- [LMQL](https://lmql.ai/) - Language Model Query Language

### Documentation
- [JSON Schema Specification](https://json-schema.org/)
- [Regular Expression Syntax](https://en.wikipedia.org/wiki/Regular_expression)
- [Context-Free Grammars](https://en.wikipedia.org/wiki/Context-free_grammar)
- [LLM Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)

### Video Courses
- [Generative AI and LLMs](https://github.com/Developer-Y/cs-video-courses#generative-ai-and-llms)
- [Natural Language Processing](https://github.com/Developer-Y/cs-video-courses#natural-language-processing)
- [Theoretical CS and Programming Languages](https://github.com/Developer-Y/cs-video-courses#theoretical-cs-and-programming-languages)
- [Deep Learning](https://github.com/Developer-Y/cs-video-courses#deep-learning)

## Common Pitfalls

1. **Inefficient constraint checking**: Check constraints token-by-token, not after generation
2. **Poor cache design**: Use content-addressable caching, not positional
3. **Ignoring edge cases**: Test with malformed schemas, invalid regex, ambiguous grammars
4. **Memory leaks**: Properly clean up KV-caches and intermediate states
5. **Blocking on I/O**: Use async/await for backend calls
6. **Over-optimization**: Profile before optimizing

## Debug Tips

### Compiler Issues
```python
# Enable verbose compilation
compiler.compile(program, verbose=True)

# Print AST
print(compiler.get_ast(program))

# Check type inference
print(compiler.get_type_info(program))
```

### Runtime Issues
```python
# Enable trace logging
runtime.execute(program, trace=True)

# Inspect state at each step
runtime.execute(program, breakpoint=lambda s: print(s))

# Validate constraints
runtime.execute(program, validate_constraints=True)
```

### Performance Issues
```python
# Profile execution
import cProfile
cProfile.run('runtime.execute(program)')

# Measure cache hit rate
stats = runtime.get_cache_stats()
print(f"Hit rate: {stats.hit_rate:.2%}")

# Trace token generation
runtime.execute(program, profile_tokens=True)
```

## Advanced Topics

After completing the core implementation, explore:

### Advanced Optimizations
- Parallel sampling for multiple outputs
- Continuous batching with priorities
- Multi-level caching (memory + disk)
- Tensor parallelism for large models

### Language Extensions
- Loops and conditionals in generation
- Function composition and reuse
- Dynamic constraint modification
- Streaming partial results

### Integration
- REST API server
- WebSocket streaming
- gRPC for high performance
- Message queue integration

## Contributing

This is a learning project focused on understanding LLM generation frameworks. Areas for exploration:
- More constraint types (TypeScript types, SQL schemas)
- Better optimization strategies
- Novel batching algorithms
- Integration with more backends

## License

Educational purposes. Use freely for learning.

## Acknowledgments

Inspired by:
- SGLang and its structured generation approach
- Outlines for grammar-based generation
- Guidance for LLM programming paradigms
- vLLM for efficient serving techniques
