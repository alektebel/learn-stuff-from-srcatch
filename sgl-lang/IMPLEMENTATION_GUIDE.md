# SGL Implementation Guide

This guide provides step-by-step instructions for implementing a structured generation language (SGL) for LLMs from scratch.

## Overview

You will build a complete SGL system with:
1. **Compiler**: Parse and compile SGL programs
2. **Runtime**: Execute generation with constraints
3. **Constraints**: JSON schema, regex, CFG validation
4. **Optimizations**: Caching, batching, speculative decoding

## Phase 1: Basic Compiler and Runtime

### Step 1: Design the SGL AST

**Goal**: Define abstract syntax tree for SGL programs

**Data structures to implement**:
```python
@dataclass
class ASTNode:
    """Base class for AST nodes"""
    pass

@dataclass
class FunctionDef(ASTNode):
    name: str
    parameters: List[Parameter]
    body: List[Statement]

@dataclass
class GenCall(ASTNode):
    """s.gen("var_name", constraints)"""
    var_name: str
    constraints: Dict[str, Any]
    
@dataclass
class SelectCall(ASTNode):
    """s.select(["choice1", "choice2"])"""
    choices: List[str]
```

**Implementation tasks**:
- [ ] Define AST node types for all SGL constructs
- [ ] Add visitor pattern for AST traversal
- [ ] Implement pretty-printing for debugging
- [ ] Test: Create sample ASTs manually

**Testing**:
```python
# Test AST construction
ast = FunctionDef(
    name="test_func",
    parameters=[Parameter("s", "State")],
    body=[GenCall("name", {"max_tokens": 20})]
)
print(ast)  # Should print readable structure
```

---

### Step 2: Implement Lexer

**Goal**: Tokenize SGL source code

**Tokens to recognize**:
- Keywords: `def`, `return`, `if`, `else`, `for`, `while`
- Operators: `+`, `+=`, `=`, `==`, etc.
- Identifiers: variable names, function names
- Literals: strings, numbers
- Special: `@sgl_function`, `s.gen()`, `s.select()`

**Implementation approach**:
```python
class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.tokens = []
    
    def tokenize(self) -> List[Token]:
        """Tokenize entire source"""
        # TODO: Implement lexical analysis
        pass
    
    def next_token(self) -> Token:
        """Get next token from source"""
        # TODO: Scan next token
        pass
```

**Implementation tasks**:
- [ ] Recognize all SGL keywords
- [ ] Handle string literals with proper escaping
- [ ] Tokenize method calls (s.gen, s.select)
- [ ] Track line/column numbers for errors
- [ ] Test: Tokenize sample SGL programs

**Testing**:
```python
source = '''
@sgl_function
def generate_name(s):
    s += "Name: "
    s += s.gen("name", max_tokens=20)
'''
lexer = Lexer(source)
tokens = lexer.tokenize()
# Verify tokens are correct
```

---

### Step 3: Implement Parser

**Goal**: Build AST from tokens

**Parsing strategy**:
- Recursive descent parser
- Operator precedence for expressions
- Handle SGL-specific syntax

**Implementation approach**:
```python
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0
    
    def parse(self) -> Program:
        """Parse entire program"""
        # TODO: Parse top-level definitions
        pass
    
    def parse_function(self) -> FunctionDef:
        """Parse @sgl_function definition"""
        # TODO: Parse function signature and body
        pass
    
    def parse_statement(self) -> Statement:
        """Parse single statement"""
        # TODO: Handle gen calls, assignments, control flow
        pass
```

**Implementation tasks**:
- [ ] Parse function definitions with decorators
- [ ] Parse gen/select calls with keyword arguments
- [ ] Parse control flow (if/for/while)
- [ ] Handle syntax errors gracefully
- [ ] Test: Parse various SGL programs

**Testing**:
```python
# Test parsing
parser = Parser(tokens)
ast = parser.parse()
assert isinstance(ast.functions[0], FunctionDef)
assert ast.functions[0].name == "generate_name"
```

---

### Step 4: Implement Basic Runtime

**Goal**: Execute SGL programs

**Core classes**:
```python
class GenerationState:
    """Manages generation state"""
    def __init__(self, backend):
        self.backend = backend
        self.prompt = ""
        self.variables = {}
        self.history = []
    
    def __iadd__(self, text: str):
        """Implement s += operation"""
        self.prompt += text
        return self
    
    def gen(self, var_name: str, **kwargs):
        """Generate text with constraints"""
        # TODO: Call backend, apply constraints
        pass
    
    def select(self, choices: List[str]):
        """Select from choices"""
        # TODO: Force generation to match one choice
        pass

class Runtime:
    def __init__(self, backend):
        self.backend = backend
    
    def execute(self, function: FunctionDef, **kwargs):
        """Execute SGL function"""
        state = GenerationState(self.backend)
        # TODO: Execute function body
        return state
```

**Implementation tasks**:
- [ ] Implement GenerationState class
- [ ] Add support for += operator
- [ ] Implement basic gen() without constraints
- [ ] Implement select() operator
- [ ] Test: Execute simple programs

**Testing**:
```python
# Test execution
@sgl_function
def simple_gen(s):
    s += "Hello, "
    s += s.gen("name", max_tokens=10)

runtime = Runtime(mock_backend)
result = runtime.execute(simple_gen)
print(result.prompt)  # Should have "Hello, [generated name]"
```

---

## Phase 2: Constraint Systems

### Step 5: JSON Schema Constraints

**Goal**: Guide generation to produce valid JSON

**Algorithm**:
1. Parse JSON schema
2. At each token, determine valid next tokens
3. Mask logits to only allow valid tokens
4. Continue until complete valid JSON

**Implementation approach**:
```python
class JSONSchemaValidator:
    def __init__(self, schema: dict):
        self.schema = schema
        self.parser_state = JSONParserState()
    
    def get_valid_tokens(self, partial_json: str, vocab) -> Set[int]:
        """Return token IDs that keep JSON valid"""
        # TODO: Parse partial JSON
        # TODO: Determine what tokens are valid next
        # TODO: Return set of valid token IDs
        pass
    
    def is_complete(self, partial_json: str) -> bool:
        """Check if JSON is complete and valid"""
        # TODO: Validate against schema
        pass
```

**Implementation tasks**:
- [ ] Implement JSON schema parser
- [ ] Build incremental JSON parser
- [ ] Implement token-level validation
- [ ] Handle nested objects and arrays
- [ ] Test: Generate JSON matching various schemas

**Testing**:
```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    }
}

validator = JSONSchemaValidator(schema)
result = generate_with_constraint(prompt, validator)
parsed = json.loads(result)  # Should not raise error
assert isinstance(parsed["age"], int)
```

---

### Step 6: Regex Constraints

**Goal**: Generate text matching regex pattern

**Algorithm**:
1. Convert regex to finite state automaton (FSA)
2. Track FSA state during generation
3. At each token, check which transitions are valid
4. Mask logits accordingly

**Implementation approach**:
```python
class RegexConstraint:
    def __init__(self, pattern: str):
        self.pattern = pattern
        self.fsa = self._regex_to_fsa(pattern)
        self.state = self.fsa.start_state
    
    def _regex_to_fsa(self, pattern: str) -> FSA:
        """Convert regex to finite state automaton"""
        # TODO: Use Thompson's construction
        # TODO: Or use existing regex library internals
        pass
    
    def get_valid_tokens(self, vocab) -> Set[int]:
        """Get tokens that keep matching valid"""
        valid_tokens = set()
        for token_id, token_str in vocab.items():
            # Try each token, see if it keeps us in valid state
            if self._can_transition(token_str):
                valid_tokens.add(token_id)
        return valid_tokens
    
    def update_state(self, token: str):
        """Update FSA state with generated token"""
        # TODO: Transition FSA
        pass
```

**Implementation tasks**:
- [ ] Convert regex to FSA (use library or implement)
- [ ] Implement state tracking during generation
- [ ] Handle token-level validation
- [ ] Support common regex features
- [ ] Test: Generate text matching patterns

**Testing**:
```python
# Test regex constraint
pattern = r"\d{3}-\d{3}-\d{4}"  # Phone number
constraint = RegexConstraint(pattern)
result = generate_with_constraint("Phone: ", constraint)
assert re.match(pattern, result)
```

---

### Step 7: Context-Free Grammar

**Goal**: Generate code/structured text with CFG

**Algorithm**:
1. Parse grammar (EBNF format)
2. Build pushdown automaton (PDA)
3. Track PDA state and stack
4. Guide generation with grammar rules

**Implementation approach**:
```python
class CFGConstraint:
    def __init__(self, grammar_str: str):
        self.grammar = self._parse_grammar(grammar_str)
        self.stack = [self.grammar.start_symbol]
    
    def _parse_grammar(self, grammar_str: str) -> Grammar:
        """Parse EBNF grammar"""
        # TODO: Parse production rules
        pass
    
    def get_valid_tokens(self, vocab) -> Set[int]:
        """Get tokens matching current grammar state"""
        # TODO: Check top of stack
        # TODO: Find valid tokens for current non-terminal
        pass
```

**Implementation tasks**:
- [ ] Parse EBNF grammar notation
- [ ] Build grammar data structure
- [ ] Implement PDA for validation
- [ ] Guide generation with grammar
- [ ] Test: Generate code snippets

**Testing**:
```python
grammar = """
expression := term (('+' | '-') term)*
term := factor (('*' | '/') factor)*
factor := NUMBER | '(' expression ')'
"""

constraint = CFGConstraint(grammar)
result = generate_with_constraint("Calculate: ", constraint)
# Result should be valid arithmetic expression
```

---

## Phase 3: Advanced Optimizations

### Step 8: Prefix Caching

**Goal**: Reuse KV cache for common prompts

**Algorithm**:
1. Hash prompt prefixes
2. Store KV cache blocks with ref counting
3. On new request, find longest matching prefix
4. Reuse cached KV, generate only new tokens

**Implementation approach**:
```python
class PrefixCache:
    def __init__(self, max_size: int):
        self.cache = {}  # prefix_hash -> CacheEntry
        self.lru = LRUCache(max_size)
    
    def lookup(self, prompt: str) -> Optional[CacheEntry]:
        """Find longest matching prefix"""
        # TODO: Find best match from cache
        pass
    
    def insert(self, prompt: str, kv_cache):
        """Cache KV for this prompt"""
        # TODO: Hash prompt
        # TODO: Store KV cache
        # TODO: Update LRU
        pass
```

**Implementation tasks**:
- [ ] Implement prefix matching algorithm
- [ ] Add KV cache storage
- [ ] Implement cache eviction (LRU/LFU)
- [ ] Track cache hit rate
- [ ] Test: Measure speedup on repeated prefixes

**Benchmarking**:
```python
# Measure cache impact
prompts = ["Generate a story about " + topic for topic in topics]
times_no_cache = benchmark(prompts, use_cache=False)
times_with_cache = benchmark(prompts, use_cache=True)
speedup = times_no_cache / times_with_cache
print(f"Speedup: {speedup:.2f}x")
```

---

### Step 9: Continuous Batching

**Goal**: Batch requests dynamically

**Algorithm**:
1. Maintain queue of pending requests
2. Batch requests of similar state
3. Handle variable sequence lengths
4. Remove completed sequences from batch

**Implementation approach**:
```python
class ContinuousBatcher:
    def __init__(self, max_batch_size: int):
        self.running = []
        self.waiting = []
        self.max_batch_size = max_batch_size
    
    def add_request(self, request: Request):
        """Add new request to queue"""
        self.waiting.append(request)
    
    def form_batch(self) -> Batch:
        """Create batch for next iteration"""
        # TODO: Add waiting requests if space
        # TODO: Remove completed from running
        # TODO: Return batch
        pass
    
    def step(self, batch: Batch):
        """Execute one generation step"""
        # TODO: Generate next token for each sequence
        # TODO: Update sequence states
        pass
```

**Implementation tasks**:
- [ ] Implement request queue
- [ ] Add dynamic batch formation
- [ ] Handle variable lengths with padding
- [ ] Remove completed sequences
- [ ] Test: Measure throughput improvement

**Benchmarking**:
```python
# Compare throughput
throughput_single = measure_throughput(batch_size=1)
throughput_batched = measure_throughput(batch_size=32)
improvement = throughput_batched / throughput_single
print(f"Throughput improvement: {improvement:.2f}x")
```

---

### Step 10: Speculative Decoding

**Goal**: Speed up with draft model

**Algorithm**:
1. Generate K tokens with small draft model
2. Verify all K tokens with target model in parallel
3. Accept matching prefix
4. Discard from first mismatch

**Implementation approach**:
```python
class SpeculativeDecoder:
    def __init__(self, draft_model, target_model):
        self.draft_model = draft_model
        self.target_model = target_model
        self.k = 5  # speculation depth
    
    def generate_step(self, prompt: str) -> List[str]:
        """Generate with speculation"""
        # 1. Draft model generates K tokens
        draft_tokens = self.draft_model.generate(
            prompt, 
            max_tokens=self.k
        )
        
        # 2. Target model verifies all at once
        verified = self.target_model.verify_batch(
            prompt,
            draft_tokens
        )
        
        # 3. Find first mismatch
        accepted = []
        for draft, target in zip(draft_tokens, verified):
            if draft == target:
                accepted.append(draft)
            else:
                break
        
        return accepted
```

**Implementation tasks**:
- [ ] Integrate draft model
- [ ] Implement parallel verification
- [ ] Add tree attention for multiple candidates
- [ ] Tune speculation depth
- [ ] Test: Measure wall-clock speedup

**Benchmarking**:
```python
# Measure speculative decoding speedup
time_standard = benchmark_generation(use_speculation=False)
time_speculative = benchmark_generation(use_speculation=True)
speedup = time_standard / time_speculative
print(f"Speedup: {speedup:.2f}x")
```

---

## Phase 4: Production Features

### Step 11: Backend Integration

**Goal**: Support multiple LLM backends

**Implementation approach**:
```python
class BackendInterface:
    """Abstract backend interface"""
    
    def generate(self, prompt: str, max_tokens: int, **kwargs) -> str:
        """Generate completion"""
        raise NotImplementedError
    
    def get_logits(self, prompt: str) -> np.ndarray:
        """Get logits for next token"""
        raise NotImplementedError

class OpenAIBackend(BackendInterface):
    def __init__(self, api_key: str):
        self.client = openai.Client(api_key=api_key)
    
    def generate(self, prompt: str, max_tokens: int, **kwargs) -> str:
        response = self.client.completions.create(
            model="gpt-3.5-turbo",
            prompt=prompt,
            max_tokens=max_tokens
        )
        return response.choices[0].text

class HuggingFaceBackend(BackendInterface):
    def __init__(self, model_name: str):
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
    def generate(self, prompt: str, max_tokens: int, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0])
```

**Implementation tasks**:
- [ ] Define backend interface
- [ ] Implement OpenAI adapter
- [ ] Implement HuggingFace adapter
- [ ] Add vLLM integration
- [ ] Test: Run same program on all backends

---

### Step 12: Error Handling

**Goal**: Robust error handling and recovery

**Implementation tasks**:
- [ ] Define error types (ConstraintViolation, BackendError, etc.)
- [ ] Add retry logic with exponential backoff
- [ ] Implement fallback strategies
- [ ] Handle partial results
- [ ] Add timeout management
- [ ] Test: Simulate various failure modes

**Error types to handle**:
```python
class ConstraintViolationError(Exception):
    """Constraint cannot be satisfied"""
    pass

class BackendError(Exception):
    """Backend API error"""
    pass

class TimeoutError(Exception):
    """Generation timeout"""
    pass
```

---

### Step 13: Monitoring and Debugging

**Goal**: Add observability

**Metrics to collect**:
- Generation latency (p50, p90, p99)
- Token generation rate
- Constraint satisfaction rate
- Cache hit rate
- Backend API latency

**Implementation approach**:
```python
class Metrics:
    def __init__(self):
        self.latencies = []
        self.cache_hits = 0
        self.cache_misses = 0
    
    def record_generation(self, latency_ms: float):
        self.latencies.append(latency_ms)
    
    def record_cache_hit(self):
        self.cache_hits += 1
    
    def get_stats(self) -> Dict:
        return {
            "p50_latency": np.percentile(self.latencies, 50),
            "p99_latency": np.percentile(self.latencies, 99),
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses)
        }
```

**Implementation tasks**:
- [ ] Add metrics collection
- [ ] Implement trace logging
- [ ] Create debug visualization
- [ ] Build performance dashboard
- [ ] Test: Collect and analyze metrics

---

## Testing Strategy

### Unit Tests

Test each component independently:
```bash
pytest tests/test_compiler.py     # Test parsing
pytest tests/test_runtime.py      # Test execution
pytest tests/test_constraints.py  # Test validation
```

### Integration Tests

Test complete workflows:
```python
def test_json_generation():
    """Test JSON schema constraint"""
    @sgl_function
    def gen_user(s):
        s += s.gen("user", json_schema=user_schema)
    
    result = runtime.execute(gen_user)
    user = json.loads(result.variables["user"])
    assert validate_schema(user, user_schema)

def test_regex_generation():
    """Test regex constraint"""
    @sgl_function
    def gen_phone(s):
        s += s.gen("phone", regex=r"\d{3}-\d{3}-\d{4}")
    
    result = runtime.execute(gen_phone)
    assert re.match(r"\d{3}-\d{3}-\d{4}", result.variables["phone"])
```

### Benchmarks

Measure performance:
```python
def benchmark_generation_speed():
    """Measure tokens/second"""
    start = time.time()
    for _ in range(100):
        runtime.execute(test_function)
    duration = time.time() - start
    tokens_per_sec = total_tokens / duration
    print(f"Throughput: {tokens_per_sec:.1f} tokens/sec")

def benchmark_cache_efficiency():
    """Measure cache hit rate"""
    runtime.clear_cache()
    for prompt in test_prompts:
        runtime.execute(prompt)
    stats = runtime.get_cache_stats()
    print(f"Cache hit rate: {stats['hit_rate']:.1%}")
```

---

## Common Issues and Solutions

### Issue 1: Constraint too restrictive
**Problem**: No valid tokens available
**Solution**: Relax constraint or backtrack

### Issue 2: Poor cache hit rate
**Problem**: Prompts not sharing prefixes
**Solution**: Canonicalize prompts, adjust hash function

### Issue 3: Slow token generation
**Problem**: Constraint checking overhead
**Solution**: Cache valid token sets, optimize FSA

### Issue 4: Memory leaks
**Problem**: KV cache not released
**Solution**: Proper reference counting, periodic cleanup

---

## Performance Optimization Tips

1. **Precompile constraints**: Parse schemas/regex once
2. **Cache valid tokens**: Don't recompute at each step
3. **Use efficient data structures**: Sets for token masks
4. **Batch validation**: Check multiple candidates at once
5. **Profile hotspots**: Use cProfile to find bottlenecks

---

## Next Steps

After completing the implementation:
1. Add more constraint types
2. Optimize for production use
3. Build more backends
4. Create example applications
5. Write comprehensive documentation

## Resources

- [Compiler Design Techniques](https://en.wikipedia.org/wiki/Compiling_techniques)
- [Regex to FSA Conversion](https://en.wikipedia.org/wiki/Thompson%27s_construction)
- [JSON Schema Specification](https://json-schema.org/)
- [LLM Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
