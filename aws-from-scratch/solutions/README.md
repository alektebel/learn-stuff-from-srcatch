# AWS From Scratch — Solutions

Complete implementations of every template in the parent directory. Pure Python 3
standard library; everything runs in about a second.

```bash
python3 iam.py  s3.py  sqs.py  dynamodb.py  lambda_svc.py  sns.py  kms.py  vpc.py
python3 capstone.py
```

## What each file demonstrates

### `iam.py`
The three-line rule, with the property that matters:

```
admin alone                           ALLOW
admin + guardrail                     DENY  explicit Deny in Guardrail (NoDeletes)
guardrail + admin (order swapped)     DENY  explicit Deny in Guardrail (NoDeletes)
```

Order does not change the answer, because statements are partitioned by effect rather
than walked in sequence. Conditions AND across keys and OR within one, and a **missing
context key never matches** — conditions fail closed.

STS checks **both** directions: an outsider whose own policy allows `sts:AssumeRole` is
still refused by the role's trust policy.

### `s3.py`
Flat keys with delimiter-invented folders; multipart ETags ending `-3`; and versioning:

```
delete -> pushed marker v000003
get   -> NoSuchKey: doc.txt (a delete marker is on top)
versions still stored: 3 (22 bytes, all billable)
versioned get -> b'version one'
```

Plus abandoned multipart uploads, which nothing lists by default and everything bills.

### `sqs.py`
The lease, and the duplicate it creates:

```
t=0   consumer A receives 'expensive-job' (receive_count=1)
t=11  consumer B receives ['expensive-job'] — the lease expired
t=12  A finishes and deletes with its stale handle: False
```

**Note the implementation detail that makes this work:** `receive()` returns a `copy` of
the message so each consumer holds its own receipt handle. Sharing one object lets a
redelivery rewrite the earlier consumer's handle, and the stale delete wrongly succeeds —
that was a real bug here, caught by the checker.

`oldest_age()` counts in-flight messages too; a message being retried forever is exactly
what the age alarm exists to catch.

### `dynamodb.py`
```
table capacity: 40 WCU, spread over 4 partitions = 10 per partition
wrote 10 items, then: partition 'p0': 11.0 write units requested, 10.0 available

query('alice'): 3 items, read 3
scan(filter):   3 items, read 12
```

Two implementation notes that were both bugs first:

- **Partitions are keyed by the FULL composite key.** Keying by sort key alone made
  `alice/000` and `bob/000` collide — silent data loss rather than an error.
- **`_partition` uses `hashlib.md5`, not `hash()`.** Python randomises string hashing per
  process, so `hash()` would place a key in a different partition every run and make every
  capacity result irreproducible.

### `lambda_svc.py`
```
invoke 0: <COLD 850ms env=1>  {'calls_in_this_env': 1}
invoke 1: <warm  50ms env=1>  {'calls_in_this_env': 2}

reserved 6 for 'critical'
unreserved pool left for everything else: 4
10 events to 'api' -> 4 ran, 6 throttled      <- 'api' was never changed
```

Async invocation retries three times and then dead-letters; sync surfaces the error
immediately. Same bug, different blast radius.

### `sns.py`
```
message               pager  email  firehose
prod critical           yes    yes       yes
staging critical          -    yes       yes
no env attribute          -    yes       yes
```

`publish()` threads `now` through to SQS subscribers. Without it a topic on wall-clock
time enqueues into a simulated-clock queue "in the future", and the messages silently
never become visible — which is exactly what happened before this was fixed.

EventBridge matches nested shapes, and an event matching two rules fires both with no
deduplication.

### `kms.py`
```
plaintext:       2900 bytes
sent to KMS:        0 bytes of your data
round-trips:        1 per object, regardless of size

rotated -> current version is now 2
old ciphertext still decrypts: b'written before rotation'
bytes of data re-encrypted by the rotation: 0
```

Then the limit case: delete version 1 and that ciphertext is gone permanently.

### `vpc.py`
Security group vs NACL with identical rules, side by side — the SG connection works, the
NACL connection fails on the reply. `test_connection` models the whole connection
precisely so this is visible.

### `capstone.py`
The pipeline, and the diagnosis that matters:

```
drain: {'processed': 5, 'throttled': 0, 'failed': 5}
```

Lambda errored; DynamoDB caused it. The failed messages were never deleted, so their
leases expired and they came back — nothing was lost, because the queue sits between the
topic and the function.

## Implementation notes

- **`iam.evaluate` returns on the first matching Deny** and collects Allows otherwise.
  Never iterate letting the last match win.
- **A permissions boundary never grants**; `evaluate_with_boundary` requires both.
- **`compute_etag` takes an optional `parts` list** — that is what produces the `-N` form.
- **`Queue.receive` returns copies**; see above.
- **`Table.consume` lives on the Partition**, not the Table. Capacity per partition is the
  entire reason hot keys throttle.
- **`Function` environments have a TTL**, so warm state is neither reliable nor reliably
  absent.
- **`matches_filter` treats a missing attribute as no-match**, matching IAM's fail-closed
  rule.
- **`generate_data_key` binds the encryption context into the wrap**, so decrypting
  requires supplying it exactly.
- **`test_connection` evaluates the reply on the ephemeral port**, which is what makes the
  stateless NACL fail where a stateful SG succeeds.
