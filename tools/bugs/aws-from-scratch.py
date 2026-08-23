"""Deliberately break the solutions and confirm the checker notices."""
import pathlib, shutil, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2] / "week-01" / "aws-from-scratch"

BUGS = [
    (19, "pricing.py", "tiered_cost prices the whole volume at one rate",
     '''    remaining = float(quantity)
    previous = 0.0
    total = 0.0
    for upper, unit_price in tiers:
        if remaining <= 0:
            break
        span = min(remaining, upper - previous)
        total += span * unit_price
        remaining -= span
        previous = upper
    return total''',
     '''    rate = tiers[-1][1]
    for upper, unit_price in tiers:
        if quantity <= upper:
            rate = unit_price
            break
    return quantity * rate'''),

    (19, "pricing.py", "billable_units rounds instead of ceiling",
     "    return max(minimum_units, math.ceil(quantity / unit_size - 1e-9))",
     "    return max(minimum_units, round(quantity / unit_size))"),

    (20, "pricing.py", "scaled() multiplies the fixed lines too",
     '''            if _is_fixed(item):
                out.items.append(item)
            else:
                out.items.append(item._replace(''',
     '''            if False:
                out.items.append(item)
            else:
                out.items.append(item._replace('''),

    (20, "pricing.py", "money() rounds sub-cent line items to $0.00",
     '''    if amount == 0:
        return "$0"
    if abs(amount) >= 0.01:
        return f"${amount:,.2f}"
    return f"${amount:.6f}"''',
     '''    if amount == 0:
        return "$0"
    return f"${amount:,.2f}"'''),

    (21, "billing.py", "stored_bytes counts only the visible version",
     '''                elif version is current:
                    live += version.size
                else:
                    noncurrent += version.size''',
     '''                elif version is current:
                    live += version.size'''),

    (21, "billing.py", "bill_s3 ignores the minimum billable object size",
     "                kb = max(version.size / 1024.0, min_object_kb)",
     "                kb = version.size / 1024.0"),

    (22, "billing.py", "on-demand reads billed on items RETURNED",
     '''    read_units = table.stats["scanned_items"] * billable_units(
        avg_item_kb, 4.0) * (1.0 if consistent_reads else 0.5)''',
     '''    read_units = table.stats["returned_items"] * billable_units(
        avg_item_kb, 4.0) * (1.0 if consistent_reads else 0.5)'''),

    (22, "billing.py", "GB-seconds uses 1000 MB per GB",
     "    gb = function.memory_mb / 1024.0",
     "    gb = function.memory_mb / 1000.0"),

    (22, "billing.py", "idle polls folded into the work request line",
     '''        bill.add("sqs", "idle-poll-requests", empty_receives, "request",''',
     '''        bill.add("sqs", "requests", empty_receives, "request",'''),

    (22, "billing.py", "cross-AZ billed on one side only",
     '''        bill.add("network", "cross-az-gb", cross_az_gb * 2, "GB",''',
     '''        bill.add("network", "cross-az-gb", cross_az_gb, "GB",'''),

    (23, "optimize.py", "the crossover recalled instead of derived",
     "    return hourly / (3600.0 * on_demand)",
     "    return 0.15"),

    (23, "optimize.py", "the I/O wait scales with memory too",
     "    return io_wait_ms + cpu_work_ms_at_1_vcpu / max(vcpu, 1e-9)",
     "    return (io_wait_ms + cpu_work_ms_at_1_vcpu) / max(vcpu, 1e-9)"),

    (24, "optimize.py", "rank_savings groups by dimension, not service/dimension",
     '''        key = f"{item.service}/{item.dimension}"
        by_line[key] = by_line.get(key, 0.0) + item.cost''',
     '''        key = item.dimension
        by_line[key] = by_line.get(key, 0.0) + item.cost
    by_line.update({f"{i.service}/{i.dimension}": by_line[i.dimension]
                    for i in bill.items})'''),
]

failures = []
for step, filename, label, old, new in BUGS:
    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp) / "s"
        shutil.copytree(ROOT / "solutions", work,
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(ROOT / "check.py", work / "check.py")
        target = work / filename
        text = target.read_text()
        if old not in text:
            failures.append(f"[{step}] {label}: PATCH DID NOT APPLY")
            continue
        target.write_text(text.replace(old, new, 1))
        result = subprocess.run([sys.executable, "check.py", str(step)],
                                cwd=work, capture_output=True, text=True)
        caught = "✗" in result.stdout
        mark = "caught" if caught else "MISSED"
        print(f"  step {step:>2}  {mark:<7} {label}")
        if not caught:
            failures.append(f"[{step}] {label}\n{result.stdout}")

print()
if failures:
    print("PROBLEMS:")
    for f in failures:
        print(" ", f)
    sys.exit(1)
print(f"all {len(BUGS)} injected bugs were caught")
