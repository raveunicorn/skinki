# Stage <N> — <Title> (SPEC)

> Copy this file to `STAGE_<N>.md` and fill every section. A spec is "done
> enough to delegate" when a capable model could implement it without asking
> clarifying questions, and a CI gate can decide pass/fail automatically.

- **Status:** draft | ready-to-build | in-progress | done
- **Owner of the design (frontier/human):** <who decided the approach>
- **Delegatable to (cheaper model):** yes/no — <which tickets>

## 1. Hypothesis

One paragraph: what we believe is true and are testing. State it so it can be
proven false. (e.g. "5M units are searchable within budget via X, recall >= 95%
vs full precision.")

## 2. Budgets / fitness function (the gate)

Concrete, machine-checkable numbers. These become tests and/or a `--assert-gate`
check. No vibes.

| Metric | Budget | How measured |
| --- | --- | --- |
| <recall@k> | >= 0.95 | vs exact baseline on synthetic corpus |
| <RAM @5M> | <= 250 MB | projected from bytes/vector |
| <p95 latency> | <= 150 ms | telemetry over query set |

## 3. Public interface

The exact Rust trait(s)/signature(s) the implementation must satisfy. This is
the contract; everything behind it is the implementer's freedom.

```rust
// pub trait Foo { fn bar(&self, ...) -> ...; }
```

## 4. Invariants (must always hold)

- Determinism: same seed -> identical output.
- Provenance preserved: <...>
- No `unsafe` outside <quarantined module>.
- <stage-specific invariants>

## 5. Test plan

- **Unit:** <key behaviors>
- **Property:** <invariants to fuzz, e.g. "round-trip", "monotonicity">
- **Golden:** <fixed inputs with locked expected outputs>
- **Gate command:** the exact CLI/CI invocation that returns non-zero on failure.

## 6. Task decomposition

Split the work so the subtle parts are isolated from the mechanical parts.

| Ticket | Type | Assignee tier | Acceptance |
| --- | --- | --- | --- |
| T1 design: <algorithm/format choice> | design | frontier/human | spec + benchmark rationale |
| T2 impl: <wire up / plumbing> | impl | cheaper model | gate green, tests pass |
| T3 impl: <...> | impl | cheaper model | <test> |

## 7. Definition of done

- [ ] Gate green (`--assert-gate` / tests) in CI.
- [ ] `cargo test`, `clippy -D warnings`, `cargo fmt --check` clean.
- [ ] Docs updated per spec (README/ROADMAP rows, this spec's Status -> done).
- [ ] Decision recorded: did we beat existing tools, or invent? Why.

## 8. Out of scope

What this stage explicitly does NOT do (defer to which later stage).
