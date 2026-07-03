# Stage 6C — Distribution: releases, one-line install, versioning (SPEC)

> Product backlog, batch 2026-07-B. Distribution is cheaper than features:
> today the barrier to trying skinki is `git clone && cargo build` (a Rust
> toolchain), which filters out ~everyone who is merely curious. This stage
> makes "agent has memory in 2 minutes" true: prebuilt binaries, a one-line
> install, a copy-paste MCP config, and the release hygiene (semver +
> CHANGELOG) that makes depending on skinki safe.

- **Status:** ready to build (any time; no code dependencies on other stages)
- **Owner of the design (frontier/human):** frontier — the release/versioning
  policy below is locked; the workflow YAML is mechanical.
- **Delegatable to (cheaper model):** **yes, everything** except D1 (catalog
  submissions + the privacy doc sign-off — human).

> Read [`../AGENTS.md`](../AGENTS.md). Nothing here touches engine code paths;
> CI additions must not slow the existing test jobs (release workflow runs on
> tags only).

## 1. Hypothesis

A tagged release producing signed, checksummed binaries for the 3 mainstream
targets, plus an install script and a documented MCP snippet, reduces
time-to-first-search for a new user from ~20 minutes (toolchain + build) to
**< 2 minutes** — measured by the smoke test executing the full path
(download → install → `skinki-mcp` handshake → search) in CI on every release.

## 2. Budgets / fitness function (the gate)

| Metric | Budget | How measured |
| --- | --- | --- |
| Release artifacts | `skinki`, `skinki-mcp`, `libskinki_ffi` + `skinki.h` for `aarch64-apple-darwin`, `x86_64-apple-darwin`, `x86_64-unknown-linux-gnu` | release workflow output on a tag |
| Integrity | `SHA256SUMS` published; every artifact listed | workflow step |
| **Install smoke test** | `install.sh` on a clean runner: `skinki --version` OK, `skinki-mcp` answers `initialize` + `tools/list` over stdio | CI job on every tag (and a dry-run on `main` weekly) |
| Time-to-first-search | install → demo corpus → first `search` reply ≤ 120 s on the CI runner | timed smoke step |
| Version honesty | tag == `sk_version()` == `Cargo.toml` workspace version == top CHANGELOG entry | release workflow asserts, fails the release otherwise |
| Repro | `--version` prints version + git sha + target triple | unit test |

## 3. Public interface

```
scripts/install.sh        # detects OS/arch, downloads the matching release
                          # tarball from GitHub releases, verifies SHA256,
                          # installs to ~/.local/bin (or $SKINKI_INSTALL_DIR),
                          # prints the MCP config snippet. POSIX sh, no deps.
.github/workflows/release.yml   # on: push tags 'v*'
CHANGELOG.md              # keep-a-changelog format; Unreleased section required
PRIVACY.md                # the zero-telemetry stance, stated as a testable
                          # claim: "0 network bytes at runtime; the only
                          # network call this project ever makes is YOUR
                          # download of it" + how to verify (offline run).
docs/INSTALL.md           # binaries, install.sh, cargo install, MCP snippets
                          # (Claude Code, Cursor), C-ABI quickstart
```

Versioning policy, locked:

- **SemVer over the observable surface:** the C-ABI (`skinki.h`), the MCP tool
  schemas, the CLI subcommand contract, and **all on-disk formats** (a format
  change that old binaries cannot read = major bump; read-old-write-new =
  minor). Internal Rust APIs are not covered (workspace-internal).
- Every PR that changes an observable surface must touch `CHANGELOG.md`
  (`Unreleased`) — enforced by a CI check on changed paths.

## 4. Invariants (must always hold)

- Release binaries are built by CI from the tag, never uploaded by hand.
- `install.sh` verifies checksums before installing; refuses on mismatch.
- No telemetry, no update-checks, no network at runtime — `PRIVACY.md` is a
  contract, and the smoke test runs the binary with network disabled
  (`--network none` container / `unshare -n` where available) to prove it.
- Existing CI jobs unchanged and green.

## 5. Test plan

- **Unit:** version-string format; CHANGELOG-touched CI check on a synthetic
  diff.
- **Smoke (the gate):** clean container → `install.sh` → `skinki --version` →
  `skinki demo --seed 42 --years 1` → `skinki-mcp --corpus <demo dump>` fed
  `initialize` + `tools/list` + one `search` over stdio → assert well-formed
  replies; total wall time ≤ 120 s; network disabled after the download step.
- **Gate command:** the `release` workflow itself (a tag on a test branch
  exercises it end to end before the first real tag).

## 6. Task decomposition

| Ticket | Type | Tier | Acceptance |
| --- | --- | --- | --- |
| T1 `release.yml`: matrix build, strip, tar, SHA256SUMS, GitHub release; version-consistency assert | impl | cheaper | test-tag produces all artifacts |
| T2 `install.sh` + `docs/INSTALL.md` + MCP snippets | impl | cheaper | smoke test green |
| T3 smoke-test job (container, no-network run, timing) | impl | cheaper | §5 smoke green on the test tag |
| T4 `CHANGELOG.md` (backfill from git history at coarse grain) + CI touched-check; `--version` with git sha | impl | cheaper | checks green |
| T5 `PRIVACY.md` + no-network smoke assertion | impl | cheaper (human reviews wording) | doc merged; assertion green |
| **D1** submit to MCP catalogs / awesome-lists; decide the release cadence | human | human | listings live; cadence written in CONTRIBUTING |

## 7. Definition of done

- [ ] A real `v0.x.0` release exists with all artifacts + checksums; smoke
      green on the tag.
- [ ] README quickstart shows the binary path first, `cargo build` second.
- [ ] `cargo test`, clippy, fmt clean (untouched, but asserted).
- [ ] Decision recorded: cadence, and which catalogs listed skinki.

## 8. Out of scope

- Code signing / notarization (`.dmg` land — Stage 7).
- Homebrew/apt packaging (revisit after real demand; install.sh first).
- Windows targets (revisit on demand; nothing forbids it, nobody asked yet).
- Auto-update (violates the zero-network stance; never in scope).
