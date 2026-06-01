# Skinki — macOS app (parked, Stage 7)

This is the **consumer wrapper** for the Exocortex engine: a native, joy-design
macOS app (menu-bar + floating HUD + interactive mascot) that lets you capture
thoughts (voice/text) and surfaces grounded memories and insights.

> **Status: parked scaffolding.** Per the [Exocortex pivot](../../README.md),
> the primary product is the headless Rust engine in [`kortex/`](../../kortex/).
> This app is revived at **Stage 7** of the [roadmap](../../ROADMAP.md), once the
> engine meets its M1 Air budgets. The SwiftUI/Tuist scaffolding here is kept
> intact and will consume `kortex` through Swift bindings (Stage 6) instead of
> the originally-planned in-app SQLite RAG.

## What's here

A Tuist-managed, non-sandboxed macOS 15+ app split into local Swift packages:

| Path | Role |
| --- | --- |
| `App/` | Thin app target: lifecycle, menu-bar item, window/HUD hosting. |
| `Packages/SkinkiCore` | Domain models, protocols, DI, config, logging. |
| `Packages/InferenceEngine` | Gemma 4 via `mlx-swift-lm` (will defer memory/RAG to `kortex`). |
| `Packages/MemoryStore` | Legacy in-app RAG skeleton — superseded by `kortex`, kept for reference. |
| `Packages/SystemBridge` | Accessibility capture, input simulation, hotkeys, shell. |
| `Packages/TextEngine` | Rewrite / translate / summarize pipelines. |
| `Packages/VoiceEngine` | Dictation in, speech synthesis out. |
| `Packages/DesignSystem` | Tokens, joy components, the Rive mascot. |
| `Packages/Features` | Composition: ChatHUD, MenuBar, Onboarding, Settings. |

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the layered design and the
SwiftUI ↔ inference ↔ macOS bridge, and [`ROADMAP.md`](ROADMAP.md) for the
original app MVP plan (now reframed as the Stage 7 wrapper).

## Development (when revived)

```bash
# from this directory (apps/skinki-macos)
curl -Ls https://install.tuist.io | bash   # install Tuist
tuist generate                              # generate the Xcode project + workspace
open Skinki.xcworkspace
```

Manifests (`Project.swift`, `Workspace.swift`, `Tuist.swift`) use paths relative
to this directory, so the app builds standalone from here within the monorepo.
