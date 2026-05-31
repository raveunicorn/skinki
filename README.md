<div align="center">

# 🦎 Skinki

**The ultimate local AI assistant for macOS — what Apple Intelligence should have been.**

Native. Private. Joyful. Powered by on-device Gemma 4 via Apple MLX.

</div>

---

## What is Skinki?

Skinki is a fully local, privacy-first AI assistant for macOS, shipped as a consumer-ready `.dmg`. Its mascot — a friendly, expressive lizard — is a modern, joy-design reimagining of the classic desktop assistant. Skinki is built to feel like a *hidden feature of the OS itself*: deeply integrated, beautifully animated, and respectful of your machine's resources.

It is designed to be equally **intuitive for a non-technical user** on a base MacBook and **incredibly powerful for a developer** on a Mac Studio.

> Everything runs on your Mac. No cloud. No subscription. No telemetry.

## Core Pillars

1. **Uncompromising UI/UX** — native blurs, fluid SwiftUI/Metal animations, a *living* mascot, and a zero-friction onboarding. No terminals, ever.
2. **Native-like macOS integration** — global hotkeys, Finder Quick Actions, a Status Bar home, and system-wide text capture via the Accessibility API.
3. **Extreme hardware efficiency** — `mmap` cold start, idle model unload, quantization. Skinki should *fly* even on modest machines, in the background.
4. **Evolution & memory** — long-term memory that learns your preferences, tone of voice, frequent paths, and coding style (RAG over a local vector store).
5. **LLM core & multilingual** — built on Google's open-weight **Gemma 4** family with first-class Russian and English support out of the box.

## Tech Stack (at a glance)

| Layer | Choice |
| --- | --- |
| UI | SwiftUI + [Rive](https://rive.app) (interactive mascot) + Metal |
| Inference | [`mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm) (Apple MLX, **no Python**) |
| Models | Gemma 4 **E4B** (base Macs) · Gemma 4 **26B-A4B MoE** (32GB+ unified memory) |
| Embeddings / RAG | EmbeddingGemma (`MLXEmbedders`) + SQLite + [`sqlite-vec`](https://github.com/asg017/sqlite-vec) |
| Voice | Native dictation in · `AVSpeechSynthesizer` out (neural TTS later) |
| Project | [Tuist](https://tuist.dev) + modular local Swift packages |

## Requirements

- Apple Silicon Mac (M1 or newer)
- macOS 15 (Sequoia) or later
- 16 GB unified memory recommended for the E4B tier; 32 GB+ for the 26B-A4B tier

## Getting Started (development)

> The project is scaffolded but not yet implemented. These are the intended commands.

```bash
# 1. Install Tuist (https://tuist.dev)
curl -Ls https://install.tuist.io | bash

# 2. Generate the Xcode project + workspace
tuist generate

# 3. Open and run
open Skinki.xcworkspace
```

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — high-level architecture, layers, and the SwiftUI ↔ MLX ↔ macOS bridge.
- [`ROADMAP.md`](ROADMAP.md) — the 4-week MVP plan and the beyond.
- [`docs/DESIGN.md`](docs/DESIGN.md) — joy-design principles and the mascot system.
- [`docs/MEMORY.md`](docs/MEMORY.md) — the long-term memory & RAG design.

## Status

🚧 **Foundation / scaffolding stage.** This repository currently contains the architecture, documentation, project configuration, and module skeletons. Feature implementation follows the [roadmap](ROADMAP.md).

## License

To be released under a permissive open-source license (MIT or Apache-2.0) once it goes public. Currently developed in a private repository.
