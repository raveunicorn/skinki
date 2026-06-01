# Skinki — Roadmap

The plan to get from an empty repo to a consumer-ready `.dmg`, and beyond.

Legend: `[ ]` not started · `[~]` in progress · `[x]` done.

---

## MVP focus

Per the product brief, the first prototype concentrates on three killer capabilities:

1. **Conversational chat with the lizard mascot** — a beautiful floating HUD.
2. **Global Text Engine** — rewrite / translate / summarize selected text in any app, with in-place replacement.
3. **Voice** — dictation input and spoken responses.

Everything else (Screen Awareness, Finder Quick Actions, neural TTS, RAG memory, the 26B tier polish) is documented here as post-MVP.

---

## Phase 0 — Foundation (this stage)

- [x] Architecture, documentation, and module boundaries (`ARCHITECTURE.md`, `docs/`).
- [x] Tuist project + workspace configuration; non-sandboxed, hardened-runtime, macOS 15+.
- [x] Skeletons for all local Swift packages with protocol-driven APIs.
- [x] Thin app target (menu-bar app shell, hotkey registration hooks).
- [ ] `tuist generate` produces a buildable, empty-but-running menu-bar app. *(first implementation task)*

---

## Phase 1 — The 4-week MVP

### Week 1 — Foundation & first token
- [ ] `tuist generate` builds and runs a menu-bar (`LSUIElement`) app.
- [ ] `DesignSystem`: color/blur/motion tokens, base components, app icon placeholder.
- [ ] `InferenceEngine`: integrate `mlx-swift-lm`, download + load **Gemma 4 E4B**, stream tokens to a debug window.
- [ ] Rive mascot spike: import a `.riv`, drive idle/think states from code.

### Week 2 — Chat with the mascot
- [ ] `ChatHUD`: borderless floating `NSPanel`, spotlight-style, native blur.
- [ ] Global hotkey to summon/dismiss the HUD (`SystemBridge`).
- [ ] Streaming chat UI bound to `InferenceEngine` with the mascot reacting (idle → thinking → typing → done).
- [ ] Hardware tiering: auto-select E4B vs 26B-A4B; manual override in Settings.
- [ ] `mmap` fast start + idle `unload()` + memory-pressure handling.
- [ ] Onboarding flow that requests Accessibility permission with the mascot guiding the user.

### Week 3 — Global Text Engine
- [ ] `SystemBridge`: read selected text via Accessibility across arbitrary apps.
- [ ] `TextEngine`: rewrite / translate / summarize pipelines + prompt templates (RU + EN).
- [ ] In-place replacement via simulated input (`CGEvent`).
- [ ] Smart Clipboard: clipboard history with quick recall.

### Week 4 — Voice, polish, packaging
- [ ] Voice input via native dictation (`Speech` / `SFSpeechRecognizer`).
- [ ] `VoiceEngine`: `AVSpeechSynthesizer` TTS with quality RU/EN Premium voices, behind the `SpeechSynthesizing` protocol.
- [ ] Joy-design polish pass: micro-interactions, transitions, sounds, haptics.
- [ ] `.dmg` packaging: codesign (Developer ID) + notarize + staple.
- [ ] `git init` private repo, CI lint/build check.
- [ ] **Stretch:** enable RAG long-term memory end to end.

**MVP exit criteria:** a downloadable `.dmg` that installs with no terminal, onboards gracefully, chats locally with the mascot, transforms selected text anywhere, and speaks/listens — in Russian and English.

---

## Phase 2 — Native depth & senses

- [ ] **Screen Awareness:** `ScreenCaptureKit` capture on hotkey; multimodal Gemma 4 (vision) for contextual help (IDE errors, table digitization).
- [ ] **Finder Quick Actions:** right-click context-menu actions (summarize file, rename batch, organize).
- [ ] **System Mastery:** scheduled + on-demand file operations (smart Downloads sorting, log archival); safe shell with mandatory human-in-the-loop confirmation and previews.
- [ ] **Long-term memory (full):** preferences, tone of voice, frequent paths, code style — see [`docs/MEMORY.md`](docs/MEMORY.md).

## Phase 3 — Voice & expressiveness

- [ ] Local neural TTS (Piper/Kokoro-class) with natural, human RU/EN intonation behind the existing `SpeechSynthesizing` seam.
- [ ] Wake word / push-to-talk conversational mode.
- [ ] Richer mascot performances synced to speech.

## Phase 4 — Power users & ecosystem

- [ ] Tool/function calling and a safe action framework (agentic workflows).
- [ ] Developer mode: deeper code context, repo-aware assistance.
- [ ] Plugin/extension API.
- [ ] Open-source the repository under a permissive license; public website + signed releases.

---

## Cross-cutting, ongoing

- Performance budget tracking (TTFT, tokens/sec, idle RAM).
- Accessibility (VoiceOver) and localization QA (RU/EN parity).
- Automated build/lint in CI; reproducible `tuist generate`.
