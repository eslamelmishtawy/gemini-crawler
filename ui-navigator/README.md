# UI Navigator ☸️

An autonomous AI agent that explores, understands, and tests web applications — powered by Google Gemini and ADK.

## What It Does

Give it a URL. It explores the entire application autonomously:
1. **Page Analyzer** — Gemini Pro sees every screen and identifies all interactive elements
2. **Navigator** — Gemini generates Playwright code on the fly (no predefined scripts)
3. **Sentinel** — Compares before/after screenshots to understand what changed
4. **Scout** — Strategically decides what to explore next using BFS traversal
5. **Pattern Dedup** — Detects repeated UI structures (product grids, list items) and only tests representative samples

The result: a complete navigation map of the application with every screen, every action, and every flow documented.

## Architecture

![Architecture Diagram](docs/architecture.png)

**15 nodes, 29 edges** — hub-and-spoke architecture where all agent communication flows through the Orchestrator.

### Agents
- **Orchestrator** — Central hub, runs the main loop (Gemini Flash)
- **Scout** — Discovery strategy, reads screen/action state (no LLM)
- **Page Analyzer** — Screen understanding with DOM + Vision in parallel (Gemini Pro + Flash)
- **Sentinel** — State identity gatekeeper, before/after comparison (Gemini Pro)
- **Navigator** — Dynamic code generation + execution (Gemini Pro)
- **Validator** — Assertion engine (Gemini Pro)
- **Reporter** — Report generation (Gemini Flash)

### Key Features
- **Dynamic Tool Generation** — Navigator generates Playwright code at runtime instead of using predefined tool schemas
- **Action Groups** — Batches related actions (form fills) for 3-5x speed improvement
- **Pattern Dedup** — 90 product card actions → 6 representatives tested, 84 inherited
- **Visual Fingerprinting** — Identifies screens by what they LOOK LIKE, not DOM hash
- **Scrollability Detection** — Vision-based detection of scrollable content

## Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | Google ADK (Python) |
| AI Models | Gemini 2.5 Pro (vision) + Gemini 2.5 Flash (parsing) |
| Web Automation | Playwright (headless Chromium) |
| State Storage | Cloud Firestore |
| Artifacts | Cloud Storage |
| Deployment | Cloud Run |
| Frontend | Streamlit |

## Quick Start

### Prerequisites
- Python 3.11+
- Google Cloud project with Gemini API enabled
- Firestore database

### Setup
```bash
git clone https://github.com/YOUR_USERNAME/ui-navigator.git
cd ui-navigator

# Install dependencies
pip install -r requirements.txt
playwright install chromium

# Configure
cp .env.example .env
# Edit .env with your GCP project ID and Gemini API key

# Run
streamlit run src/ui/app.py
```

### Docker
```bash
docker build -t ui-navigator .
docker run -p 8080:8080 --env-file .env ui-navigator
```

### Deploy to Cloud Run
```bash
gcloud run deploy ui-navigator \
  --source . \
  --region us-central1 \
  --memory 2Gi \
  --timeout 300 \
  --set-env-vars "GOOGLE_API_KEY=your-key,GCP_PROJECT_ID=your-project"
```

## How It Works

### Exploratory Mode (Main Loop)
```
Orchestrator → Scout (what next?)
  → Action Group Resolver (batch actions)
  → Test Data Provider (fill values)
  → Navigator (generate + execute code via Sandbox)
  → Sentinel (verify: same screen or new?)
  → if new: Page Analyzer (understand + dedup)
  → loop
```

### Architecture Decisions
- **Vision-first state identity** — DOM is unreliable (scroll = new DOM nodes but same page). Gemini vision determines screen identity.
- **Hub-and-spoke** — All agents communicate through Orchestrator. No direct agent↔agent calls.
- **Mobile-ready** — Architecture supports Appium as drop-in executor replacement (not implemented in MVP).

## Project Structure
```
ui-navigator/
├── src/
│   ├── agents/          # ADK agent definitions
│   ├── tools/           # Tool functions agents can call
│   ├── models/          # Pydantic data models
│   ├── services/        # Business logic (fingerprint, dedup, grouping)
│   ├── sandbox/         # Code execution environment
│   └── ui/              # Streamlit frontend
├── tests/
├── docs/
├── Dockerfile
└── requirements.txt
```

## Built For

Google AI Hackathon — UI Navigator ☸️ track.

**Mandatory Requirements:**
- ✅ Gemini model (Pro + Flash)
- ✅ Google ADK (Agent Development Kit)
- ✅ Google Cloud service (Cloud Run + Firestore)
