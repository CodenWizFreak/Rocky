# GNN Movie Recommender — Full Stack Setup

A conversational movie recommendation system with machine unlearning (Graph Eraser),
powered by a Groq LLM backend and a Next.js glassmorphism frontend with a live GNN visualizer.

---

## Directory Structure

```
project/
├── backend/               ← Python FastAPI server
│   ├── api.py             ← FastAPI app (NEW — main entry point)
│   ├── main.py            ← Original CLI entrypoint (unchanged)
│   ├── intent_parser.py
│   ├── scoring_engine.py
│   ├── state_manager.py
│   ├── embedder.py
│   ├── display.py
│   ├── requirements.txt
│   ├── .env               ← copy from .env.example and fill in
│   ├── .env.example
│   ├── user_state.json    ← persisted user preference graph
│   └── data/
│       ├── movies_metadata.csv
│       ├── credits.csv
│       └── ratings.csv
│
└── frontend/              ← Next.js 14 App Router
    ├── app/
    │   ├── chat/page.tsx           ← main chat UI (updated)
    │   ├── api/
    │   │   ├── chat/route.ts       ← proxies to Python SSE stream
    │   │   ├── graph/route.ts      ← fetches GNN graph data
    │   │   ├── greet/route.ts      ← gets greeting from bot
    │   │   ├── state/route.ts      ← fetches user state
    │   │   └── reset/route.ts      ← resets conversation + state
    │   └── ...
    ├── components/
    │   ├── chat/
    │   │   ├── ChatWindow.tsx      ← updated (uses local ChatMessage type)
    │   │   ├── ChatInput.tsx       ← unchanged
    │   │   ├── MessageBubble.tsx   ← updated (uses local ChatMessage type)
    │   │   └── GnnVisualizer.tsx   ← NEW — live force-directed GNN graph
    │   └── ui/                     ← unchanged glass UI components
    ├── .env.local                  ← BACKEND_URL=http://localhost:8000
    └── ...
```

---

## Quick Start

### 1. Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — set your GROQ_API_KEY and GROQ_MODEL

# Start FastAPI server
uvicorn api:app --reload --port 8000
```

The first startup takes ~30–60 seconds to:
- Load the movie database (~45k movies)
- Build/load the plot embedding index
- Initialize user state

You'll see `[API] Ready ✓` when it's done.

### 2. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

Open http://localhost:3000/chat

---

## Environment Variables

### Backend `.env`

```env
# Required
GROQ_API_KEY=gsk_...              # Get from console.groq.com
GROQ_MODEL=llama-3.1-8b-instant   # or llama-3.3-70b-versatile

# Optional
GROQ_TEMPERATURE=0.8
GROQ_MAX_TOKENS=1024
TOP_N_RESULTS=5
EMBEDDING_BACKEND=local           # "local" uses sentence-transformers
STATE_FILE=user_state.json
EMBEDDINGS_CACHE=embeddings_cache.npy
```

### Frontend `.env.local`

```env
BACKEND_URL=http://localhost:8000
```

---

## How It Works

### Chat Flow

1. User types a message in the frontend
2. Next.js `/api/chat` route POSTs `{ message }` to Python `/chat`
3. Python backend:
   - Parses intent (LLM-first with Groq, regex fallback)
   - Updates user preference graph (likes/dislikes/genres)
   - Scores and ranks movies using the hybrid scoring engine
   - Streams the Groq response token-by-token via SSE
   - Sends a final `graph` event with updated GNN data
4. Frontend streams tokens into the chat bubble in real time
5. Graph visualizer updates automatically after each message

### SSE Event Format

```
data: {"type": "token", "content": "Here are"}
data: {"type": "token", "content": " my top picks..."}
data: {"type": "graph", "data": { nodes: [...], edges: [...] }}
data: {"type": "done"}
```

### GNN Visualizer

Click **"See Live Visualizer"** above the chat to open an overlay with a force-directed graph showing:

| Node type   | Color  | Meaning                          |
|-------------|--------|----------------------------------|
| User        | Green  | You (center node)                |
| Recommended | Purple | Movies recommended this session  |
| Liked       | Blue   | Movies you said you liked        |
| Erased      | Red    | Movies erased by Graph Eraser    |
| Genre       | Amber  | Genre nodes with weights         |

Edge weights show recommendation scores. Erased nodes appear with a red strikethrough.

Controls: scroll to zoom, drag to pan, hover nodes for details.

---

## Machine Unlearning (Graph Eraser)

Tell the bot things like:
- "I hate Conjuring 2, forget it"
- "Never recommend horror again"
- "I dislike The Emoji Movie"

The system will:
1. Remove the movie/genre from your preference graph
2. Decay the genre weights in the scoring engine
3. Exclude the movie/franchise from all future recommendations
4. Update the GNN visualizer to show the erased node in red

---

## API Endpoints

| Method | Path     | Description                          |
|--------|----------|--------------------------------------|
| GET    | /health  | Backend health check                 |
| GET    | /state   | Current user preference state        |
| GET    | /graph   | GNN graph data for visualizer        |
| GET    | /greet   | Get initial greeting from bot        |
| POST   | /chat    | Send message, receive SSE stream     |
| POST   | /reset   | Reset all state and conversation     |

---

## Troubleshooting

**Backend not starting:**
- Check `GROQ_API_KEY` and `GROQ_MODEL` are set in `.env`
- Ensure `data/movies_metadata.csv`, `credits.csv`, `ratings.csv` exist

**"Backend Offline" shown in frontend:**
- Make sure the Python server is running on port 8000
- Check CORS is not blocked (it's configured for `localhost:3000`)

**Embeddings slow on first run:**
- First run builds a `embeddings_cache.npy` file — subsequent runs are fast
- Set `EMBEDDING_BACKEND=local` to use the free local sentence-transformers model

**Graph visualizer empty:**
- It populates after your first chat message that returns recommendations
- Ask "recommend me a movie" to populate it
