# 🎬 Conversational Movie Recommendation Engine

A terminal-based conversational movie recommender powered by **TMDB dataset**, **Regex/NLP intent parsing**, a **graph-weight scoring matrix**, and **Groq's Llama 3.1** as the conversational LLM.

---

## 📁 Project Structure

```
movie-recommendation-backend/
│
├── main.py                  # Entry point — CLI loop + Groq integration
├── intent_parser.py         # Regex/NLP intent extraction
├── scoring_engine.py        # Dataset ingestion, normalization, scoring
├── state_manager.py         # user_state.json read/write + graph updates
├── display.py               # Terminal formatting (ANSI colours, monitor)
│
├── user_state.json          # Live preference state (auto-created)
├── requirements.txt
│
├── data/                    # ← Put your Kaggle CSVs here
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
│
├── README.md
└── INSTRUCTIONS.md
```

---

## ⚡ Quick Start

See **[INSTRUCTIONS.md](INSTRUCTIONS.md)** for the full step-by-step setup guide.

```bash
# 1. Clone / unzip the project
cd movie-recommendation-backend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key
export GROQ_API_KEY="your_groq_api_key_here"   # Mac/Linux
# $env:GROQ_API_KEY="your_groq_api_key_here"   # Windows PowerShell

# 4. Run
python main.py
```

---

## 🧠 How It Works

### 5-Step Loop (every message)

```
User Input
    │
    ▼
1. Intent Parser  →  genres, actors, plot keywords, liked/disliked titles
    │
    ▼
2. State Manager  →  update user_state.json (boost likes, decay dislikes)
    │
    ▼
3. Scoring Engine →  filter TMDB dataset, compute Final Score per movie
    │
    ▼
4. Groq LLM       →  convert scored list into natural conversational reply
    │
    ▼
5. Live Monitor   →  print updated user_state.json to terminal
```

### Scoring Formula

$$\text{Final Score} = (\text{normalized\_base\_weight} + \text{matching\_points}) \times W_{\text{movie}} \times \prod W_{\text{genres}}$$

| Variable | Description |
|---|---|
| `normalized_base_weight` | Min-Max normalised TMDB `vote_average` (0.0–1.0) |
| `matching_points` | +2.0 per genre match, +3.0 per actor match, +1.5 per keyword match |
| `W_movie` | 0.0 if disliked (Graph Eraser), 1.0 otherwise |
| `∏ W_genres` | Product of all genre weights from `user_state.json` |

### State Updates

| Event | Effect |
|---|---|
| User **likes** a movie | Genre weights **+0.15** (capped at 1.5) |
| User **dislikes** a movie | Movie weight → **0.0** forever; genre weights **×0.70** |

---

## 💬 Example Session

```
User: Suggest me some sci-fi movies
Bot:  Here are some great science fiction picks for you:
      1. Inception (Score: 4.21)
      2. Interstellar (Score: 4.08)
      ...

User: I hated Interstellar
[Graph Eraser Action] Severing link user → 'Interstellar' (Weight: 0.0)
[Graph Eraser Action] Decaying category cluster [Science Fiction]: 1.00 → 0.70

Bot:  Got it! I've removed Interstellar from your recommendations and tuned
      down your sci-fi weighting. Want me to suggest something else?
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Your Groq API key from console.groq.com |

---

## 📊 Dataset

This project uses the **TMDB 5000 Movies Dataset** from Kaggle:
👉 https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

Download and place these two files in the `data/` folder:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Dataset | TMDB 5000 (Kaggle) |
| LLM | Groq — Llama 3.1 8B Instant |
| NLP | Regex + keyword matching |
| Data | pandas, numpy |
| State | Local JSON file (`user_state.json`) |