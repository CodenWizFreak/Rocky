"""
display.py — Terminal formatting and ANSI output helpers
"""

# ANSI colour codes
CYAN   = "\033[36m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"
RED    = "\033[31m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def print_banner() -> None:
    print(f"""
{CYAN}{BOLD}{'=' * 70}
  MOVIE RECOMMENDATION ENGINE  —  VS CODE TERMINAL MODE
  Powered by TMDB dataset · Groq LLM · Semantic Embeddings
{'=' * 70}{RESET}
""")


def print_system(msg: str) -> None:
    print(f"{DIM}[System] {msg}{RESET}")


def print_bot(message: str) -> None:
    print(f"{CYAN}Bot:{RESET} {message}")


def print_debug(intent: dict) -> None:
    method     = intent.get("parse_method", "?")
    m_colour   = GREEN if method == "llm" else YELLOW

    print(f"\n{DIM}", end="")
    print(f"[Debug] Parse method  : {m_colour}{method}{RESET}{DIM}")
    print(f"[Debug] Genres        : {intent.get('genres_requested',    [])}")
    print(f"[Debug] Actors        : {intent.get('actors_requested',    [])}")
    print(f"[Debug] Directors     : {intent.get('directors_requested', [])}")
    print(f"[Debug] Plot desc     : {intent.get('plot_description',    '') or '—'}")
    print(f"[Debug] Plot keywords : {intent.get('plot_keywords',       [])}")
    print(f"[Debug] Liked         : {intent.get('liked_found',         [])}")
    print(f"[Debug] Disliked      : {intent.get('disliked_found',      [])}")
    print(f"[Debug] Liked genres  : {intent.get('liked_genres',        [])}")
    print(f"[Debug] Block genres  : {intent.get('disliked_genres',     [])}")
    print(f"[Debug] Last-recs feel: {intent.get('sentiment_last_recs', None)}")
    print(f"[Debug] Wants diff    : {intent.get('wants_different',     False)}{RESET}")


def print_state_monitor(state: dict) -> None:
    weights  = state.get("genre_weights",        {})
    blocked  = state.get("disliked_movies",      [])
    blocked_g = state.get("disliked_genres",     [])
    liked    = state.get("liked_movies",         [])
    liked_g  = state.get("liked_genres",         [])
    last     = state.get("last_recommendations", [])

    print(f"\n{CYAN}{'─' * 62}")
    print(f"  LIVE STATE MONITOR  —  user_state.json")
    print(f"{'─' * 62}{RESET}")

    print(f"{BOLD}  Genre Weights:{RESET}")
    for genre, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        filled = int(round(w * 10))
        filled = max(0, min(filled, 15))
        bar    = "█" * filled + "░" * (15 - filled)
        if w < 0.999:
            colour, flag = RED,   f" {RED}↓ decayed{RESET}"
        elif w > 1.001:
            colour, flag = GREEN, f" {GREEN}↑ boosted{RESET}"
        else:
            colour, flag = DIM,   ""
        print(f"    {colour}{genre:<22}{RESET}{bar}  {w:.4f}{flag}")

    print(f"\n{BOLD}  Liked    :{RESET} {liked   if liked   else '—'}")
    print(f"{BOLD}  Like G   :{RESET} {liked_g if liked_g else '—'}")
    print(f"{BOLD}  Blocked  :{RESET} {blocked if blocked else '—'}")
    print(f"{BOLD}  Block G  :{RESET} {blocked_g if blocked_g else '—'}")
    print(f"{BOLD}  Last recs:{RESET} {last    if last    else '—'}")
    print(f"{CYAN}{'─' * 62}{RESET}\n")
