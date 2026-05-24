"""
intent_parser.py — LLM-first intent classification with exhaustive regex fallback

PRIMARY  : Groq call at temperature=0, returns strict JSON
FALLBACK : Regex pipeline (fires if LLM call fails for any reason)

Output schema (identical from both paths):
{
    "genres_requested":    ["Comedy", "Thriller"],
    "actors_requested":    ["Cillian Murphy"],
    "directors_requested": ["Christopher Nolan"],
    "plot_description":    "a man loses his memory and tries to rebuild his identity",
    "plot_keywords":       ["memory", "identity"],
    "liked_found":         ["Inception"],
    "disliked_found":      ["The Emoji Movie"],
    "liked_genres":        ["Action"],
    "disliked_genres":     ["Music"],
    "sentiment_last_recs": "negative" | "positive" | "neutral" | null,
    "wants_different":     false,
    "parse_method":        "llm" | "regex"
}
"""

import re
import json
import logging
import pandas as pd
from groq import Groq

logger = logging.getLogger(__name__)

CANONICAL_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
    "Romance", "Science Fiction", "Thriller", "War", "Western",
]

GENERIC_TITLE_REJECTS = {
    "fun", "cool", "awesome", "shit", "explosions", "adventurous", "something",
    "movies", "movie", "films", "film", "it", "that", "this", "those", "these",
    "them", "ones",
}

EXPLICIT_GENRE_KEYWORDS = {
    "action", "adventure", "animated", "animation", "anime", "comedy",
    "comedies", "crime", "documentary", "documentaries", "drama", "family",
    "fantasy", "history", "historical", "horror", "music", "musical",
    "musicals", "mystery", "romance", "romantic", "sci-fi", "scifi",
    "science fiction", "thriller", "thrillers", "war", "western", "westerns",
}

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — EXHAUSTIVE REGEX PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

LIKED_PATTERNS = [
    r"\b(?:gotta|gott) love (.+?)(?:\.|!|\?|$)",
    r"\b(?:love|loved|like|liked|enjoy|enjoyed) (.+?)(?:\.|!|\?|$)",
    r"\bi (?:really |absolutely |genuinely |truly |completely |totally )?(?:loved?|liked?|enjoyed|adored?|appreciated?|relished|cherished|dug|vibed? with|was into|got into|am into) (.+?)(?:\.|!|\?|$)",
    r"(.+?) (?:was|is|were|are) (?:really |absolutely |genuinely |truly |so )?(?:great|amazing|fantastic|awesome|wonderful|brilliant|excellent|incredible|outstanding|superb|perfect|beautiful|stunning|breathtaking|masterpiece|a gem|a classic|gold|fire|lit|goated|peak|peak cinema|so good|the best|top tier|S-tier)",
    r"(?:(?:big|huge|massive|total|absolute) )?fan of (.+?)(?:\.|!|\?|$)",
    r"(?:keep|more|bring) (?:them|those|it|movies?) (?:like|coming|similar to) (.+?)(?:\.|!|\?|$)",
    r"(.+?) (?:blew me away|hit different|slapped|went hard|was a banger|was fire|was peak|was chef'?s? kiss|was everything|exceeded my expectations|surpassed my expectations)",
    r"(?:that|this|it) (?:was|is) (?:a )?(?:masterpiece|banger|gem|great|amazing|fantastic|awesome|wonderful|brilliant|excellent)",
    r"\bi (?:really |absolutely )?(?:loved?|enjoyed|liked?) (?:watching|seeing|viewing) (.+?)(?:\.|!|\?|$)",
    r"\bi(?:'d| would| could) (?:watch|see|rewatch) (.+?) (?:again|forever|on repeat|anytime)",
    r"(.+?) (?:was|is) exactly what i (?:was looking for|needed|wanted)",
    r"\bi(?:'m| am) (?:glad|happy|grateful|pleased) (?:i |that i )?(?:watched?|saw|viewed) (.+?)(?:\.|!|\?|$)",
    r"\bi (?:couldn'?t|could not) (?:stop watching|put it down|turn it off|look away from) (.+?)(?:\.|!|\?|$)",
    r"\bi(?:'ve| have) (?:watched|seen|viewed) (.+?) (?:multiple times|twice|three times|\d+ times)",
    r"\bobsessed with (.+?)(?:\.|!|\?|$)",
    r"(.+?) (?:changed my life|blew my mind|was mind.?blowing|is a must.?watch|is a must.?see)",
    r"more (?:like|movies like|films like|of) (.+?)(?:\.|!|\?|$)",
    r"something (?:like|similar to|in the style of) (.+?)(?:\.|!|\?|$)",
    r"^(?:yes+|yep|yeah|yass+|love it|love them|perfect|great|amazing|awesome|fantastic|brilliant|excellent|wonderful|nice|cool|sick|fire|lit|goated|banger|slaps?|chef'?s? kiss|exactly|spot on|that'?s? it|that'?s? the one|more like (?:this|that|these))[\s!.]*$",
]

DISLIKED_PATTERNS = [
    r"\bi (?:really |absolutely |genuinely |truly |completely |totally )?(?:hated?|disliked?|despised?|loathed?|detested?|couldn'?t stand|can'?t stand|wasn'?t into|am not into|was not into) (.+?)(?:\.|!|\?|$)",
    r"(.+?) (?:was|is|were|are) (?:really |absolutely |genuinely |truly |so )?(?:terrible|awful|bad|horrible|boring|trash|garbage|rubbish|dreadful|appalling|atrocious|unwatchable|a waste|a mess|disappointing|mid|meh|mediocre|overrated|not good|not great|not that good|not for me|not my thing|not my cup of tea|painfully bad|cringe|cringey|corny|a snooze(?:fest)?|a slog)",
    r"(?:don'?t|never|please don'?t|do not) (?:show|recommend|suggest|give) (?:me )?(.+?)(?:\.|!|\?|$)",
    r"(?:not (?:a |(?:really ))?fan of|no interest in|zero interest in|couldn'?t care less about) (.+?)(?:\.|!|\?|$)",
    r"(?:skip|pass on|avoid|stayed away from|steered clear of) (.+?)(?:\.|!|\?|$)",
    r"(.+?) (?:put me to sleep|bored me (?:to death|to tears|out of my mind)|dragged(?:on)?|was painfully slow)",
    r"\bi (?:didn'?t|did not|don'?t|do not|won'?t|would not|can'?t|cannot) (?:really |particularly |actually )?(?:like|enjoy|appreciate|get|understand the hype (?:of|for|around)) (.+?)(?:\.|!|\?|$)",
    r"\bi (?:lost interest in|gave up on|stopped watching|turned off|walked out of|couldn'?t finish|didn'?t finish|never finished) (.+?)(?:\.|!|\?|$)",
    r"(.+?) (?:wasn'?t|isn'?t|is not|was not) (?:really )?(?:for me|my thing|my vibe|my style|my cup of tea|up my alley|what i expected|what i hoped)",
    r"\bi (?:was )?(?:disappointed (?:by|with|in)|let down by|underwhelmed by) (.+?)(?:\.|!|\?|$)",
    r"(.+?) (?:didn'?t|did not) (?:live up to|meet|match) (?:the hype|my expectations|expectations)",
    r"(?:remove|block|exclude|take off|take out|get rid of) (.+?) (?:from (?:my )?(?:list|recommendations?))",
    r"\bi(?:'ve| have) (?:already |)(?:seen|watched|viewed) (.+?) (?:already|before|multiple times|a hundred times)?(?:\.|!|\?|$)",
    r"(?:not (?:really )?in the mood for|don'?t feel like watching) (.+?)(?:\.|!|\?|$)",
    r"\bi (?:wasted|lost) .+? (?:hours?|mins?|minutes?) (?:on|watching) (.+?)(?:\.|!|\?|$)",
    r"\bi (?:regret|regretted) watching (.+?)(?:\.|!|\?|$)",
    r"(.+?) (?:was|is) (?:such a |a )?(?:waste of (?:my )?time|disappointment)",
    r"^(?:no+|nope|nah|ugh|yikes|pass|hard pass|not (?:really|quite)|not feeling (?:it|these|those|this|them)|meh|mid|these (?:are |look )?(?:boring|bad|terrible|awful|not great)|not what i (?:wanted|expected|was looking for)|something (?:else|different|better) please|none of (?:these|those)|can you (?:do better|try again)|try again)[\s!.]*$",
]

DIRECTOR_PATTERNS = [
    r"(?:directed by|films? by|movies? by|directed by|from (?:the )?director) (.+?)(?:\.|!|\?|$)",
    r"(?:i love|i like|i enjoy|fan of|big fan of) (?:(?:the )?(?:films?|movies?|work|directing) (?:of|by)) (.+?)(?:\.|!|\?|$)",
    r"(.+?)(?:'s| films?| movies?| directed films?| directorial) (?:films?|movies?|work|style|directing)",
    r"(?:anything|something|movies?|films?) (?:from|by) (.+?)(?:\.|!|\?|$)",
    r"(?:show|recommend|suggest|give) me (?:more )?(?:films?|movies?) by (.+?)(?:\.|!|\?|$)",
]

PLOT_DESCRIPTION_PATTERNS = [
    r"(?:movies?|films?) (?:where|about|in which|that (?:involve|feature|have|show)) (.{15,}?)(?:\.|!|\?|$)",
    r"(?:recommend|suggest|show|find) (?:me )?(?:movies?|films?) (?:where|about|with|featuring) (.{15,}?)(?:\.|!|\?|$)",
    r"(?:i(?:'m| am) looking for (?:a )?(?:movie|film) (?:where|about|with|in which)) (.{15,}?)(?:\.|!|\?|$)",
    r"(?:something (?:where|about|with|involving)) (.{15,}?)(?:\.|!|\?|$)",
    r"(?:a (?:movie|film) (?:where|about|with|featuring)) (.{15,}?)(?:\.|!|\?|$)",
    r"(?:plot(?:line)?|storyline|story) (?:like|similar to|where|about|involving) (.{15,}?)(?:\.|!|\?|$)",
]

GENRE_KEYWORDS: dict[str, list[str]] = {
    "Action":          ["action", "action-packed", "fight", "fights", "battle", "battles", "combat", "fighting", "martial arts", "kung fu", "karate", "explosion", "explosions", "explosive", "adrenaline", "high octane", "guns", "gunfight", "chase", "car chase", "hand to hand", "brawl", "intense"],
    "Adventure":       ["adventure", "adventures", "adventurous", "quest", "quests", "journey", "journeys", "expedition", "explore", "exploration", "discover", "discovery", "treasure hunt", "road trip", "epic journey"],
    "Animation":       ["animation", "animated", "cartoon", "cartoons", "pixar", "disney", "dreamworks", "anime", "2d", "3d animated", "claymation", "stop motion", "kids movie", "kids film"],
    "Comedy":          ["comedy", "comedies", "funny", "humour", "humor", "humorous", "laugh", "laughs", "laughing", "hilarious", "hilarity", "witty", "wit", "jokes", "joke", "silly", "lighthearted", "light-hearted", "feel good", "feel-good", "wholesome comedy", "rom-com", "romcom", "slapstick", "dark comedy", "black comedy", "satire", "satirical", "parody"],
    "Crime":           ["crime", "crimes", "criminal", "heist", "heists", "robbery", "theft", "mafia", "mob", "gangster", "gang", "detective", "detectives", "murder mystery", "murder", "killer", "assassin", "hitman", "whodunit", "law enforcement", "police procedural", "noir", "film noir"],
    "Documentary":     ["documentary", "documentaries", "doc", "docs", "real life", "true story", "based on true events", "non-fiction", "nonfiction", "factual", "educational", "informative", "nature documentary", "true crime doc"],
    "Drama":           ["drama", "dramas", "dramatic", "emotional", "emotionally charged", "character study", "character driven", "character-driven", "deep", "thought provoking", "thought-provoking", "moving", "powerful", "intense drama", "slice of life", "realistic", "grounded"],
    "Family":          ["family", "families", "family friendly", "family-friendly", "kids", "children", "wholesome", "all ages", "for kids", "for children", "suitable for kids", "pg", "g rated"],
    "Fantasy":         ["fantasy", "fantasies", "magical", "magic", "fairy tale", "fairy-tale", "mythical", "mythological", "mythology", "legend", "legendary", "wizards", "wizard", "sorcery", "sorcerer", "enchanted", "enchantment", "dragons", "dragon", "elves", "elf", "dwarves", "dwarf", "epic fantasy", "high fantasy", "dark fantasy", "supernatural fantasy"],
    "History":         ["history", "historical", "period", "period piece", "period drama", "ancient", "medieval", "renaissance", "victorian", "war era", "world war", "wwii", "wwi", "cold war era", "set in the past", "biographical", "biopic", "real events"],
    "Horror":          ["horror", "scary", "terrifying", "terrified", "frightening", "frightened", "spooky", "haunted", "haunt", "haunting", "creepy", "eerie", "sinister", "gore", "gory", "slasher", "psychological horror", "supernatural horror", "monster", "monsters", "demon", "demons", "possessed", "possession", "jumpscares", "jump scare", "survival horror", "body horror"],
    "Music":           ["music", "musical", "musicals", "concert", "concerts", "band", "bands", "singer", "singing", "song", "songs", "dance", "dancing", "jazz", "rock", "hip hop", "rap", "pop music", "opera", "soundtrack"],
    "Mystery":         ["mystery", "mysteries", "mysterious", "whodunit", "whodunnit", "suspense", "suspenseful", "clues", "investigation", "investigate", "detective", "puzzle", "twists", "plot twist", "unsolved", "crime mystery", "locked room"],
    "Romance":         ["romance", "romantic", "love story", "love stories", "relationship", "relationships", "dating", "couple", "couples", "heartwarming", "heart-warming", "sweeping romance", "passionate", "slow burn", "enemies to lovers", "will they won't they"],
    "Science Fiction": ["sci-fi", "science fiction", "scifi", "space", "alien", "aliens", "futuristic", "future", "robot", "robots", "ai", "artificial intelligence", "dystopia", "dystopian", "utopia", "cyberpunk", "steampunk", "time travel", "parallel universe", "multiverse", "interstellar", "intergalactic", "spaceship", "wormhole", "extraterrestrial", "post-apocalyptic", "tech", "technology driven"],
    "Thriller":        ["thriller", "thrillers", "suspense", "suspenseful", "tense", "tension", "psychological thriller", "psychological", "mind games", "plot twists", "edge of your seat", "edge-of-your-seat", "nail-biting", "gripping", "cat and mouse", "cat-and-mouse", "twist ending", "paranoia", "paranoid"],
    "War":             ["war", "wars", "military", "soldier", "soldiers", "combat", "battlefield", "wwii", "world war ii", "world war 2", "ww2", "wwi", "world war i", "world war 1", "ww1", "vietnam", "korean war", "civil war", "anti-war", "war drama", "war film"],
    "Western":         ["western", "westerns", "cowboy", "cowboys", "wild west", "frontier", "gunslinger", "saloon", "outlaw", "sheriff", "showdown", "spaghetti western"],
}

PLOT_KEYWORDS = [
    "space", "alien", "aliens", "robot", "robots", "time travel", "apocalypse",
    "post-apocalyptic", "dystopia", "cyberpunk", "artificial intelligence", "ai",
    "wormhole", "multiverse", "parallel universe", "zombie", "zombies", "vampire",
    "vampires", "werewolf", "witch", "witches", "ghost", "ghosts", "haunted",
    "demon", "demons", "monster", "monsters", "dragon", "dragons", "magic",
    "supernatural", "superhero", "superheroes", "spy", "spies", "heist", "heists",
    "assassin", "hitman", "serial killer", "detective", "prison", "school", "college",
    "ocean", "underwater", "jungle", "desert", "mountain", "forest", "small town",
    "big city", "survival", "revenge", "redemption", "betrayal", "conspiracy",
    "friendship", "family", "loss", "grief", "identity", "coming of age",
    "growing up", "love", "heartbreak", "obsession", "addiction", "corruption",
    "justice", "morality", "cult", "road trip", "treasure hunt", "mystery",
    "twist ending", "based on true story", "biographical", "true crime",
    "revolution", "political", "royal", "historical", "memory", "amnesia",
    "dreams", "hallucination", "simulation", "undercover", "double agent",
    "kidnapping", "hostage", "war", "genocide", "slavery", "immigration",
    "pandemic", "virus", "outbreak", "nature", "wildlife", "ocean", "deep sea",
]

WANTS_DIFFERENT_PATTERNS = [
    r"\b(?:something|anything) (?:else|different|other|new|fresh)\b",
    r"\b(?:different|other|another) (?:kind|type|genre|style)\b",
    r"\bnot (?:these|those|this|that)\b",
    r"\b(?:can you |could you )?(?:try again|do better|suggest (?:something )?(?:else|different))\b",
    r"\bnone of (?:these|those)\b",
    r"\bchange it up\b",
    r"\bswitch (?:it )?up\b",
    r"\bi(?:'ve| have) seen (?:all|most) of (?:these|those)\b",
    r"\bnot what i (?:had in mind|was thinking|wanted|expected)\b",
    r"\bcan(?:not|'t) you do better\b",
]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — REGEX ENGINE (FALLBACK)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_sentiment_titles(text: str, patterns: list[str]) -> list[str]:
    found, seen = [], set()
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            try:
                raw = m.group(1).strip().strip(".,!?\"'")
            except IndexError:
                continue
            if raw and raw.lower() not in seen and len(raw) > 1:
                found.append(raw)
                seen.add(raw.lower())
    return found


def _lenient_match(raw_title: str, movie_db: pd.DataFrame) -> str | None:
    """Case-insensitive substring match. Prefers longer (more specific) matches."""
    cleaned = raw_title.lower()
    cleaned = re.split(r"\b(?:especially|but|and|with|that|which)\b", cleaned, maxsplit=1)[0]
    cleaned = re.sub(r"\b(the movie|the film|movies|films|movie|film|ones?)\b", "", cleaned).strip()
    cleaned = re.sub(r"^(how|all|lots of|a lot of)\s+", "", cleaned).strip()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned).strip()
    if not cleaned or len(cleaned) < 2:
        return None
    if cleaned in GENERIC_TITLE_REJECTS:
        return None
    cleaned_tokens = set(re.findall(r"[a-z0-9]+", cleaned))
    if cleaned_tokens and cleaned_tokens <= GENERIC_TITLE_REJECTS:
        return None
    cleaned_tokens = set(re.findall(r"[a-z0-9]+", cleaned))
    best, best_score, best_len = None, 0.0, 0
    for _, row in movie_db.iterrows():
        t = str(row["title_clean"])
        if not t:
            continue
        t_norm = re.sub(r"^(?:the|a|an)\s+", "", t).strip()
        t_norm = re.sub(r"[^a-z0-9]+", " ", t_norm).strip()
        title_tokens = set(re.findall(r"[a-z0-9]+", t_norm))
        if cleaned == t or cleaned == t_norm:
            return row["title"]
        score = 0.0
        if title_tokens and title_tokens <= cleaned_tokens:
            score = 2.0
        elif cleaned_tokens and cleaned_tokens <= title_tokens:
            score = 1.5
        if score and (score > best_score or (score == best_score and len(t_norm) < best_len)):
            best, best_score, best_len = row["title"], score, len(t_norm)
    return best


def _extract_genres(text: str) -> list[str]:
    found = []
    lower = text.lower()
    for genre, kws in GENRE_KEYWORDS.items():
        for kw in kws:
            if re.search(r"\b" + re.escape(kw) + r"\b", lower, re.IGNORECASE):
                if genre not in found:
                    found.append(genre)
                break
    return found


NEGATIVE_GENRE_PATTERNS = [
    r"\b(?:fuck|hate|hated|dislike|disliked|despise|despised|loathe|loathed|avoid|block|exclude|skip|no|never|not into|can't stand|cannot stand)\b.{0,35}?\b({keywords})\b",
    r"\b({keywords})\b.{0,25}?\b(?:suck|sucks|are trash|is trash|are garbage|is garbage|are boring|is boring|are awful|is awful)\b",
]

POSITIVE_GENRE_PATTERNS = [
    r"\b(?:love|loved|like|liked|enjoy|enjoyed|want|wanna|more|show me|recommend)\b.{0,35}?\b({keywords})\b",
]


def _genre_keyword_regex() -> str:
    kws = []
    for words in GENRE_KEYWORDS.values():
        kws.extend(words)
    kws.sort(key=len, reverse=True)
    return "|".join(re.escape(k) for k in kws)


def _canonical_genre_from_keyword(keyword: str) -> str | None:
    lowered = keyword.lower()
    for genre, kws in GENRE_KEYWORDS.items():
        if lowered == genre.lower() or lowered in [kw.lower() for kw in kws]:
            return genre
    return None


def _extract_sentiment_genres(text: str, positive: bool) -> list[str]:
    patterns = POSITIVE_GENRE_PATTERNS if positive else NEGATIVE_GENRE_PATTERNS
    kw_re = _genre_keyword_regex()
    found = []
    for pat in patterns:
        compiled = pat.replace("{keywords}", kw_re)
        for match in re.finditer(compiled, text, re.IGNORECASE):
            raw_kw = match.group(1).lower()
            if positive and raw_kw not in EXPLICIT_GENRE_KEYWORDS:
                continue
            genre = _canonical_genre_from_keyword(raw_kw)
            if genre and genre not in found:
                found.append(genre)
    return found


def _extract_actors(text: str, movie_db: pd.DataFrame) -> list[str]:
    lower = text.lower()
    found, seen = [], set()
    cast_map: dict[str, str] = {}
    for cast_list in movie_db["cast_list"]:
        for name in (cast_list or []):
            nl = name.lower()
            if nl not in cast_map:
                cast_map[nl] = name
    for nl, name in cast_map.items():
        if nl in seen or len(nl) <= 3:
            continue
        if re.search(r"\b" + re.escape(nl) + r"\b", lower):
            found.append(name)
            seen.add(nl)
    return found


def _extract_directors_regex(text: str, movie_db: pd.DataFrame) -> list[str]:
    """Two-pass: (1) pattern match raw name, (2) verify against dataset."""
    lower   = text.lower()
    raw_names = []
    for pat in DIRECTOR_PATTERNS:
        for m in re.finditer(pat, lower, re.IGNORECASE):
            try:
                raw_names.append(m.group(1).strip().strip(".,!?\"'"))
            except IndexError:
                continue

    # Build director lookup from dataset
    dir_map: dict[str, str] = {}
    for dir_list in movie_db["directors_list"]:
        for name in (dir_list or []):
            nl = name.lower()
            if nl not in dir_map:
                dir_map[nl] = name

    found, seen = [], set()
    for raw in raw_names:
        rl = raw.lower()
        # Exact or partial match in director map
        for dl, dname in dir_map.items():
            if dl in seen:
                continue
            if rl == dl or rl in dl or dl in rl:
                found.append(dname)
                seen.add(dl)
                break
    return found


def _extract_plot_description(text: str) -> str:
    """Return the best plot description captured from the text, or ''."""
    for pat in PLOT_DESCRIPTION_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            desc = m.group(1).strip()
            if len(desc) > 15:
                return desc
    return ""


def _extract_plot_keywords(text: str) -> list[str]:
    lower = text.lower()
    return [kw for kw in PLOT_KEYWORDS if re.search(r"\b" + re.escape(kw) + r"\b", lower, re.IGNORECASE)]


def _check_wants_different(text: str) -> bool:
    lower = text.lower()
    return any(re.search(p, lower, re.IGNORECASE) for p in WANTS_DIFFERENT_PATTERNS)


def _regex_parse(user_input: str, movie_db: pd.DataFrame) -> dict:
    raw_liked    = _extract_sentiment_titles(user_input, LIKED_PATTERNS)
    raw_disliked = _extract_sentiment_titles(user_input, DISLIKED_PATTERNS)

    liked_found, disliked_found = [], []
    for r in raw_liked:
        m = _lenient_match(r, movie_db)
        if m and m not in liked_found:
            liked_found.append(m)
    for r in raw_disliked:
        m = _lenient_match(r, movie_db)
        if m and m not in disliked_found:
            disliked_found.append(m)

    liked_genres = _extract_sentiment_genres(user_input, positive=True)
    disliked_genres = _extract_sentiment_genres(user_input, positive=False)
    requested_genres = [
        genre for genre in _extract_genres(user_input)
        if genre not in liked_genres and genre not in disliked_genres
    ]

    result = {
        "genres_requested":    requested_genres,
        "actors_requested":    _extract_actors(user_input, movie_db),
        "directors_requested": _extract_directors_regex(user_input, movie_db),
        "plot_description":    _extract_plot_description(user_input),
        "plot_keywords":       _extract_plot_keywords(user_input),
        "liked_found":         liked_found,
        "disliked_found":      disliked_found,
        "liked_genres":        liked_genres,
        "disliked_genres":     disliked_genres,
        "soft_disliked_genres": [],          # ← ADD THIS LINE
        "sentiment_last_recs": None,
        "wants_different":     _check_wants_different(user_input),
        "parse_method":        "regex",
    }
    return _postprocess_intent(result, user_input)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — LLM CLASSIFIER (PRIMARY)
# ══════════════════════════════════════════════════════════════════════════════

_CLASSIFIER_SYSTEM = """\
You are a precise, agentic intent classification engine for a movie recommendation system.
Your ONLY job is to analyse the user message and return a single JSON object.
Return ONLY valid JSON — no preamble, no explanation, no markdown fences, no comments.

JSON schema (return ALL keys, even if empty/null):
{
  "genres_requested":      [],
  "actors_requested":      [],
  "directors_requested":   [],
  "plot_description":      "",
  "plot_keywords":         [],
  "liked_found":           [],
  "disliked_found":        [],
  "liked_genres":          [],
  "disliked_genres":       [],
  "soft_disliked_genres":  [],
  "sentiment_last_recs":   null,
  "wants_different":       false
}

FIELD RULES:

Canonical genre names only:
  Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family,
  Fantasy, History, Horror, Music, Mystery, Romance, Science Fiction, Thriller, War, Western

genres_requested — genres the user WANTS recommendations for in this turn.
  Do NOT put hated/blocked genres here.
  "I want action" → ["Action"]
  "fuck musicals" → []

liked_genres — genres the user explicitly praised or wants as ongoing preference.
  "I love action movies" → ["Action"]

disliked_genres — genres the user EXPLICITLY and STRONGLY rejected or wants BLOCKED permanently.
  ONLY use this for clear hate/block language:
  "I hate horror", "no western movies ever", "fuck musicals", "never recommend sci-fi again"
  → disliked_genres = ["Horror"] / ["Western"] / ["Music"] / ["Science Fiction"]

  DO NOT use disliked_genres when the user dislikes a SPECIFIC MOVIE for having too much of a genre.
  "I don't like Event Horizon since it's too much sci-fi" → disliked_found=["Event Horizon"], soft_disliked_genres=["Science Fiction"]
  "that movie had too much comedy for me" → disliked_found=["<movie>"], soft_disliked_genres=["Comedy"]
  The movie is erased. The genre is only softly penalised — NOT blocked.

soft_disliked_genres — genres the user found too prominent in a specific movie they disliked,
  but has NOT explicitly blocked the genre itself.
  Examples:
  "too much sci-fi" about a movie → soft_disliked_genres=["Science Fiction"]
  "Event Horizon is too sci-fi" → soft_disliked_genres=["Science Fiction"]
  "that was too action-heavy" → soft_disliked_genres=["Action"]
  This decays the genre weight slightly but does NOT add it to the blocked list.
  The user can still request this genre and receive recommendations.

actors_requested — actor names mentioned by user.
  Only people, never teams/franchises/genres.
  Do NOT infer actors from a movie title unless user explicitly asks for that actor.

directors_requested — director names from phrases like:
  "directed by X", "films by X", "movies by X", "X's films"
  "I love Nolan's work" → "Christopher Nolan"
  Do NOT infer from liked movie titles unless explicitly asked.

plot_description — full natural language description of desired plot.
  "movies where a man loses his memory" → "a man loses his memory"
  Leave "" if no plot was described.

plot_keywords — short thematic keywords from the query (max 5).
  Only when plot_description is non-empty.

liked_found — titles user expressed POSITIVE sentiment about:
  "I loved X", "X slapped", "X was fire", "X blew me away", "more like X"
  Resolve positional refs ("the second one") using LAST_RECOMMENDATIONS.
  Do NOT copy titles from USER_PROFILE or LAST_RECOMMENDATIONS unless current message clearly praises them.

disliked_found — titles user expressed NEGATIVE sentiment about:
  "I hated X", "X wasn't for me", "X was boring", "I didn't like X"
  "I don't like Event Horizon since it's too much sci-fi" → disliked_found=["Event Horizon"]
  Do NOT copy old disliked titles from USER_PROFILE/RECENT_CONVERSATION.

sentiment_last_recs — overall feeling about the LAST SHOWN RECOMMENDATIONS batch:
  "positive", "negative", "neutral", or null

wants_different — true only if user wants a fresh batch different from the last.
  "something different", "try again", "none of these", "can you do better"

Return ONLY the JSON. No other text whatsoever."""


def _build_prompt(
    user_input: str,
    last_recs: list[str],
    history: list[dict],
    state: dict | None = None,
) -> str:
    history_lines = []
    for turn in history[-4:]:
        role    = turn.get("role", "")
        content = str(turn.get("content", ""))[:400]
        history_lines.append(f"{role.upper()}: {content}")

    return (
        f"LAST_RECOMMENDATIONS: {json.dumps(last_recs)}\n\n"
        f"USER_PROFILE_DO_NOT_COPY_AS_NEW_INTENT: {json.dumps(state or {})}\n\n"
        f"RECENT_CONVERSATION:\n" + "\n".join(history_lines) + "\n\n"
        f"USER_MESSAGE: {user_input}\n\n"
        f"Return the JSON classification now."
    )


def _canonicalize_genres(raw_genres: list) -> list[str]:
    found = []
    for raw in raw_genres or []:
        raw_s = str(raw).strip()
        if not raw_s:
            continue
        genre = None
        for canonical in CANONICAL_GENRES:
            if raw_s.lower() == canonical.lower():
                genre = canonical
                break
        if genre is None:
            genre = _canonical_genre_from_keyword(raw_s)
        if genre and genre not in found:
            found.append(genre)
    return found


def _canonical_people(raw_people: list, movie_db: pd.DataFrame, column: str, user_input: str) -> list[str]:
    names: dict[str, str] = {}
    for people in movie_db[column]:
        for name in (people or []):
            names.setdefault(name.lower(), name)

    found = []
    text_tokens = set(re.findall(r"[a-z0-9]+", user_input.lower()))
    for raw in raw_people or []:
        raw_s = str(raw).strip().lower()
        if not raw_s:
            continue
        raw_tokens = {
            tok for tok in re.findall(r"[a-z0-9]+", raw_s)
            if len(tok) >= 3
        }
        if raw_tokens and not (raw_tokens & text_tokens):
            continue
        match = None
        for key, name in names.items():
            if raw_s == key or raw_s in key or key in raw_s:
                match = name
                break
        if match and match not in found:
            found.append(match)
    return found


def _genre_is_grounded(genre: str, user_input: str) -> bool:
    text = user_input.lower()
    keys = [genre.lower(), *[kw.lower() for kw in GENRE_KEYWORDS.get(genre, [])]]
    return any(re.search(r"\b" + re.escape(key) + r"\b", text) for key in keys)


def _grounded_genres(genres: list[str], user_input: str) -> list[str]:
    return [genre for genre in genres if _genre_is_grounded(genre, user_input)]


def _has_positive_preference(text: str) -> bool:
    return bool(re.search(
        r"\b(?:love|loved|like|liked|enjoy|enjoyed|into|fan of|gotta love|gott love)\b",
        text,
        re.IGNORECASE,
    ))


def _postprocess_intent(result: dict, user_input: str) -> dict:
    """
    Deterministic guardrails over LLM output.
    LLM decides intent; this prevents stale context and missed state updates.
    """
    # Asking for a genre while saying "I like them" should persist preference.
    if _has_positive_preference(user_input):
        for genre in result.get("genres_requested", []):
            if genre not in result["liked_genres"]:
                result["liked_genres"].append(genre)

    # LLM often marks fresh topical requests as "wants different". Keep only
    # explicit dissatisfaction or explicit ask for another batch.
    explicit_diff = _check_wants_different(user_input)
    if not explicit_diff and result.get("sentiment_last_recs") != "negative":
        result["wants_different"] = False

    return result


def _title_reference_is_grounded(raw_title: str, user_input: str, last_recs: list[str]) -> bool:
    """
    Reject stale LLM carryover from profile/history.
    Accept only explicit title text or positional/deictic references to last recs.
    """
    raw = raw_title.strip().lower()
    text = user_input.lower()
    if raw and (raw in text or text in raw):
        return True

    raw_tokens = {
        tok for tok in re.findall(r"[a-z0-9]+", raw)
        if len(tok) >= 4 and tok not in {"movie", "film"}
    }
    text_tokens = set(re.findall(r"[a-z0-9]+", text))
    if raw_tokens & text_tokens:
        return True

    for rec in last_recs:
        rec_l = str(rec).lower()
        if raw == rec_l and rec_l in text:
            return True

    has_positional_reference = bool(re.search(
        r"\b(first|second|third|fourth|fifth|last one|top one|#\d|\d+(?:st|nd|rd|th))\b",
        text,
        re.IGNORECASE,
    ))
    has_single_item_reference = (
        len(last_recs) == 1
        and bool(re.search(r"\b(this|that|it)\b", text, re.IGNORECASE))
    )
    return bool(
        (has_positional_reference or has_single_item_reference)
        and raw in {str(rec).lower() for rec in last_recs}
    )


def _llm_parse(
    user_input:  str,
    movie_db:    pd.DataFrame,
    client:      Groq,
    model:       str,
    last_recs:   list[str],
    history:     list[dict],
    state:       dict | None = None,
) -> dict:
    prompt = _build_prompt(user_input, last_recs, history, state)

    resp = client.chat.completions.create(
        model                = model,
        messages             = [
            {"role": "system", "content": _CLASSIFIER_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature          = 0,
        max_completion_tokens= 600,
        top_p                = 1,
        stream               = False,
    )

    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",          "", raw)
    parsed: dict = json.loads(raw)   # raises on bad JSON → caller falls back to regex

    result = {
        "genres_requested":    _grounded_genres(...),
        "actors_requested":    _canonical_people(...),
        "directors_requested": _canonical_people(...),
        "plot_description":    parsed.get("plot_description", "") or "",
        "plot_keywords":       parsed.get("plot_keywords", []),
        "liked_found":         [],
        "disliked_found":      [],
        "liked_genres":        _grounded_genres(...),
        "disliked_genres":     _grounded_genres(...),
        "sentiment_last_recs": parsed.get("sentiment_last_recs", None),
        "wants_different":     bool(parsed.get("wants_different", False)),
        "soft_disliked_genres": _grounded_genres(   # ← ADD THESE 3 LINES
            _canonicalize_genres(parsed.get("soft_disliked_genres", [])),
            user_input,
        ),
        "parse_method":        "llm",
    }

    # Negative genre intent wins. Avoid "fuck musicals" becoming request for Music.
    result["genres_requested"] = [
        genre for genre in result["genres_requested"]
        if genre not in result["disliked_genres"]
    ]

    for raw_t in parsed.get("liked_found", []):
        if not _title_reference_is_grounded(str(raw_t), user_input, last_recs):
            continue
        canon = _lenient_match(str(raw_t), movie_db)
        if canon and canon not in result["liked_found"]:
            result["liked_found"].append(canon)

    for raw_t in parsed.get("disliked_found", []):
        if not _title_reference_is_grounded(str(raw_t), user_input, last_recs):
            continue
        canon = _lenient_match(str(raw_t), movie_db)
        if canon and canon not in result["disliked_found"]:
            result["disliked_found"].append(canon)

    return _postprocess_intent(result, user_input)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — PUBLIC INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def parse_intent(
    user_input:     str,
    movie_db:       pd.DataFrame,
    groq_client:    Groq  | None = None,
    groq_model:     str   | None = None,
    last_recs:      list  | None = None,
    recent_history: list  | None = None,
    user_state:     dict  | None = None,
) -> dict:
    last_recs      = last_recs      or []
    recent_history = recent_history or []

    if groq_client and groq_model:
        try:
            return _llm_parse(
                user_input, movie_db,
                groq_client, groq_model,
                last_recs, recent_history,
                user_state,
            )
        except Exception as exc:
            logger.warning(f"[Intent] LLM parse failed ({exc!r}) — regex fallback.")

    return _regex_parse(user_input, movie_db)
