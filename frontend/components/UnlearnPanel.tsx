"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  Eraser,
  X,
  Loader2,
  Film,
  Layers,
  Trash2,
} from "lucide-react";
import { searchMovies, type MovieInfo, type MovieSearchResult } from "@/lib/api";

interface Props {
  userId: number;
  history: MovieInfo[];
  onUnlearn: (movieId: number, scope: "movie" | "franchise") => void;
  loading: boolean;
  message: string;
}

export default function UnlearnPanel({ userId, history, onUnlearn, loading, message }: Props) {
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<MovieSearchResult[]>([]);
  const [selectedMovie, setSelectedMovie] = useState<MovieSearchResult | null>(null);
  const [scope, setScope] = useState<"movie" | "franchise">("movie");
  const [searching, setSearching] = useState(false);

  const handleSearch = async (query: string) => {
    setSearchQuery(query);
    if (query.length < 2) {
      setSearchResults([]);
      return;
    }

    setSearching(true);
    try {
      const data = await searchMovies(query);
      setSearchResults(data.results.slice(0, 8));
    } catch {
      setSearchResults([]);
    } finally {
      setSearching(false);
    }
  };

  const handleSelectMovie = (movie: MovieSearchResult) => {
    setSelectedMovie(movie);
    setSearchQuery(movie.title);
    setSearchResults([]);
    // Auto-set scope to franchise if movie has one
    if (movie.franchise) {
      setScope("franchise");
    }
  };

  const handleSubmit = () => {
    if (!selectedMovie) return;
    onUnlearn(selectedMovie.movie_id, scope);
    setSelectedMovie(null);
    setSearchQuery("");
  };

  // Quick-select from history
  const topWatched = history
    .filter((m) => m.rating && m.rating >= 3.5)
    .slice(0, 6);

  return (
    <div className="glass-card p-5 sticky top-24">
      <h3 className="text-sm font-semibold text-[var(--accent-rose)] mb-1 flex items-center gap-2">
        <Eraser className="h-4 w-4 shrink-0" strokeWidth={2} aria-hidden />
        Amnesia beam
      </h3>
      <p className="text-[10px] text-[var(--text-muted)] mb-4 leading-relaxed">
        Rocky forgets an edge like it&apos;s a bad astrophage — one movie, or a whole franchise chain.
      </p>

      {/* Search input */}
      <div className="relative mb-3">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => handleSearch(e.target.value)}
          placeholder="Search a title to remove from the signal..."
          className="w-full px-3 py-2.5 rounded-xl text-sm bg-[var(--bg-card)] 
                     border border-[var(--border-subtle)] text-[var(--text-primary)]
                     placeholder-[var(--text-muted)]
                     focus:outline-none focus:border-[var(--accent-rose)] transition-colors"
        />
        {selectedMovie && (
          <button
            onClick={() => {
              setSelectedMovie(null);
              setSearchQuery("");
            }}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-[var(--text-muted)] hover:text-[var(--accent-rose)] p-0.5"
            aria-label="Clear"
          >
            <X className="h-4 w-4" strokeWidth={2} />
          </button>
        )}

        {/* Search dropdown */}
        {searchResults.length > 0 && !selectedMovie && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            className="absolute top-full left-0 right-0 mt-1 bg-[var(--bg-secondary)] border border-[var(--border-subtle)] rounded-xl overflow-hidden z-20 shadow-xl"
          >
            {searchResults.map((movie) => (
              <button
                key={movie.movie_id}
                onClick={() => handleSelectMovie(movie)}
                className="w-full px-3 py-2 text-left hover:bg-[var(--bg-card-hover)] transition-colors flex items-center gap-2"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-[var(--text-primary)] truncate">{movie.title}</p>
                  <div className="flex gap-1 mt-0.5">
                    {movie.genres.slice(0, 2).map((g) => (
                      <span key={g} className="genre-badge !text-[9px] !py-0 !px-1">
                        {g}
                      </span>
                    ))}
                    {movie.franchise && (
                      <span className="franchise-badge !text-[9px] !py-0 !px-1">
                        {movie.franchise}
                      </span>
                    )}
                  </div>
                </div>
              </button>
            ))}
          </motion.div>
        )}
      </div>

      {/* Scope toggle */}
      {selectedMovie && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          className="mb-3"
        >
          <p className="text-[10px] text-[var(--text-muted)] mb-2">Scope (Eridian precision optional):</p>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setScope("movie")}
              className={`flex-1 px-3 py-2 rounded-lg text-xs font-medium border transition-all inline-flex flex-col items-center gap-0.5 ${
                scope === "movie"
                  ? "bg-rose-500/10 border-rose-500/30 text-rose-400"
                  : "bg-[var(--bg-card)] border-[var(--border-subtle)] text-[var(--text-muted)]"
              }`}
            >
              <span className="inline-flex items-center gap-1">
                <Film className="h-3 w-3" strokeWidth={2} />
                This film only
              </span>
            </button>
            <button
              type="button"
              onClick={() => setScope("franchise")}
              disabled={!selectedMovie?.franchise}
              className={`flex-1 px-3 py-2 rounded-lg text-xs font-medium border transition-all inline-flex flex-col items-center gap-0.5 ${
                scope === "franchise"
                  ? "bg-purple-500/10 border-purple-500/30 text-purple-400"
                  : "bg-[var(--bg-card)] border-[var(--border-subtle)] text-[var(--text-muted)]"
              } ${!selectedMovie?.franchise ? "opacity-40 cursor-not-allowed" : ""}`}
            >
              <span className="inline-flex items-center gap-1">
                <Layers className="h-3 w-3" strokeWidth={2} />
                Whole franchise
              </span>
              {selectedMovie?.franchise && (
                <span className="block text-[9px] opacity-70">{selectedMovie.franchise}</span>
              )}
            </button>
          </div>
        </motion.div>
      )}

      {/* Submit button */}
      <button
        type="button"
        onClick={handleSubmit}
        disabled={!selectedMovie || loading}
        className={`w-full py-3 rounded-xl text-sm font-semibold transition-all duration-300 ${
          selectedMovie && !loading
            ? "bg-gradient-to-r from-rose-500 to-pink-500 text-white hover:shadow-lg hover:shadow-rose-500/20 hover:scale-[1.02]"
            : "bg-[var(--bg-card)] text-[var(--text-muted)] cursor-not-allowed"
        }`}
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin" strokeWidth={2} aria-hidden />
            Running the beam...
          </span>
        ) : (
          <span className="inline-flex items-center justify-center gap-2">
            <Trash2 className="h-4 w-4" strokeWidth={2} aria-hidden />
            Apply unlearning
          </span>
        )}
      </button>

      {/* Status message */}
      {message && (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-3 p-3 rounded-lg bg-[var(--bg-card)] border border-[var(--border-subtle)]"
        >
          <p className="text-xs text-[var(--text-secondary)] leading-relaxed">{message}</p>
        </motion.div>
      )}

      {/* Quick select from history */}
      {!selectedMovie && topWatched.length > 0 && (
        <div className="mt-4 pt-4 border-t border-[var(--border-subtle)]">
          <p className="text-[10px] text-[var(--text-muted)] mb-2">Quick from your logbook:</p>
          <div className="flex flex-wrap gap-1.5">
            {topWatched.map((movie) => (
              <button
                key={movie.movie_id}
                onClick={() =>
                  handleSelectMovie({
                    movie_id: movie.movie_id,
                    title: movie.title,
                    genres: movie.genres,
                    franchise: movie.franchise,
                  })
                }
                className="text-[10px] px-2 py-1 rounded-md bg-[var(--bg-card)] border border-[var(--border-subtle)]
                           text-[var(--text-muted)] hover:border-[var(--accent-rose)] hover:text-[var(--accent-rose)]
                           transition-all truncate max-w-[120px]"
              >
                {movie.title.replace(/ \(\d{4}\)$/, "")}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
