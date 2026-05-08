"use client";

import React from "react";
import { motion } from "framer-motion";
import { Clapperboard } from "lucide-react";
import type { HistoryResponse, MovieInfo } from "@/lib/api";

interface Props {
  history: HistoryResponse | null;
  loading: boolean;
  onMovieClick?: (movie: MovieInfo) => void;
}

export default function WatchHistory({ history, loading, onMovieClick }: Props) {
  if (loading) {
    return (
      <div className="glass-card p-5">
        <div className="shimmer h-5 w-32 mb-4" />
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="shimmer h-12 mb-2" />
        ))}
      </div>
    );
  }

  if (!history || history.history.length === 0) {
    return (
      <div className="glass-card p-5">
        <h3 className="text-sm font-semibold text-[var(--text-secondary)] mb-3 flex items-center gap-2">
          <Clapperboard className="h-4 w-4 opacity-80" strokeWidth={2} aria-hidden />
          Watch history
        </h3>
        <p className="text-xs text-[var(--text-muted)]">No history available</p>
      </div>
    );
  }

  // Show top 20 by rating
  const topHistory = history.history.slice(0, 20);

  const ratingColor = (r: number) => {
    if (r >= 5) return "text-yellow-400";
    if (r >= 4) return "text-emerald-400";
    if (r >= 3) return "text-blue-400";
    return "text-gray-400";
  };

  return (
    <div className="glass-card p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-[var(--text-secondary)] flex items-center gap-2">
          <Clapperboard className="h-4 w-4 opacity-80" strokeWidth={2} aria-hidden />
          Watch history
        </h3>
        <span className="text-xs text-[var(--text-muted)]">
          {history.total_watched} movies
        </span>
      </div>

      <div className="space-y-1 max-h-[400px] overflow-y-auto pr-1">
        {topHistory.map((movie, i) => (
          <motion.div
            key={movie.movie_id}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.03 }}
            className="flex items-center gap-2 p-2 rounded-lg hover:bg-[var(--bg-card-hover)] 
                       cursor-pointer transition-colors group"
            onClick={() => onMovieClick?.(movie)}
          >
            <div className="flex-1 min-w-0">
              <p className="text-xs text-[var(--text-primary)] truncate group-hover:text-[var(--accent-indigo)] transition-colors">
                {movie.title}
              </p>
              <div className="flex gap-1 mt-0.5 flex-wrap">
                {movie.genres.slice(0, 2).map((g) => (
                  <span key={g} className="genre-badge !text-[10px] !py-0 !px-1.5">
                    {g}
                  </span>
                ))}
                {movie.franchise && (
                  <span className="franchise-badge !text-[10px] !py-0 !px-1.5">
                    {movie.franchise}
                  </span>
                )}
              </div>
            </div>
            <div className="flex items-center gap-1 shrink-0">
              <span className={`text-xs font-medium ${ratingColor(movie.rating || 0)}`}>
                {"★".repeat(Math.round(movie.rating || 0))}
              </span>
              <span className="text-[10px] text-[var(--text-muted)]">{movie.rating}</span>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
