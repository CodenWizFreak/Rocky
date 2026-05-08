"use client";

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Lightbulb, X, Ban } from "lucide-react";
import { fetchExplanation, type Recommendation, type ExplainResponse } from "@/lib/api";

interface Props {
  userId: number;
  movie: Recommendation;
  onClose: () => void;
}

export default function ExplainCard({ userId, movie, onClose }: Props) {
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const data = await fetchExplanation(userId, movie.movie_id);
        setExplanation(data);
      } catch {
        setExplanation(null);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [userId, movie.movie_id]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 10, scale: 0.98 }}
      className="glass-card p-5 border-[var(--accent-cyan)]/30"
    >
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="text-sm font-semibold text-[var(--accent-cyan)] flex items-center gap-2">
            <Lightbulb className="h-4 w-4 shrink-0" strokeWidth={2} aria-hidden />
            Why Rocky ranked this
          </h3>
          <p className="text-xs text-[var(--text-primary)] mt-1 font-medium">
            {movie.title}
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors p-1"
          aria-label="Close"
        >
          <X className="h-4 w-4" strokeWidth={2} />
        </button>
      </div>

      {loading ? (
        <div className="space-y-2">
          <div className="shimmer h-4 w-full" />
          <div className="shimmer h-4 w-3/4" />
        </div>
      ) : explanation ? (
        <div className="space-y-3">
          <div className="p-3 rounded-lg bg-[var(--bg-card)]">
            <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
              {explanation.explanation}
            </p>
          </div>

          {explanation.excluded && explanation.exclusion_reason && (
            <div className="p-3 rounded-lg bg-rose-500/5 border border-rose-500/20">
              <p className="text-xs text-rose-400 leading-relaxed flex items-start gap-2">
                <Ban className="h-3.5 w-3.5 shrink-0 mt-0.5" strokeWidth={2} aria-hidden />
                {explanation.exclusion_reason}
              </p>
            </div>
          )}

          <div className="flex gap-1 flex-wrap">
            {movie.genres.map((g) => (
              <span key={g} className="genre-badge">{g}</span>
            ))}
            {movie.franchise && (
              <span className="franchise-badge">{movie.franchise}</span>
            )}
          </div>
        </div>
      ) : (
        <p className="text-xs text-[var(--text-muted)]">
          {movie.reason || "Strong overlap with your watch history — Rocky ran the numbers."}
        </p>
      )}
    </motion.div>
  );
}
