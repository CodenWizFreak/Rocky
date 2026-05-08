"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ClipboardList, Sparkles, Telescope } from "lucide-react";
import type { Recommendation } from "@/lib/api";

interface Props {
  title: string;
  recommendations: Recommendation[];
  loading: boolean;
  variant?: "default" | "before" | "after";
  onExplain?: (movie: Recommendation) => void;
  removedIds?: number[];
  newIds?: number[];
}

export default function RecommendList({
  title,
  recommendations,
  loading,
  variant = "default",
  onExplain,
  removedIds = [],
  newIds = [],
}: Props) {
  const removedSet = new Set(removedIds);
  // In "before" view, find movies that are NOT in the after list
  const beforeRemovedIds = variant === "before"
    ? recommendations.filter((r) => !removedSet.has(r.movie_id)).map((r) => r.movie_id)
    : [];
  const actualRemovedSet = new Set(beforeRemovedIds);
  const newSet = new Set(newIds);

  const headerColor =
    variant === "before"
      ? "text-[var(--text-secondary)]"
      : variant === "after"
      ? "text-[var(--accent-emerald)]"
      : "text-[var(--text-secondary)]";

  const HeaderIcon =
    variant === "before" ? ClipboardList : variant === "after" ? Sparkles : Telescope;

  if (loading) {
    return (
      <div className="glass-card p-5">
        <div className="shimmer h-5 w-40 mb-4" />
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="shimmer h-20 mb-3" />
        ))}
      </div>
    );
  }

  return (
    <div className="glass-card p-5">
      <h3 className={`text-sm font-semibold ${headerColor} mb-4 flex items-center gap-2`}>
        <HeaderIcon className="h-4 w-4 shrink-0 opacity-90" strokeWidth={2} aria-hidden />
        {title}
        <span className="ml-auto text-xs text-[var(--text-muted)] font-normal">
          Top {recommendations.length}
        </span>
      </h3>

      <div className="space-y-3">
        <AnimatePresence mode="popLayout">
          {recommendations.map((rec, i) => {
            const isRemoved =
              variant === "before" && actualRemovedSet.has(rec.movie_id);
            const isNew = variant === "after" && newSet.has(rec.movie_id);

            return (
              <motion.div
                key={rec.movie_id}
                layout
                initial={
                  isNew
                    ? { opacity: 0, x: 30, scale: 0.95 }
                    : { opacity: 0, y: 10 }
                }
                animate={{
                  opacity: isRemoved ? 0.35 : 1,
                  x: 0,
                  y: 0,
                  scale: 1,
                }}
                exit={{ opacity: 0, x: -30, scale: 0.95 }}
                transition={{ delay: i * 0.05, duration: 0.3 }}
                className={`relative p-3 rounded-xl border transition-all cursor-pointer group
                  ${
                    isRemoved
                      ? "bg-rose-500/5 border-rose-500/20 line-through decoration-rose-500/50"
                      : isNew
                      ? "bg-emerald-500/5 border-emerald-500/20"
                      : "bg-[var(--bg-card)] border-[var(--border-subtle)] hover:border-[var(--border-glow)]"
                  }`}
                onClick={() => onExplain?.(rec)}
              >
                {/* Rank badge */}
                <div className="absolute -left-2 -top-2 w-6 h-6 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center text-[10px] font-bold text-white shadow-lg">
                  {i + 1}
                </div>

                {/* New/Removed indicator */}
                {isRemoved && (
                  <div className="absolute right-2 top-2">
                    <span className="text-xs text-rose-400 font-medium">REMOVED</span>
                  </div>
                )}
                {isNew && (
                  <div className="absolute right-2 top-2">
                    <span className="text-xs text-emerald-400 font-medium animate-pulse">NEW</span>
                  </div>
                )}

                <div className="ml-3">
                  <p
                    className={`text-sm font-medium mb-1 ${
                      isRemoved ? "text-rose-400/60" : "text-[var(--text-primary)]"
                    } group-hover:text-[var(--accent-indigo)] transition-colors`}
                  >
                    {rec.title}
                  </p>

                  <div className="flex flex-wrap gap-1 mb-2">
                    {rec.genres.slice(0, 3).map((g) => (
                      <span key={g} className="genre-badge !text-[10px] !py-0">
                        {g}
                      </span>
                    ))}
                    {rec.franchise && (
                      <span className="franchise-badge !text-[10px] !py-0">
                        {rec.franchise}
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-1">
                      <div className="h-1.5 bg-[var(--bg-secondary)] rounded-full overflow-hidden w-16">
                        <motion.div
                          className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-purple-500"
                          initial={{ width: 0 }}
                          animate={{ width: `${Math.min(rec.score * 100, 100)}%` }}
                          transition={{ delay: i * 0.05 + 0.3 }}
                        />
                      </div>
                      <span className="text-[10px] text-[var(--text-muted)]">
                        {rec.score.toFixed(2)}
                      </span>
                    </div>
                    <button
                      className="text-[10px] text-[var(--text-muted)] hover:text-[var(--accent-cyan)] transition-colors opacity-0 group-hover:opacity-100"
                      onClick={(e) => {
                        e.stopPropagation();
                        onExplain?.(rec);
                      }}
                    >
                      Why? →
                    </button>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
  );
}
