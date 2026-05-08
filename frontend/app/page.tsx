"use client";

import React, { useState, useCallback } from "react";
import { AnimatePresence } from "framer-motion";
import { Orbit, Network, X, Ban, AlertTriangle } from "lucide-react";
import UserSelector from "@/components/UserSelector";
import WatchHistory from "@/components/WatchHistory";
import RecommendList from "@/components/RecommendList";
import UnlearnPanel from "@/components/UnlearnPanel";
import ExplainCard from "@/components/ExplainCard";
import GraphViz from "@/components/GraphViz";

import {
  fetchHistory,
  fetchRecommendations,
  fetchDislikes,
  triggerUnlearn,
  clearDislikes,
  type UserInfo,
  type HistoryResponse,
  type Recommendation,
  type DislikesResponse,
} from "@/lib/api";

export default function Home() {
  const [selectedUser, setSelectedUser] = useState<UserInfo | null>(null);
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [beforeRecs, setBeforeRecs] = useState<Recommendation[]>([]);
  const [afterRecs, setAfterRecs] = useState<Recommendation[]>([]);
  const [dislikes, setDislikes] = useState<DislikesResponse | null>(null);
  const [showAfter, setShowAfter] = useState(false);
  const [loading, setLoading] = useState(false);
  const [unlearnLoading, setUnlearnLoading] = useState(false);
  const [unlearnMessage, setUnlearnMessage] = useState("");
  const [explainMovie, setExplainMovie] = useState<Recommendation | null>(null);
  const [showGraph, setShowGraph] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadUserData = useCallback(async (user: UserInfo) => {
    setLoading(true);
    setError(null);
    setShowAfter(false);
    setAfterRecs([]);
    setUnlearnMessage("");
    setExplainMovie(null);

    try {
      const [historyData, recsData, dislikesData] = await Promise.all([
        fetchHistory(user.user_id),
        fetchRecommendations(user.user_id, 10),
        fetchDislikes(user.user_id),
      ]);

      setHistory(historyData);
      setBeforeRecs(recsData.recommendations);
      setDislikes(dislikesData);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load user data");
    } finally {
      setLoading(false);
    }
  }, []);

  const handleSelectUser = (user: UserInfo) => {
    setSelectedUser(user);
    loadUserData(user);
  };

  const handleUnlearn = async (movieId: number, scope: "movie" | "franchise") => {
    if (!selectedUser) return;

    setUnlearnLoading(true);
    setUnlearnMessage("Trimming the edge from the knowledge graph...");

    try {
      await new Promise((r) => setTimeout(r, 500));
      setUnlearnMessage("Recomputing your embedding — Hail Mary pass on the math...");
      await new Promise((r) => setTimeout(r, 400));

      const result = await triggerUnlearn({
        user_id: selectedUser.user_id,
        movie_id: movieId,
        scope,
      });

      setUnlearnMessage("Pulling fresh picks from Rocky...");
      await new Promise((r) => setTimeout(r, 300));

      const newRecs = await fetchRecommendations(selectedUser.user_id, 10);
      const updatedDislikes = await fetchDislikes(selectedUser.user_id);

      setAfterRecs(newRecs.recommendations);
      setDislikes(updatedDislikes);
      setShowAfter(true);
      setUnlearnMessage(result.message);
    } catch (err) {
      setUnlearnMessage(
        `Error: ${err instanceof Error ? err.message : "Unlearning failed"}`
      );
    } finally {
      setUnlearnLoading(false);
    }
  };

  const handleReset = async () => {
    if (!selectedUser) return;
    try {
      await clearDislikes(selectedUser.user_id);
      setShowAfter(false);
      setAfterRecs([]);
      setUnlearnMessage("");
      await loadUserData(selectedUser);
    } catch {
      setError("Failed to reset dislikes");
    }
  };

  return (
    <main className="min-h-screen bg-[var(--bg-primary)]">
      <header className="border-b border-[var(--border-subtle)] bg-[var(--bg-secondary)]/80 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-[1400px] mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[var(--bg-card)] border border-[var(--border-subtle)]">
              <Orbit className="h-5 w-5 text-[var(--accent-cyan)]" strokeWidth={2} aria-hidden />
            </div>
            <div>
              <h1 className="text-xl font-bold gradient-text">Rocky</h1>
              <p className="text-xs text-[var(--text-muted)]">
                Graph recs, selective forgetting — MovieLens 100K
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {selectedUser && (
              <button
                type="button"
                onClick={() => setShowGraph(!showGraph)}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium 
                           bg-[var(--bg-card)] border border-[var(--border-subtle)]
                           hover:border-[var(--accent-cyan)] hover:text-[var(--accent-cyan)]
                           transition-all duration-300"
              >
                {showGraph ? (
                  <>
                    <X className="h-4 w-4" strokeWidth={2} />
                    Close graph
                  </>
                ) : (
                  <>
                    <Network className="h-4 w-4" strokeWidth={2} />
                    Taste web
                  </>
                )}
              </button>
            )}
            <div className="flex items-center gap-2">
              <div className="pulse-dot" />
              <span className="text-xs text-[var(--text-muted)]">Beetlejuice nominal</span>
            </div>
          </div>
        </div>
      </header>

      <AnimatePresence>
        {showGraph && selectedUser && (
          <GraphViz
            userId={selectedUser.user_id}
            onClose={() => setShowGraph(false)}
          />
        )}
      </AnimatePresence>

      <div className="max-w-[1400px] mx-auto px-6 py-8">
        {!selectedUser ? (
          <div className="flex flex-col items-center justify-center min-h-[70vh] gap-8">
            <div className="text-center">
              <h2 className="text-3xl font-bold gradient-text mb-2">
                Choose a subject for the experiment
              </h2>
              <p className="text-[var(--text-secondary)] max-w-lg mx-auto text-sm leading-relaxed">
                Rocky pairs a LightGCN over your knowledge graph with selective forgetting — so you
                can science the recommendations, then tell the model what never to surface again.
                Pick a MovieLens user to start.
              </p>
            </div>
            <UserSelector onSelect={handleSelectUser} />
          </div>
        ) : (
          <div className="grid grid-cols-12 gap-6">
            <div className="col-span-12 lg:col-span-3 space-y-6">
              <div className="glass-card p-5">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-12 h-12 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center text-white font-bold text-lg">
                    {selectedUser.user_id}
                  </div>
                  <div>
                    <p className="font-semibold text-[var(--text-primary)]">
                      User #{selectedUser.user_id}
                    </p>
                    <p className="text-xs text-[var(--text-muted)]">
                      {selectedUser.age}y • {selectedUser.gender === "M" ? "Male" : "Female"} •{" "}
                      {selectedUser.occupation}
                    </p>
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedUser(null);
                      setHistory(null);
                      setBeforeRecs([]);
                      setAfterRecs([]);
                      setShowAfter(false);
                      setUnlearnMessage("");
                    }}
                    className="flex-1 px-3 py-2 rounded-lg text-xs font-medium
                               bg-[var(--bg-card)] border border-[var(--border-subtle)]
                               hover:border-[var(--accent-rose)] hover:text-[var(--accent-rose)]
                               transition-all"
                  >
                    Change user
                  </button>
                  {dislikes && (dislikes.disliked_movies.length > 0 || dislikes.disliked_franchises.length > 0) && (
                    <button
                      type="button"
                      onClick={handleReset}
                      className="flex-1 px-3 py-2 rounded-lg text-xs font-medium
                                 bg-[var(--bg-card)] border border-[var(--border-subtle)]
                                 hover:border-[var(--accent-emerald)] hover:text-[var(--accent-emerald)]
                                 transition-all"
                    >
                      Reset dislikes
                    </button>
                  )}
                </div>
              </div>

              <WatchHistory
                history={history}
                loading={loading}
                onMovieClick={(movie) => {
                  setExplainMovie({
                    movie_id: movie.movie_id,
                    title: movie.title,
                    genres: movie.genres,
                    franchise: movie.franchise ?? null,
                    score: 0,
                    reason: null,
                  });
                }}
              />

              {dislikes && (dislikes.disliked_movies.length > 0 || dislikes.disliked_franchises.length > 0) && (
                <div className="glass-card p-5">
                  <h3 className="text-sm font-semibold text-[var(--accent-rose)] mb-3 flex items-center gap-2">
                    <Ban className="h-4 w-4 shrink-0" strokeWidth={2} aria-hidden />
                    On Rocky&apos;s no-fly list
                  </h3>
                  {dislikes.disliked_franchises.length > 0 && (
                    <div className="mb-2">
                      {dislikes.disliked_franchises.map((f) => (
                        <span key={f} className="franchise-badge mr-1 mb-1 !bg-rose-500/10 !text-rose-400 !border-rose-500/20">
                          {f} franchise
                        </span>
                      ))}
                    </div>
                  )}
                  {dislikes.disliked_movies.map((m) => (
                    <p key={m.movie_id} className="text-xs text-[var(--text-muted)] truncate pl-1 border-l border-rose-500/30">
                      {m.title}
                    </p>
                  ))}
                </div>
              )}
            </div>

            <div className="col-span-12 lg:col-span-6 space-y-6">
              <div className={`grid ${showAfter ? "grid-cols-2" : "grid-cols-1"} gap-6`}>
                <RecommendList
                  title={showAfter ? "Before unlearning" : "Rocky suggests"}
                  recommendations={beforeRecs}
                  loading={loading}
                  variant={showAfter ? "before" : "default"}
                  onExplain={setExplainMovie}
                  removedIds={showAfter ? afterRecs.map((r) => r.movie_id) : []}
                />
                {showAfter && (
                  <RecommendList
                    title="After the amnesia beam"
                    recommendations={afterRecs}
                    loading={false}
                    variant="after"
                    onExplain={setExplainMovie}
                    newIds={afterRecs.filter((r) => !beforeRecs.some((b) => b.movie_id === r.movie_id)).map((r) => r.movie_id)}
                  />
                )}
              </div>

              <AnimatePresence>
                {explainMovie && selectedUser && (
                  <ExplainCard
                    userId={selectedUser.user_id}
                    movie={explainMovie}
                    onClose={() => setExplainMovie(null)}
                  />
                )}
              </AnimatePresence>
            </div>

            <div className="col-span-12 lg:col-span-3">
              <UnlearnPanel
                userId={selectedUser.user_id}
                history={history?.history || []}
                onUnlearn={handleUnlearn}
                loading={unlearnLoading}
                message={unlearnMessage}
              />
            </div>
          </div>
        )}
      </div>

      <AnimatePresence>
        {error && (
          <div className="fixed bottom-6 right-6 glass-card p-4 border-[var(--accent-rose)] max-w-sm z-50">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-[var(--accent-rose)] shrink-0" strokeWidth={2} aria-hidden />
              <p className="text-sm text-[var(--text-primary)]">{error}</p>
              <button
                type="button"
                onClick={() => setError(null)}
                className="ml-auto text-[var(--text-muted)] hover:text-[var(--text-primary)] p-1"
                aria-label="Dismiss"
              >
                <X className="h-4 w-4" strokeWidth={2} />
              </button>
            </div>
          </div>
        )}
      </AnimatePresence>
    </main>
  );
}
