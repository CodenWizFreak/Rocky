"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { GlassNav } from "@/components/ui/GlassNav";
import { motion, AnimatePresence } from "framer-motion";
import { pageTransition } from "@/lib/animations";
import { GlassCard } from "@/components/ui/GlassCard";
import { ChatWindow, ChatMessage } from "@/components/chat/ChatWindow";
import { ChatInput } from "@/components/chat/ChatInput";
import { GlassButton } from "@/components/ui/GlassButton";
import { GnnVisualizer } from "@/components/chat/GnnVisualizer";
import { MessageSquare, Circle, GitGraph, Trash2, Loader2 } from "lucide-react";

interface GraphData {
  nodes: any[];
  edges: any[];
  genre_weights: Record<string, number>;
  stats: {
    liked_count: number;
    disliked_count: number;
    liked_genres: string[];
    disliked_genres: string[];
  };
}

const EMPTY_GRAPH: GraphData = {
  nodes: [],
  edges: [],
  genre_weights: {},
  stats: { liked_count: 0, disliked_count: 0, liked_genres: [], disliked_genres: [] },
};

export default function ChatPage() {
  const [messages, setMessages]             = useState<ChatMessage[]>([]);
  const [input, setInput]                   = useState("");
  const [isLoading, setIsLoading]           = useState(false);
  const [showVisualizer, setShowVisualizer] = useState(false);
  const [graphData, setGraphData]           = useState<GraphData>(EMPTY_GRAPH);
  const [isOnline, setIsOnline]             = useState(false);
  const [isResetting, setIsResetting]       = useState(false);
  const abortRef                            = useRef<AbortController | null>(null);

  const fetchGraph = useCallback(async () => {
    try {
      const r = await fetch("/api/graph", { cache: "no-store" });
      if (r.ok) setGraphData(await r.json());
    } catch {}
  }, []);

  useEffect(() => {
    const init = async () => {
      try {
        const greetRes = await fetch("/api/greet", { cache: "no-store" });
        if (greetRes.ok) {
          setIsOnline(true);
          const data = await greetRes.json();
          if (data.greeting) {
            setMessages([{ id: "greeting", role: "assistant", content: data.greeting }]);
          }
          await fetchGraph();
        }
      } catch {
        setIsOnline(false);
      }
    };
    init();
  }, [fetchGraph]);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
  };

  const handleSubmit = useCallback(async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || isLoading) return;

    const userMsg: ChatMessage = { id: `u-${Date.now()}`, role: "user", content: text };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    const assistantId = `a-${Date.now()}`;
    setMessages(prev => [...prev, { id: assistantId, role: "assistant", content: "" }]);

    try {
      abortRef.current = new AbortController();
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
        signal: abortRef.current.signal,
      });

      if (!res.ok || !res.body) {
        setMessages(prev =>
          prev.map(m => m.id === assistantId
            ? { ...m, content: "⚠️ Failed to reach the backend. Make sure the Python server is running on port 8000." }
            : m
          )
        );
        setIsLoading(false);
        return;
      }

      const reader    = res.body.getReader();
      const decoder   = new TextDecoder();
      let buffer      = "";
      let fullContent = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;
          try {
            const event = JSON.parse(raw);
            if (event.type === "token") {
              fullContent += event.content;
              setMessages(prev =>
                prev.map(m => m.id === assistantId ? { ...m, content: fullContent } : m)
              );
            } else if (event.type === "graph") {
              setGraphData(event.data);
            }
          } catch {}
        }
      }
    } catch (err: any) {
      if (err.name !== "AbortError") {
        setMessages(prev =>
          prev.map(m => m.id === assistantId
            ? { ...m, content: "⚠️ Connection error. Is the Python backend running on port 8000?" }
            : m
          )
        );
      }
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading]);

  const handleReset = async () => {
    if (!confirm("Reset all preferences and conversation history?")) return;
    setIsResetting(true);
    try {
      await fetch("/api/reset", { method: "POST" });
      setMessages([]);
      setGraphData(EMPTY_GRAPH);
      const greetRes = await fetch("/api/greet", { cache: "no-store" });
      if (greetRes.ok) {
        const data = await greetRes.json();
        if (data.greeting) {
          setMessages([{ id: `greeting-${Date.now()}`, role: "assistant", content: data.greeting }]);
        }
      }
    } finally {
      setIsResetting(false);
    }
  };

  const stats = graphData?.stats;

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key="chat"
        variants={pageTransition}
        initial="initial"
        animate="animate"
        exit="exit"
        className="h-screen flex flex-col pt-4 overflow-hidden"
      >
        <GlassNav />

        <main className="flex-1 max-w-7xl w-full mx-auto p-4 flex flex-col lg:flex-row gap-6 mt-4 min-h-0 container">
          <aside className="hidden lg:flex w-72 flex-col gap-4">
            <GlassCard className="p-4 flex flex-col h-full">
              <GlassButton
                variant="secondary"
                className="w-full mb-6"
                onClick={handleReset}
                disabled={isResetting}
              >
                {isResetting ? (
                  <span className="flex items-center gap-2 justify-center">
                    <Loader2 size={14} className="animate-spin" />
                    Resetting…
                  </span>
                ) : (
                  <span className="flex items-center gap-2 justify-center">
                    <Trash2 size={14} />
                    New Chat
                  </span>
                )}
              </GlassButton>

              <div className="mb-4 p-3 bg-white/5 rounded-xl border border-white/10 text-xs font-inter space-y-2">
                <div className="text-white/40 uppercase tracking-wider text-[10px] font-medium mb-2">GNN State</div>
                <div className="flex justify-between items-center">
                  <span className="text-white/60">Liked</span>
                  <span className="text-accent-green font-medium tabular-nums">{stats?.liked_count ?? 0}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-white/60">Erased</span>
                  <span className="text-red-400 font-medium tabular-nums">{stats?.disliked_count ?? 0}</span>
                </div>

                {stats?.liked_genres && stats.liked_genres.length > 0 && (
                  <div className="pt-2 border-t border-white/10">
                    <div className="text-white/40 text-[10px] mb-1.5">Boosted genres</div>
                    <div className="flex flex-wrap gap-1">
                      {stats.liked_genres.slice(0, 5).map(g => (
                        <span key={g} className="px-1.5 py-0.5 bg-accent-green/15 text-accent-green rounded text-[10px]">{g}</span>
                      ))}
                    </div>
                  </div>
                )}

                {stats?.disliked_genres && stats.disliked_genres.length > 0 && (
                  <div className="pt-2 border-t border-white/10">
                    <div className="text-white/40 text-[10px] mb-1.5">Blocked genres</div>
                    <div className="flex flex-wrap gap-1">
                      {stats.disliked_genres.slice(0, 5).map(g => (
                        <span key={g} className="px-1.5 py-0.5 bg-red-500/15 text-red-400 rounded text-[10px]">{g}</span>
                      ))}
                    </div>
                  </div>
                )}

                {graphData?.genre_weights && Object.keys(graphData.genre_weights).length > 0 && (
                  <div className="pt-2 border-t border-white/10">
                    <div className="text-white/40 text-[10px] mb-1.5">Edge weights</div>
                    <div className="space-y-1">
                      {Object.entries(graphData.genre_weights)
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 5)
                        .map(([genre, w]) => (
                          <div key={genre} className="flex items-center gap-2">
                            <div className="flex-1 h-1 bg-white/10 rounded-full overflow-hidden">
                              <div
                                className="h-full rounded-full transition-all duration-500"
                                style={{
                                  width: `${Math.min(100, (w / 1.5) * 100)}%`,
                                  backgroundColor: w >= 1.0 ? "#00ff88" : "#ef4444",
                                }}
                              />
                            </div>
                            <span className="text-white/50 text-[10px] w-16 truncate">{genre}</span>
                            <span className={`text-[10px] font-mono ${w >= 1.0 ? "text-accent-green" : "text-red-400"}`}>
                              {w.toFixed(2)}
                            </span>
                          </div>
                        ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="flex-1" />

              <div className="mt-4 pt-4 border-t border-white/10 flex items-center gap-3">
                <div className="relative">
                  <Circle size={12} className={isOnline ? "text-accent-green fill-accent-green" : "text-red-400 fill-red-400"} />
                  {isOnline && <div className="absolute inset-0 bg-accent-green blur-[4px] rounded-full animate-pulse" />}
                </div>
                <span className="text-sm text-white/70">{isOnline ? "Agent Online" : "Backend Offline"}</span>
              </div>
            </GlassCard>
          </aside>

          <section className="flex-1 flex flex-col min-w-0">
            <GlassCard className="flex-1 flex flex-col p-2 md:p-4 overflow-hidden relative">
              <div className="flex items-center justify-between mb-3 shrink-0">
                <div className="flex items-center gap-2 text-white/40 text-xs font-inter">
                  <MessageSquare size={13} />
                  <span>GNN Movie Assistant</span>
                </div>
                <button
                  onClick={() => setShowVisualizer(true)}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 hover:border-accent-purple/50 text-white/60 hover:text-white transition-all duration-200 text-xs font-inter group"
                >
                  <GitGraph size={13} className="text-accent-purple group-hover:scale-110 transition-transform" />
                  <span>See Live Visualizer</span>
                  <div className="w-1.5 h-1.5 rounded-full bg-accent-green animate-pulse ml-0.5" />
                </button>
              </div>

              <ChatWindow messages={messages} isLoading={isLoading} />

              <div className="mt-2 shrink-0">
                <ChatInput
                  input={input}
                  handleInputChange={handleInputChange}
                  handleSubmit={handleSubmit}
                  isLoading={isLoading}
                />
              </div>
            </GlassCard>
          </section>
        </main>

        <AnimatePresence>
          {showVisualizer && (
            <GnnVisualizer graphData={graphData} onClose={() => setShowVisualizer(false)} />
          )}
        </AnimatePresence>
      </motion.div>
    </AnimatePresence>
  );
}