"use client";

import { useChat } from "@ai-sdk/react";
import { GlassNav } from "@/components/ui/GlassNav";
import { motion, AnimatePresence } from "framer-motion";
import { pageTransition } from "@/lib/animations";
import { GlassCard } from "@/components/ui/GlassCard";
import { ChatWindow } from "@/components/chat/ChatWindow";
import { ChatInput } from "@/components/chat/ChatInput";
import { GlassButton } from "@/components/ui/GlassButton";
import { MessageSquare, Circle } from "lucide-react";

export default function ChatPage() {
  const chatOpts: any = { api: "/api/chat" };
  const { messages, input, handleInputChange, handleSubmit, isLoading } =
    useChat(chatOpts) as any;

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
          {/* Sidebar */}
          <aside className="hidden lg:flex w-72 flex-col gap-4">
            <GlassCard className="p-4 flex flex-col h-full">
              <GlassButton variant="secondary" className="w-full mb-6">
                New Chat
              </GlassButton>

              <div className="flex-1 overflow-y-auto space-y-2 pr-2">
                <div className="p-3 bg-white/5 rounded-xl border-l-2 border-accent-green cursor-pointer shadow-[0_0_10px_rgba(0,255,136,0.1)]">
                  <p className="text-sm text-white font-medium truncate">
                    Forget Conjuring 2
                  </p>
                  <p className="text-xs text-white/40 mt-1">Just now</p>
                </div>
                <div className="p-3 hover:bg-white/5 rounded-xl cursor-pointer transition-colors glass-mask border border-transparent">
                  <p className="text-sm text-white/70 font-medium truncate">
                    Horror Recommendations
                  </p>
                  <p className="text-xs text-white/40 mt-1">2 hours ago</p>
                </div>
              </div>

              <div className="mt-4 pt-4 border-t border-white/10 flex items-center gap-3">
                <div className="relative">
                  <Circle
                    size={12}
                    className="text-accent-green fill-accent-green"
                  />
                  <div className="absolute inset-0 bg-accent-green blur-[4px] rounded-full animate-pulse" />
                </div>
                <span className="text-sm text-white/70">Agent Online</span>
              </div>
            </GlassCard>
          </aside>

          {/* Main Chat Area */}
          <section className="flex-1 flex flex-col min-w-0">
            <GlassCard className="flex-1 flex flex-col p-2 md:p-4 overflow-hidden relative">
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
      </motion.div>
    </AnimatePresence>
  );
}
