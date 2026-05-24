"use client";

import { MessageBubble } from "./MessageBubble";
import { useEffect, useRef } from "react";
import { motion } from "framer-motion";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
}

export function ChatWindow({
  messages,
  isLoading,
}: {
  messages: ChatMessage[];
  isLoading: boolean;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  return (
    <div
      ref={scrollRef}
      className="flex-1 overflow-y-auto pr-2 scroll-smooth pb-4"
    >
      {messages.length === 0 ? (
        <div className="h-full flex flex-col items-center justify-center text-center opacity-50 relative pointer-events-none">
          <div className="w-16 h-16 rounded-full bg-accent-glow blur-3xl absolute" />
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="relative z-10"
          >
            <p className="text-xl font-space-grotesk text-white">
              Ask me anything...
            </p>
            <div className="h-[2px] w-24 mx-auto mt-4 bg-gradient-to-r from-transparent via-accent-purple to-transparent animate-[aurora_3s_infinite]" />
          </motion.div>
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {messages.map((m) => (
            <MessageBubble key={m.id} message={m} />
          ))}
          {isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex mt-6 ml-2 gap-1.5"
            >
              <span
                className="w-2 h-2 rounded-full bg-accent-green animate-typing-dot"
                style={{ animationDelay: "0ms" }}
              />
              <span
                className="w-2 h-2 rounded-full bg-accent-green animate-typing-dot"
                style={{ animationDelay: "200ms" }}
              />
              <span
                className="w-2 h-2 rounded-full bg-accent-green animate-typing-dot"
                style={{ animationDelay: "400ms" }}
              />
            </motion.div>
          )}
        </div>
      )}
    </div>
  );
}
