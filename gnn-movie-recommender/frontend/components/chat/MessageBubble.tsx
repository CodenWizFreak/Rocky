"use client";

import { motion } from "framer-motion";
import { fadeUp } from "@/lib/animations";
import { cn } from "@/lib/utils";
import { ChatMessage } from "./ChatWindow";

export function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <motion.div
      variants={fadeUp}
      initial="hidden"
      animate="visible"
      className={cn(
        "flex w-full mt-4",
        isUser ? "justify-end" : "justify-start",
      )}
    >
      <div
        className={cn(
          "max-w-[85%] md:max-w-[75%] p-4 rounded-2xl glass-mask",
          isUser
            ? "bg-accent-purple/20 text-white rounded-br-sm border border-accent-purple/30 shadow-[0_0_15px_rgba(168,85,247,0.15)]"
            : "bg-white/5 text-white/90 rounded-bl-sm border-l-2 border-l-accent-green border-y border-r border-y-white/5 border-r-white/5",
        )}
      >
        <div className="text-sm md:text-base leading-relaxed whitespace-pre-wrap font-inter">
          {message.content}
        </div>
      </div>
    </motion.div>
  );
}
