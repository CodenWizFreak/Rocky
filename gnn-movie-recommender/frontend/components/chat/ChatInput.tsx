"use client";

import { GlassButton } from "../ui/GlassButton";
import { Send, Paperclip, Loader2 } from "lucide-react";
import React, { useRef, useEffect } from "react";

interface ChatInputProps {
  input: string;
  handleInputChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  handleSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
  isLoading: boolean;
}

export function ChatInput({
  input,
  handleInputChange,
  handleSubmit,
  isLoading,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 120) + "px";
    }
  }, [input]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if ((input || "").trim() && !isLoading) {
        // Create synthetic form event event to call handleSubmit
        const form = e.currentTarget.form;
        if (form) form.requestSubmit();
      }
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="relative flex items-end gap-2 bg-white/5 backdrop-blur-xl rounded-2xl p-2 border border-white/10 focus-within:border-accent-violet transition-colors duration-500 overflow-hidden group glass-mask"
    >
      {/* Animated focus border effect */}
      <div className="absolute inset-0 pointer-events-none opacity-0 group-focus-within:opacity-100 transition-opacity duration-1000">
        <div className="absolute inset[-1px] rounded-2xl bg-gradient-to-r from-accent-violet to-accent-green animate-[aurora_4s_infinite] -z-10 [mask-image:linear-gradient(#fff_0_0)_content-box,_linear-gradient(#fff_0_0)] [mask-composite:exclude] p-[1px]" />
      </div>

      <button
        type="button"
        className="p-3 text-white/50 hover:text-white hover:bg-white/5 rounded-xl transition-colors shrink-0"
      >
        <Paperclip size={20} />
      </button>

      <textarea
        ref={textareaRef}
        value={input || ""}
        onChange={handleInputChange}
        onKeyDown={onKeyDown}
        placeholder="Type a message..."
        rows={1}
        className="flex-1 bg-transparent border-none outline-none resize-none py-3 text-white placeholder:text-white/30 text-sm md:text-base font-inter max-h-[120px] scrollbar-thin overflow-y-auto"
      />

      <GlassButton
        type="submit"
        size="sm"
        className="w-12 h-12 p-0 rounded-xl shrink-0 flex items-center justify-center mb-0.5 mr-0.5"
        disabled={!(input || "").trim() || isLoading}
      >
        {isLoading ? (
          <Loader2 size={18} className="animate-spin text-bg-base" />
        ) : (
          <Send size={18} className="text-bg-base ml-1" />
        )}
      </GlassButton>

      <div className="absolute bottom-2 right-16 text-xs text-white/30 pointer-events-none hidden sm:block">
        {(input || "").length}
      </div>
    </form>
  );
}
