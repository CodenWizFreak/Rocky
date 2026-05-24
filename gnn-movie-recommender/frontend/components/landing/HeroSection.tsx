"use client";

import { motion } from "framer-motion";
import { fadeUp, stagger } from "@/lib/animations";
import { GlassButton } from "../ui/GlassButton";
import { GlassCard } from "../ui/GlassCard";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

export function HeroSection() {
  const cardRef = useRef<HTMLDivElement>(null);
  const sectionRef = useRef<HTMLElement>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [bubbles, setBubbles] = useState([false, false, false]);

  // Parallax: track mouse relative to the section, not the card
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!sectionRef.current) return;
      const { left, top, width, height } =
        sectionRef.current.getBoundingClientRect();
      const x = ((e.clientX - left) / width - 0.5) * 18;
      const y = ((e.clientY - top) / height - 0.5) * 12;
      setMousePos({ x, y });
    };
    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  // Stagger the chat bubbles appearing in sequence
  useEffect(() => {
    const timers = [
      setTimeout(() => setBubbles([true, false, false]), 600),
      setTimeout(() => setBubbles([true, true, false]), 1400),
      setTimeout(() => setBubbles([true, true, true]), 2200),
    ];
    return () => timers.forEach(clearTimeout);
  }, []);

  return (
    <section
      ref={sectionRef}
      className="relative min-h-[92vh] flex flex-col justify-center items-center px-4 md:px-8 pt-28 pb-16 overflow-hidden"
    >
      {/* Ambient centre glow */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 z-0 flex items-center justify-center"
      >
        <div className="w-[700px] h-[400px] rounded-full bg-accent-glow opacity-[0.06] blur-[140px]" />
      </div>

      <motion.div
        variants={stagger}
        initial="hidden"
        animate="visible"
        className="relative z-10 w-full max-w-6xl mx-auto flex flex-col lg:flex-row items-center gap-10 lg:gap-16"
      >
        {/* ── Left column ── */}
        <div className="flex-1 flex flex-col items-center lg:items-start text-center lg:text-left gap-6 w-full">
          {/* Badge */}
          <motion.div variants={fadeUp} className="inline-flex">
            <span className="px-4 py-1.5 rounded-full text-xs font-semibold tracking-widest uppercase text-accent-green bg-accent-green/10 border border-accent-green/25 shadow-[0_0_18px_rgba(0,255,136,0.15)]">
              AI-Powered Architecture
            </span>
          </motion.div>

          {/* Headline */}
          <motion.h1
            variants={fadeUp}
            className="font-space-grotesk text-5xl md:text-6xl lg:text-[76px] font-bold leading-[1.05] tracking-tight"
          >
            Recommendations
            <br className="hidden sm:block" /> that can{" "}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-accent-green via-accent-glow to-accent-purple">
              forget.
            </span>
          </motion.h1>

          {/* Subheadline */}
          <motion.p
            variants={fadeUp}
            className="text-white/55 text-lg md:text-xl max-w-xl font-light leading-relaxed"
          >
            A knowledge-graph powered recommendation system that selectively
            forgets user preferences on demand — no full model retraining
            required.
          </motion.p>

          {/* CTAs */}
          <motion.div
            variants={fadeUp}
            className="flex flex-col sm:flex-row items-center gap-3 mt-2 w-full sm:w-auto"
          >
            <Link href="/chat" className="w-full sm:w-auto">
              <GlassButton size="lg" className="w-full sm:w-auto">
                View Demo
              </GlassButton>
            </Link>
            <Link href="/architecture" className="w-full sm:w-auto">
              <GlassButton variant="ghost" size="lg" className="w-full sm:w-auto">
                Read Architecture
              </GlassButton>
            </Link>
          </motion.div>
        </div>

        {/* ── Right column — floating chat card ── */}
        <motion.div
          variants={fadeUp}
          className="flex-1 w-full max-w-[480px] lg:max-w-full mx-auto"
          style={{ perspective: "1000px" }}
        >
          <motion.div
            ref={cardRef}
            animate={{
              x: mousePos.x,
              y: mousePos.y,
              rotateX: -mousePos.y * 0.5,
              rotateY: mousePos.x * 0.5,
            }}
            transition={{ type: "spring", stiffness: 60, damping: 18 }}
            className="animate-float"
          >
            <Link href="/chat" className="block">
              <GlassCard glow hover className="p-5 md:p-7">
                {/* Fake terminal top bar */}
                <div className="flex items-center gap-2 border-b border-white/10 pb-4 mb-5">
                  <span className="w-3 h-3 rounded-full bg-red-500/80" />
                  <span className="w-3 h-3 rounded-full bg-yellow-500/80" />
                  <span className="w-3 h-3 rounded-full bg-green-500/80" />
                  <span className="ml-3 text-xs text-white/40 font-mono tracking-wider">
                    unlearn.sh
                  </span>
                </div>

                {/* Chat bubbles */}
                <div className="space-y-4 min-h-[180px]">
                  {/* User bubble */}
                  <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={bubbles[0] ? { opacity: 1, y: 0 } : {}}
                    transition={{ duration: 0.4 }}
                    className="flex gap-3 items-start"
                  >
                    <div className="w-8 h-8 rounded-full bg-accent-purple/20 border border-accent-purple/30 flex items-center justify-center shrink-0 mt-0.5">
                      <span className="text-xs text-accent-purple font-bold">U</span>
                    </div>
                    <div className="bg-white/[0.06] border border-white/10 rounded-2xl rounded-tl-sm px-4 py-3 text-sm text-white/80 leading-relaxed max-w-[85%]">
                      Never show me Horror movies again. I hate Conjuring 2.
                    </div>
                  </motion.div>

                  {/* Agent bubble */}
                  <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={bubbles[1] ? { opacity: 1, y: 0 } : {}}
                    transition={{ duration: 0.4 }}
                    className="flex gap-3 items-start justify-end"
                  >
                    <div className="bg-white/[0.06] border border-accent-green/25 border-l-2 border-l-accent-green rounded-2xl rounded-tr-sm px-4 py-3 text-sm text-white/80 leading-relaxed max-w-[85%]">
                      <p>Edge deleted from Knowledge Graph.</p>
                      <p>Shard 3 retraining initiated.</p>
                      <p className="text-accent-green mt-2 font-medium">
                        ✓ Franchise removed from recommendations.
                      </p>
                    </div>
                    <div className="w-8 h-8 rounded-full bg-accent-green/20 border border-accent-green/30 flex items-center justify-center shrink-0 mt-0.5">
                      <span className="text-xs text-accent-green font-bold">A</span>
                    </div>
                  </motion.div>

                  {/* Typing indicator */}
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={bubbles[2] ? { opacity: 1 } : {}}
                    transition={{ duration: 0.3 }}
                    className="flex gap-3 items-center"
                  >
                    <div className="w-8 h-8 rounded-full bg-accent-purple/20 border border-accent-purple/30 flex items-center justify-center shrink-0">
                      <span className="text-xs text-accent-purple font-bold">U</span>
                    </div>
                    <div className="bg-white/[0.06] border border-white/10 rounded-2xl rounded-tl-sm px-5 py-3 flex gap-1.5 items-center">
                      <span className="w-1.5 h-1.5 rounded-full bg-white/40 animate-[typing-dot_1.2s_ease-in-out_0ms_infinite]" />
                      <span className="w-1.5 h-1.5 rounded-full bg-white/40 animate-[typing-dot_1.2s_ease-in-out_200ms_infinite]" />
                      <span className="w-1.5 h-1.5 rounded-full bg-white/40 animate-[typing-dot_1.2s_ease-in-out_400ms_infinite]" />
                    </div>
                  </motion.div>
                </div>

                {/* Click hint */}
                <p className="text-xs text-white/25 text-center mt-5 tracking-wide">
                  Click to open chat interface →
                </p>
              </GlassCard>
            </Link>
          </motion.div>
        </motion.div>
      </motion.div>
    </section>
  );
}