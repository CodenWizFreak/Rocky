"use client";

import { motion, useInView } from "framer-motion";
import { fadeUp, stagger } from "@/lib/animations";
import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

// ── Animated counter ────────────────────────────────────────────────────────
function StatCounter({
  value,
  inView,
  duration = 1400,
}: {
  value: number;
  inView: boolean;
  duration?: number;
}) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (!inView) return;
    let startTime: number | null = null;
    let raf: number;

    const tick = (ts: number) => {
      if (!startTime) startTime = ts;
      const elapsed = ts - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // ease-out-quart
      const eased = 1 - Math.pow(1 - progress, 4);
      setCount(Math.round(eased * value));
      if (progress < 1) raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, inView, duration]);

  return <>{count}</>;
}

// ── SVG ring — each instance has a unique gradient ID ───────────────────────
const CIRCUMFERENCE = 2 * Math.PI * 44; // r = 44

function StatRing({
  id,
  inView,
  delay,
  fillPercent,
}: {
  id: string;
  inView: boolean;
  delay: number;
  fillPercent: number; // 0–100
}) {
  const offset = CIRCUMFERENCE * (1 - fillPercent / 100);

  return (
    <svg
      viewBox="0 0 100 100"
      className="absolute inset-0 w-full h-full -rotate-90"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id={id} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="var(--accent-green)" />
          <stop offset="100%" stopColor="var(--accent-purple)" />
        </linearGradient>
      </defs>

      {/* Track */}
      <circle
        cx="50"
        cy="50"
        r="44"
        fill="none"
        stroke="rgba(255,255,255,0.06)"
        strokeWidth="2.5"
      />

      {/* Filled arc */}
      <motion.circle
        cx="50"
        cy="50"
        r="44"
        fill="none"
        stroke={`url(#${id})`}
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeDasharray={CIRCUMFERENCE}
        initial={{ strokeDashoffset: CIRCUMFERENCE }}
        animate={inView ? { strokeDashoffset: offset } : { strokeDashoffset: CIRCUMFERENCE }}
        transition={{ duration: 1.6, ease: "easeOut", delay }}
      />
    </svg>
  );
}

// ── Section ──────────────────────────────────────────────────────────────────
const stats = [
  {
    label: "Shard Models",
    value: 5,
    max: 10,
    suffix: "",
    accent: "text-accent-green",
  },
  {
    label: "Graph Nodes",
    value: 102,
    max: 200,
    suffix: "K+",
    accent: "text-accent-purple",
  },
  {
    label: "Unlearn Time",
    value: 45,
    max: 200,
    suffix: "ms",
    accent: "text-accent-green",
  },
  {
    label: "Accuracy Retained",
    value: 99,
    max: 100,
    suffix: "%",
    accent: "text-accent-purple",
  },
];

export function StatsSection() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-60px" });

  return (
    <section className="py-16 px-4 md:px-8">
      {/* Subtle separator line above */}
      <div className="max-w-6xl mx-auto mb-2">
        <div className="h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />
      </div>

      <motion.div
        ref={ref}
        variants={stagger}
        initial="hidden"
        animate={isInView ? "visible" : "hidden"}
        className="max-w-6xl mx-auto grid grid-cols-2 lg:grid-cols-4 gap-px bg-white/[0.06] rounded-2xl overflow-hidden border border-white/[0.08]"
      >
        {stats.map((s, i) => (
          <motion.div
            key={s.label}
            variants={fadeUp}
            className="flex flex-col items-center justify-center text-center p-8 lg:p-10 bg-[#04040c] relative group"
          >
            {/* Hover glow */}
            <div
              aria-hidden="true"
              className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
              style={{
                background:
                  i % 2 === 0
                    ? "radial-gradient(circle at 50% 50%, rgba(0,255,136,0.05) 0%, transparent 70%)"
                    : "radial-gradient(circle at 50% 50%, rgba(168,85,247,0.06) 0%, transparent 70%)",
              }}
            />

            {/* Ring + value */}
            <div className="relative w-28 h-28 flex items-center justify-center mb-4">
              <StatRing
                id={`ring-grad-${i}`}
                inView={isInView}
                delay={i * 0.12}
                fillPercent={Math.min((s.value / s.max) * 100, 100)}
              />

              <div className={cn("relative z-10 font-space-grotesk font-bold text-3xl md:text-4xl", s.accent)}>
                <StatCounter value={s.value} inView={isInView} duration={1400} />
                <span className="text-xl md:text-2xl">{s.suffix}</span>
              </div>
            </div>

            {/* Label */}
            <p className="text-sm md:text-[15px] text-white/45 font-medium tracking-wide leading-snug">
              {s.label}
            </p>
          </motion.div>
        ))}
      </motion.div>

      <div className="max-w-6xl mx-auto mt-2">
        <div className="h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />
      </div>
    </section>
  );
}