"use client";

import { motion, useInView } from "framer-motion";
import { fadeUp, stagger } from "@/lib/animations";
import { GlassCard } from "../ui/GlassCard";
import { useRef } from "react";
import {
  Network,
  Database,
  Trash2,
  ShieldAlert,
  CheckCircle,
  Smartphone,
} from "lucide-react";

const features = [
  {
    title: "Knowledge Graph Modeling",
    desc: "Nodes: Users, Movies, Genres, Franchises. Edges capture complex interconnected metadata natively.",
    icon: Network,
    accent: "text-accent-green",
    border: "border-accent-green/20",
    bg: "bg-accent-green/[0.08]",
    glow: "rgba(0,255,136,0.12)",
  },
  {
    title: "Sharded Training",
    desc: "Users are randomly partitioned into 5 shards. Each shard trains a separate LightGCN for localized retraining.",
    icon: Database,
    accent: "text-accent-purple",
    border: "border-accent-purple/20",
    bg: "bg-accent-purple/[0.08]",
    glow: "rgba(168,85,247,0.12)",
  },
  {
    title: "Two-Level Unlearning",
    desc: "Level 1: Fast embedding update (no retraining). Level 2: Retrain only the affected shard for strong guarantees.",
    icon: Trash2,
    accent: "text-accent-green",
    border: "border-accent-green/20",
    bg: "bg-accent-green/[0.08]",
    glow: "rgba(0,255,136,0.12)",
  },
  {
    title: "Dislike Penalties",
    desc: "Penalise individual movies and whole franchises to prevent preference bleed-through after unlearning operations.",
    icon: ShieldAlert,
    accent: "text-accent-purple",
    border: "border-accent-purple/20",
    bg: "bg-accent-purple/[0.08]",
    glow: "rgba(168,85,247,0.12)",
  },
  {
    title: "Explainable Recommendations",
    desc: "Shows exactly why a movie was recommended, referencing genres, actors, and similar user graph paths.",
    icon: CheckCircle,
    accent: "text-accent-green",
    border: "border-accent-green/20",
    bg: "bg-accent-green/[0.08]",
    glow: "rgba(0,255,136,0.12)",
  },
  {
    title: "Full-Stack Demo",
    desc: "FastAPI Python backend with a Next.js App Router frontend communicating via high-speed streaming.",
    icon: Smartphone,
    accent: "text-accent-purple",
    border: "border-accent-purple/20",
    bg: "bg-accent-purple/[0.08]",
    glow: "rgba(168,85,247,0.12)",
  },
];

export function FeaturesSection() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section id="features" className="py-24 px-4 md:px-8 relative">
      <div className="max-w-6xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-16">
          <motion.p
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="text-xs font-semibold tracking-[0.2em] uppercase text-accent-green mb-4"
          >
            What it does
          </motion.p>
          <motion.h2
            initial={{ opacity: 0, y: 14 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.08 }}
            className="font-space-grotesk text-3xl md:text-5xl font-bold relative inline-block"
          >
            Key Capabilities
            <span
              aria-hidden="true"
              className="absolute -bottom-3 left-0 right-0 h-[2px] rounded-full bg-gradient-to-r from-transparent via-accent-green/50 to-transparent"
            />
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.16 }}
            className="mt-8 text-white/45 text-base md:text-lg max-w-2xl mx-auto leading-relaxed"
          >
            A modular, graph-native system designed from the ground up for
            precise, auditable machine unlearning.
          </motion.p>
        </div>

        {/* Cards grid */}
        <motion.div
          ref={ref}
          variants={stagger}
          initial="hidden"
          animate={isInView ? "visible" : "hidden"}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5"
        >
          {features.map((f, i) => {
            const Icon = f.icon;
            return (
              <motion.div key={f.title} variants={fadeUp} className="h-full">
                <GlassCard
                  hover
                  className="p-6 h-full flex flex-col gap-4 relative overflow-hidden group cursor-default"
                >
                  {/*
                    Aurora shimmer on hover:
                    A pseudo-element approach won't work in Tailwind without JIT config,
                    so we use a real div that transitions on group-hover.
                  */}
                  <div
                    aria-hidden="true"
                    className="absolute inset-0 pointer-events-none translate-x-[-110%] group-hover:translate-x-[110%] transition-transform duration-700 ease-in-out"
                    style={{
                      background:
                        "linear-gradient(105deg, transparent 30%, rgba(255,255,255,0.05) 50%, transparent 70%)",
                    }}
                  />

                  {/* Hover radial glow */}
                  <div
                    aria-hidden="true"
                    className="absolute inset-0 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-2xl"
                    style={{
                      background: `radial-gradient(circle at 30% 30%, ${f.glow} 0%, transparent 65%)`,
                    }}
                  />

                  {/* Icon */}
                  <div
                    className={`relative z-10 w-14 h-14 rounded-2xl flex items-center justify-center border ${f.border} ${f.bg} shrink-0`}
                  >
                    <Icon size={26} className={f.accent} strokeWidth={1.6} />
                  </div>

                  {/* Text */}
                  <div className="relative z-10 flex flex-col gap-2 flex-1">
                    <h3 className="text-base font-semibold text-white leading-snug">
                      {f.title}
                    </h3>
                    <p className="text-sm text-white/48 leading-relaxed flex-1">
                      {f.desc}
                    </p>
                  </div>

                  {/* Bottom accent line */}
                  <div
                    className={`relative z-10 h-px w-0 group-hover:w-full transition-all duration-500 ease-out rounded-full bg-gradient-to-r ${
                      i % 2 === 0
                        ? "from-accent-green/60 to-transparent"
                        : "from-accent-purple/60 to-transparent"
                    }`}
                  />
                </GlassCard>
              </motion.div>
            );
          })}
        </motion.div>
      </div>
    </section>
  );
}