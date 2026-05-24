"use client";

import { motion } from "framer-motion";
import { GlassCard } from "../ui/GlassCard";
import { GlassButton } from "../ui/GlassButton";
import Link from "next/link";
import { fadeUp } from "@/lib/animations";

export function CtaSection() {
  return (
    <section className="py-24 px-4 md:px-8 relative z-10">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-80px" }}
          variants={fadeUp}
        >
          <GlassCard
            glow
            className="relative overflow-hidden p-10 md:p-20 text-center"
          >
            {/*
              Morphing blob — placed INSIDE the card, z-0, aria-hidden.
              Uses animate-morph keyframe defined in tailwind.config.ts:
              morph: {
                '0%,100%': { borderRadius: '60% 40% 30% 70% / 60% 30% 70% 40%' },
                '50%':     { borderRadius: '30% 60% 70% 40% / 50% 60% 30% 60%' },
              }
            */}
            <div
              aria-hidden="true"
              className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[520px] h-[320px] pointer-events-none z-0 animate-morph opacity-25"
              style={{
                background:
                  "radial-gradient(ellipse at center, var(--accent-purple) 0%, var(--accent-violet) 40%, transparent 75%)",
                filter: "blur(60px)",
              }}
            />

            {/* Secondary orb — offset to bottom-right */}
            <div
              aria-hidden="true"
              className="absolute -bottom-16 -right-16 w-72 h-72 rounded-full pointer-events-none z-0"
              style={{
                background:
                  "radial-gradient(circle, rgba(0,255,136,0.18) 0%, transparent 65%)",
                filter: "blur(40px)",
              }}
            />

            {/* Gradient border shimmer on top edge */}
            <div
              aria-hidden="true"
              className="absolute inset-x-0 top-0 h-px pointer-events-none z-10"
              style={{
                background:
                  "linear-gradient(to right, transparent, rgba(168,85,247,0.6) 30%, rgba(0,255,136,0.5) 70%, transparent)",
              }}
            />

            {/* Content */}
            <div className="relative z-10 flex flex-col items-center gap-6">
              {/* Eyebrow */}
              <span className="text-xs font-semibold tracking-[0.2em] uppercase text-accent-green/80 px-4 py-1.5 rounded-full bg-accent-green/10 border border-accent-green/20">
                Live Demo Available
              </span>

              <h2 className="font-space-grotesk text-3xl sm:text-4xl md:text-5xl font-bold leading-tight max-w-2xl">
                Ready to try{" "}
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-accent-purple via-accent-glow to-accent-green">
                  Selective Unlearning
                </span>
                ?
              </h2>

              <p className="text-white/50 text-base md:text-lg max-w-xl leading-relaxed">
                See how our AI updates its knowledge graph and modifies
                recommendations in real time when you ask it to forget.
              </p>

              {/* Buttons row */}
              <div className="flex flex-col sm:flex-row items-center gap-3 mt-2">
                <Link href="/chat">
                  <GlassButton size="lg" variant="primary">
                    Launch Chat Interface
                  </GlassButton>
                </Link>
                <Link href="/architecture">
                  <GlassButton size="lg" variant="ghost">
                    View Architecture
                  </GlassButton>
                </Link>
              </div>

              {/* Trust line */}
              <p className="text-xs text-white/25 tracking-wide mt-2">
                No sign-up required · Runs on live graph data
              </p>
            </div>
          </GlassCard>
        </motion.div>
      </div>
    </section>
  );
}