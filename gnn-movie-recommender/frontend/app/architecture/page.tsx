"use client";

import React, { useRef } from "react";
import Link from "next/link";
import { motion, useInView } from "framer-motion";
import {
  ArrowLeft,
  Database,
  Layers,
  GitBranch,
  ShieldAlert,
  Cpu,
  Zap,
  Key,
} from "lucide-react";
import { GlassButton } from "@/components/ui/GlassButton";
import { GlassCard } from "@/components/ui/GlassCard";
import { fadeUp, stagger } from "@/lib/animations";

// ── Reusable section wrapper with scroll-triggered entrance ──────────────────
function Section({
  children,
  delay = 0,
  className = "",
}: {
  children: React.ReactNode;
  delay?: number;
  className?: string;
}) {
  const ref = useRef<HTMLElement>(null);
  const inView = useInView(ref, { once: true, margin: "-60px" });

  return (
    <motion.section
      ref={ref}
      variants={stagger}
      initial="hidden"
      animate={inView ? "visible" : "hidden"}
      transition={{ delayChildren: delay }}
      className={`space-y-10 ${className}`}
    >
      {children}
    </motion.section>
  );
}

// ── Section heading ──────────────────────────────────────────────────────────
function SectionHeading({
  icon: Icon,
  label,
  title,
  accent,
}: {
  icon: React.ElementType;
  label: string;
  title: string;
  accent: "green" | "purple";
}) {
  const accentClass =
    accent === "green" ? "text-accent-green" : "text-accent-purple";
  const badgeBg =
    accent === "green"
      ? "bg-accent-green/10 border-accent-green/25 text-accent-green"
      : "bg-accent-purple/10 border-accent-purple/25 text-accent-purple";
  const lineFrom =
    accent === "green" ? "via-accent-green/40" : "via-accent-purple/40";

  return (
    <motion.div variants={fadeUp} className="flex flex-col gap-4">
      <span
        className={`inline-flex items-center gap-2 self-start px-3 py-1 rounded-full text-xs font-semibold tracking-[0.15em] uppercase border ${badgeBg}`}
      >
        <Icon size={12} />
        {label}
      </span>
      <h2
        className={`font-space-grotesk text-2xl md:text-4xl font-bold relative inline-block ${accentClass}`}
      >
        {title}
        <span
          aria-hidden="true"
          className={`absolute -bottom-2 left-0 right-0 h-px rounded-full bg-gradient-to-r from-transparent ${lineFrom} to-transparent`}
        />
      </h2>
    </motion.div>
  );
}

// ── Data ─────────────────────────────────────────────────────────────────────
const dataFlowSteps = [
  {
    step: "01",
    title: "Data Ingestion",
    desc: "MovieLens 100K data is transformed into a rich Knowledge Graph representation with typed nodes and edges.",
    accent: "green" as const,
  },
  {
    step: "02",
    title: "Shard Partitioning",
    desc: "The Knowledge Graph is split into isolated user shards for parallel, localised processing.",
    accent: "purple" as const,
  },
  {
    step: "03",
    title: "LightGCN Training",
    desc: "Each shard trains its own isolated LightGCN model independently on its partition data.",
    accent: "green" as const,
  },
  {
    step: "04",
    title: "Score Aggregation",
    desc: "Trained shard outputs are combined via an aggregation engine into a unified ranking.",
    accent: "purple" as const,
  },
  {
    step: "05",
    title: "Targeted Unlearning",
    desc: "Unlearn requests trigger edge deletion, embedding updates, or single-shard retraining.",
    accent: "green" as const,
  },
  {
    step: "06",
    title: "Explainable Recs",
    desc: "Updated recommendations are served alongside interpretable, user-facing reasoning.",
    accent: "purple" as const,
  },
];

const coreComponents = [
  {
    icon: Database,
    title: "Knowledge Graph Builder",
    specs: [
      {
        label: "Role",
        value: "Constructs and exports graph artifacts from raw data",
      },
      { label: "Nodes", value: "Users · Movies · Genres · Franchises" },
      {
        label: "Edges",
        value: "Watched · BelongsTo Genre · BelongsTo Franchise",
      },
    ],
    accent: "green" as const,
  },
  {
    icon: Cpu,
    title: "LightGCN Models",
    specs: [
      { label: "Training", value: "Independently per shard using BPR loss" },
      { label: "Embedding Dim", value: "64" },
      { label: "Layers", value: "3" },
      {
        label: "Purpose",
        value: "Lightweight graph-conv embeddings per shard",
      },
    ],
    accent: "purple" as const,
  },
  {
    icon: ShieldAlert,
    title: "Unlearning & Sharding",
    specs: [
      { label: "Strategy", value: "Random user partition into 5 shards" },
      {
        label: "Level 1 (Fast)",
        value: "Delete edge + recompute embedding (ms)",
      },
      {
        label: "Level 2 (Strong)",
        value: "Retrain affected shard only (< 1 min)",
      },
    ],
    accent: "green" as const,
  },
  {
    icon: Zap,
    title: "Inference & Explainability",
    specs: [
      {
        label: "Aggregation",
        value: "Combines shard outputs into final ranking",
      },
      {
        label: "Penalties",
        value: "Disliked movies/franchises score-penalised",
      },
      {
        label: "Explainability",
        value: "Matching genres, similar actors, graph paths",
      },
    ],
    accent: "purple" as const,
  },
];

const endpoints = [
  { method: "GET", path: "/users", accent: "green" as const },
  { method: "GET", path: "/history/{user_id}", accent: "green" as const },
  {
    method: "GET",
    path: "/recommend/{user_id}?k=10",
    accent: "green" as const,
  },
  { method: "POST", path: "/unlearn", accent: "purple" as const },
  {
    method: "GET",
    path: "/explain/{user_id}/{movie_id}",
    accent: "green" as const,
  },
  { method: "GET", path: "/graph/{user_id}", accent: "green" as const },
];

const requestPayload = `{
  "user_id": 42,
  "movie_id": 318,
  "scope": "franchise"
}`;

const responsePayload = `{
  "user_id": 42,
  "recommendations": [
    {
      "movie_id": 101,
      "title": "Sinister",
      "genre": ["Horror", "Mystery"],
      "franchise": null,
      "score": 0.94,
      "reason": "Matched genre pattern..."
    }
  ],
  "unlearn_applied": true,
  "excluded_franchises": ["Conjuring"]
}`;

// ── Page ─────────────────────────────────────────────────────────────────────
export default function ArchitecturePage() {
  return (
    <div className="min-h-screen bg-[#04040c] text-white selection:bg-accent-violet/30 cursor-default overflow-x-hidden">
      {/* ── Fixed ambient orbs ── */}
      <div
        aria-hidden="true"
        className="fixed inset-0 pointer-events-none -z-10 overflow-hidden"
      >
        <div className="absolute top-0 left-1/4 w-[600px] h-[600px] rounded-full bg-accent-violet/10 blur-[140px] animate-[float_10s_ease-in-out_infinite]" />
        <div className="absolute bottom-0 right-1/4 w-[700px] h-[700px] rounded-full bg-accent-green/8 blur-[160px] animate-[float_13s_ease-in-out_2s_infinite_alternate-reverse]" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[400px] h-[400px] rounded-full bg-accent-purple/6 blur-[100px] animate-[float_8s_ease-in-out_1s_infinite_alternate]" />
      </div>

      {/* ── Canvas wrapper ── */}
      <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 py-12 md:py-24 flex flex-col gap-24 relative z-10">
        {/* ── Header ── */}
        <motion.header
          variants={stagger}
          initial="hidden"
          animate="visible"
          className="flex flex-col gap-8 items-start"
        >
          <motion.div variants={fadeUp}>
            <Link href="/">
              <GlassButton
                size="sm"
                variant="ghost"
                className="group flex items-center gap-2"
              >
                <ArrowLeft
                  size={15}
                  className="group-hover:-translate-x-1 transition-transform duration-200"
                />
                Back to Home
              </GlassButton>
            </Link>
          </motion.div>

          <div className="space-y-6 max-w-3xl">
            <motion.div variants={fadeUp}>
              <span className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold tracking-[0.18em] uppercase bg-accent-green/10 border border-accent-green/25 text-accent-green shadow-[0_0_20px_rgba(0,255,136,0.12)]">
                <Layers size={11} />
                System Architecture
              </span>
            </motion.div>

            <motion.h1
              variants={fadeUp}
              className="font-space-grotesk text-4xl sm:text-5xl md:text-[68px] font-bold leading-[1.05] tracking-tight"
            >
              Architecture{" "}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-accent-green via-accent-glow to-accent-purple">
                Details
              </span>
            </motion.h1>

            <motion.p
              variants={fadeUp}
              className="text-white/50 text-lg md:text-xl font-light leading-relaxed"
            >
              Deep dive into the KG + GNN Movie Recommender system — how
              knowledge graphs, sharded models, and machine unlearning compose
              into a single, auditable pipeline.
            </motion.p>

            {/* Header stat pills */}
            <motion.div variants={fadeUp} className="flex flex-wrap gap-3 pt-2">
              {[
                { val: "5", label: "Shards" },
                { val: "102K+", label: "Graph Nodes" },
                { val: "45ms", label: "Unlearn Time" },
                { val: "99%", label: "Accuracy Retained" },
              ].map((s) => (
                <div
                  key={s.label}
                  className="flex items-baseline gap-1.5 px-4 py-2 rounded-xl bg-white/[0.04] border border-white/[0.09] backdrop-blur-sm"
                >
                  <span className="font-space-grotesk text-lg font-bold text-accent-green">
                    {s.val}
                  </span>
                  <span className="text-xs text-white/40">{s.label}</span>
                </div>
              ))}
            </motion.div>
          </div>
        </motion.header>

        {/* ── Data Flow ── */}
        <Section delay={0.05}>
          <SectionHeading
            icon={GitBranch}
            label="Pipeline"
            title="Data Flow"
            accent="purple"
          />

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {dataFlowSteps.map((item, i) => {
              const isGreen = item.accent === "green";
              return (
                <motion.div
                  key={item.step}
                  variants={fadeUp}
                  className="h-full"
                >
                  <GlassCard
                    hover
                    className="p-6 h-full flex flex-col gap-5 relative overflow-hidden group"
                  >
                    {/* Shimmer sweep */}
                    <div
                      aria-hidden="true"
                      className="absolute inset-0 pointer-events-none translate-x-[-110%] group-hover:translate-x-[110%] transition-transform duration-700 ease-in-out"
                      style={{
                        background:
                          "linear-gradient(105deg, transparent 30%, rgba(255,255,255,0.04) 50%, transparent 70%)",
                      }}
                    />

                    {/* Step number */}
                    <div className="flex items-center justify-between">
                      <span
                        className={`font-space-grotesk text-4xl font-bold ${
                          isGreen
                            ? "text-accent-green/20"
                            : "text-accent-purple/20"
                        }`}
                      >
                        {item.step}
                      </span>
                      {/* Connector dot */}
                      <div
                        className={`w-2 h-2 rounded-full ${
                          isGreen ? "bg-accent-green" : "bg-accent-purple"
                        } animate-pulse-glow`}
                      />
                    </div>

                    {/* Text */}
                    <div className="flex flex-col gap-2">
                      <h3 className="text-[15px] font-semibold text-white leading-snug">
                        {item.title}
                      </h3>
                      <p className="text-sm text-white/45 leading-relaxed">
                        {item.desc}
                      </p>
                    </div>

                    {/* Bottom accent line */}
                    <div
                      className={`h-px w-0 group-hover:w-full mt-auto transition-all duration-500 ease-out rounded-full ${
                        isGreen
                          ? "bg-gradient-to-r from-accent-green/60 to-transparent"
                          : "bg-gradient-to-r from-accent-purple/60 to-transparent"
                      }`}
                    />
                  </GlassCard>
                </motion.div>
              );
            })}
          </div>

          {/* Flow connector visual */}
          <motion.div
            variants={fadeUp}
            className="flex items-center justify-center gap-0 overflow-x-auto pb-1 pt-2"
          >
            {dataFlowSteps.map((s, i) => (
              <React.Fragment key={s.step}>
                <div className="flex flex-col items-center gap-1 shrink-0">
                  <div className="w-8 h-8 rounded-full bg-white/[0.06] border border-white/10 flex items-center justify-center">
                    <span className="text-[10px] font-bold text-white/40">
                      {s.step}
                    </span>
                  </div>
                  <span className="text-[9px] text-white/30 text-center w-14 leading-tight hidden sm:block">
                    {s.title}
                  </span>
                </div>
                {i < dataFlowSteps.length - 1 && (
                  <div className="w-6 sm:w-10 h-px bg-gradient-to-r from-accent-purple/30 to-accent-green/30 shrink-0 mx-0.5" />
                )}
              </React.Fragment>
            ))}
          </motion.div>
        </Section>

        {/* ── Core Components ── */}
        <Section delay={0.08}>
          <SectionHeading
            icon={Layers}
            label="Components"
            title="Core Components & Specs"
            accent="green"
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {coreComponents.map((comp, i) => {
              const Icon = comp.icon;
              const isGreen = comp.accent === "green";
              return (
                <motion.div
                  key={comp.title}
                  variants={fadeUp}
                  className="h-full"
                >
                  <GlassCard
                    glow
                    hover
                    className="p-7 h-full flex flex-col gap-6 relative overflow-hidden group"
                  >
                    {/* Background icon watermark */}
                    <div
                      aria-hidden="true"
                      className="absolute -bottom-4 -right-4 opacity-[0.04] group-hover:opacity-[0.08] transition-opacity duration-500 pointer-events-none"
                    >
                      <Icon size={100} />
                    </div>

                    {/* Radial glow */}
                    <div
                      aria-hidden="true"
                      className="absolute inset-0 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-2xl"
                      style={{
                        background: isGreen
                          ? "radial-gradient(circle at 20% 20%, rgba(0,255,136,0.07) 0%, transparent 65%)"
                          : "radial-gradient(circle at 20% 20%, rgba(168,85,247,0.08) 0%, transparent 65%)",
                      }}
                    />

                    {/* Header */}
                    <div className="flex items-start gap-4 relative z-10">
                      <div
                        className={`w-12 h-12 rounded-2xl flex items-center justify-center border shrink-0 ${
                          isGreen
                            ? "bg-accent-green/[0.08] border-accent-green/20"
                            : "bg-accent-purple/[0.08] border-accent-purple/20"
                        }`}
                      >
                        <Icon
                          size={22}
                          className={
                            isGreen ? "text-accent-green" : "text-accent-purple"
                          }
                          strokeWidth={1.6}
                        />
                      </div>
                      <h3 className="font-space-grotesk text-lg font-semibold text-white mt-1 leading-snug">
                        {comp.title}
                      </h3>
                    </div>

                    {/* Spec rows */}
                    <div className="relative z-10 flex flex-col gap-2">
                      {comp.specs.map((spec) => (
                        <div
                          key={spec.label}
                          className="flex flex-col sm:flex-row sm:items-baseline gap-1 sm:gap-3 py-2.5 border-b border-white/[0.06] last:border-0"
                        >
                          <span className="text-xs font-semibold tracking-wider uppercase text-white/30 sm:w-36 shrink-0">
                            {spec.label}
                          </span>
                          <span className="text-sm text-white/70 leading-relaxed">
                            {spec.value}
                          </span>
                        </div>
                      ))}
                    </div>
                  </GlassCard>
                </motion.div>
              );
            })}
          </div>
        </Section>

        {/* ── Backend API Surface ── */}
        <Section delay={0.06}>
          <SectionHeading
            icon={Key}
            label="API Reference"
            title="Backend API Surface"
            accent="purple"
          />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {/* Endpoints */}
            <motion.div variants={fadeUp}>
              <GlassCard className="p-6 md:p-8 h-full flex flex-col gap-5">
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full bg-accent-green animate-[pulse-glow_2.5s_ease-in-out_infinite]" />
                  <h3 className="text-[15px] font-semibold text-white/85 tracking-wide">
                    Public Endpoints
                  </h3>
                </div>

                <div className="flex flex-col gap-2">
                  {endpoints.map((ep, i) => {
                    const isGreen = ep.accent === "green";
                    return (
                      <motion.div
                        key={`${ep.method}-${ep.path}`}
                        initial={{ opacity: 0, x: -12 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: i * 0.06, duration: 0.35 }}
                        className="flex items-center gap-3 px-4 py-2.5 rounded-xl bg-black/40 border border-white/[0.07] hover:border-white/[0.14] transition-colors duration-200 group/ep"
                      >
                        <span
                          className={`font-mono font-bold text-[11px] tracking-wider w-9 shrink-0 ${
                            isGreen ? "text-accent-green" : "text-accent-purple"
                          }`}
                        >
                          {ep.method}
                        </span>
                        <span className="font-mono text-sm text-white/55 group-hover/ep:text-white/80 transition-colors duration-200 truncate">
                          {ep.path}
                        </span>
                      </motion.div>
                    );
                  })}
                </div>
              </GlassCard>
            </motion.div>

            {/* Payloads */}
            <motion.div variants={fadeUp} className="flex flex-col gap-5">
              {/* Request */}
              <GlassCard className="p-6 flex flex-col gap-4 flex-1">
                <div className="flex items-center justify-between">
                  <h3 className="text-[15px] font-semibold text-white/85 tracking-wide">
                    Request Payload
                  </h3>
                  <span className="text-xs font-mono px-2.5 py-1 rounded-lg bg-accent-purple/10 border border-accent-purple/20 text-accent-purple">
                    POST /unlearn
                  </span>
                </div>
                <div className="relative">
                  {/* Top border glow */}
                  <div
                    aria-hidden="true"
                    className="absolute inset-x-0 top-0 h-px rounded-full bg-gradient-to-r from-transparent via-accent-purple/40 to-transparent"
                  />
                  <pre className="p-4 rounded-xl bg-black/50 border border-white/[0.07] font-mono text-xs text-white/55 overflow-x-auto leading-relaxed">
                    {requestPayload}
                  </pre>
                </div>
              </GlassCard>

              {/* Response */}
              <GlassCard className="p-6 flex flex-col gap-4 flex-1">
                <div className="flex items-center justify-between">
                  <h3 className="text-[15px] font-semibold text-white/85 tracking-wide">
                    Response Payload
                  </h3>
                  <span className="text-xs font-mono px-2.5 py-1 rounded-lg bg-accent-green/10 border border-accent-green/20 text-accent-green">
                    200 OK
                  </span>
                </div>
                <div className="relative">
                  <div
                    aria-hidden="true"
                    className="absolute inset-x-0 top-0 h-px rounded-full bg-gradient-to-r from-transparent via-accent-green/40 to-transparent"
                  />
                  <pre className="p-4 rounded-xl bg-black/50 border border-white/[0.07] font-mono text-xs text-white/55 overflow-x-auto leading-relaxed">
                    {responsePayload}
                  </pre>
                </div>
              </GlassCard>
            </motion.div>
          </div>
        </Section>

        {/* ── Footer CTA ── */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
        >
          <GlassCard
            glow
            className="p-8 md:p-12 text-center relative overflow-hidden"
          >
            {/* Blob */}
            <div
              aria-hidden="true"
              className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[400px] h-[200px] pointer-events-none rounded-full animate-morph opacity-20"
              style={{
                background:
                  "radial-gradient(ellipse, var(--accent-purple) 0%, var(--accent-violet) 50%, transparent 80%)",
                filter: "blur(50px)",
              }}
            />
            <div className="relative z-10 flex flex-col items-center gap-5">
              <h2 className="font-space-grotesk text-2xl md:text-4xl font-bold">
                Explore it{" "}
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-accent-purple via-accent-glow to-accent-green">
                  live
                </span>
              </h2>
              <p className="text-white/45 max-w-md text-base leading-relaxed">
                See the knowledge graph update and recommendations change in
                real time inside the chat interface.
              </p>
              <div className="flex flex-col sm:flex-row gap-3">
                <Link href="/chat">
                  <GlassButton size="lg" variant="primary">
                    Open Chat Interface
                  </GlassButton>
                </Link>
                <Link href="/">
                  <GlassButton size="lg" variant="ghost">
                    Back to Landing
                  </GlassButton>
                </Link>
              </div>
            </div>
          </GlassCard>
        </motion.div>
      </div>
    </div>
  );
}
