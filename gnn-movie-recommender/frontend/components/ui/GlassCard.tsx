"use client";

import { motion } from "framer-motion";
import { springLift } from "@/lib/animations";
import { cn } from "@/lib/utils"; // need to create this

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
  glow?: boolean;
}

export function GlassCard({
  children,
  className,
  hover,
  glow,
}: GlassCardProps) {
  return (
    <motion.div
      variants={hover ? springLift : undefined}
      initial="rest"
      whileHover={hover ? "hover" : undefined}
      className={cn(
        "relative rounded-2xl glass-mask shadow-[inset_0_1px_rgba(255,255,255,0.06)] bg-bg-surface backdrop-blur-[40px] backdrop-saturate-[180%]",
        glow && "hover:shadow-[0_0_15px_#00ff88]",
        className,
      )}
    >
      {children}
    </motion.div>
  );
}
