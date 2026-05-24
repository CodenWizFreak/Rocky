"use client";

import { motion } from "framer-motion";

export function AuroraOrbs() {
  return (
    <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden">
      <motion.div
        animate={{
          scale: [1, 1.2, 1],
          x: [0, 40, 0],
          y: [0, 30, 0],
        }}
        transition={{ duration: 12, repeat: Infinity, ease: "easeInOut" }}
        className="absolute -top-[20%] -left-[10%] w-[50vw] h-[50vw] rounded-full mix-blend-screen opacity-20 blur-[80px]"
        style={{
          background:
            "radial-gradient(circle, var(--accent-purple) 0%, transparent 70%)",
        }}
      />

      <motion.div
        animate={{
          scale: [1, 1.1, 1],
          x: [0, -30, 0],
          y: [0, -40, 0],
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 2,
        }}
        className="absolute -bottom-[20%] -right-[10%] w-[45vw] h-[45vw] rounded-full mix-blend-screen opacity-20 blur-[80px]"
        style={{
          background:
            "radial-gradient(circle, var(--accent-green) 0%, transparent 70%)",
        }}
      />

      <motion.div
        animate={{
          scale: [1, 1.3, 1],
          x: [0, 20, 0],
          y: [0, -20, 0],
        }}
        transition={{
          duration: 14,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 4,
        }}
        className="absolute top-[30%] left-[30%] w-[40vw] h-[40vw] rounded-full mix-blend-screen opacity-[0.15] blur-[80px]"
        style={{
          background:
            "radial-gradient(circle, var(--accent-violet) 0%, transparent 70%)",
        }}
      />
    </div>
  );
}
