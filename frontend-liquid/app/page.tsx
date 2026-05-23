"use client";

import { GlassNav } from "@/components/ui/GlassNav";
import { HeroSection } from "@/components/landing/HeroSection";
import { FeaturesSection } from "@/components/landing/FeaturesSection";
import { StatsSection } from "@/components/landing/StatsSection";
import { CtaSection } from "@/components/landing/CtaSection";
import { motion, AnimatePresence } from "framer-motion";
import { pageTransition } from "@/lib/animations";

export default function LandingPage() {
  return (
    <AnimatePresence mode="wait">
      <motion.div
        key="landing"
        variants={pageTransition}
        initial="initial"
        animate="animate"
        exit="exit"
        className="min-h-screen pt-4 pb-20"
      >
        <GlassNav />
        <main>
          <HeroSection />
          <FeaturesSection />
          <StatsSection />
          <CtaSection />
        </main>
      </motion.div>
    </AnimatePresence>
  );
}