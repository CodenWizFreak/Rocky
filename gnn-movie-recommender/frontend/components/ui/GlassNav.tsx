"use client";

import { motion } from "framer-motion";
import { GlassButton } from "./GlassButton";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, X } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";
import Image from "next/image";

interface GlassNavProps {
  onMobileMenuToggle?: (isOpen: boolean) => void;
}

export function GlassNav({ onMobileMenuToggle }: GlassNavProps) {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);

  const toggle = () => {
    setIsOpen(!isOpen);
    onMobileMenuToggle?.(!isOpen);
  };

  const navLinks = [
    { name: "Home", path: "/" },
    { name: "Features", path: "/#features" },
    { name: "Architecture", path: "/architecture" },
  ];

  return (
    <>
      <motion.nav
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ type: "spring", duration: 0.6, bounce: 0.1 }}
        className="sticky top-4 z-50 mx-4 md:mx-auto max-w-6xl rounded-2xl glass-mask shadow-[inset_0_1px_rgba(255,255,255,0.06)] bg-bg-surface backdrop-blur-[40px] backdrop-saturate-[180%] px-6 py-3 flex items-center justify-between gap-6"
      >
        {/* Logo: GIF + name */}
        <div className="flex-1 shrink-0">
          <Link
            href="/"
            className="flex items-center gap-2.5 group"
          >
            <div className="relative w-10 h-10 shrink-0">
              <Image
                src="/rocky.gif"
                alt="Rocky the robot"
                fill
                unoptimized
                className="object-contain"
                priority
              />
            </div>
            <div className="flex flex-col leading-none">
              <span className="text-[11px] font-medium text-white/40 tracking-widest uppercase">
                KG + GNN
              </span>
              <span className="text-base font-bold text-transparent bg-clip-text bg-gradient-to-r from-accent-purple to-accent-green">
                Rocky
              </span>
            </div>
          </Link>
        </div>

        {/* Desktop Nav */}
        <div className="hidden lg:flex items-center justify-center gap-8 shrink-0">
          {navLinks.map((link) => (
            <Link
              key={link.name}
              href={link.path}
              className="relative text-sm text-white/70 hover:text-white transition-colors whitespace-nowrap"
            >
              {link.name}
              {pathname === link.path && (
                <motion.div
                  layoutId="nav-indicator"
                  className="absolute -bottom-1 w-full h-[2px] bg-accent-green"
                />
              )}
            </Link>
          ))}
        </div>

        {/* Desktop CTA */}
        <div className="hidden lg:flex flex-1 justify-end shrink-0">
          <Link href="/chat">
            <GlassButton size="sm">Talk to Agent</GlassButton>
          </Link>
        </div>

        {/* Mobile Toggle */}
        <button className="lg:hidden text-white ml-auto" onClick={toggle}>
          {isOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </motion.nav>

      {/* Mobile Menu */}
      <motion.div
        initial={false}
        animate={{ height: isOpen ? "auto" : 0, opacity: isOpen ? 1 : 0 }}
        className="lg:hidden fixed top-24 left-4 right-4 z-40 overflow-hidden rounded-2xl bg-bg-surface backdrop-blur-2xl glass-mask"
      >
        <div className="p-6 flex flex-col gap-4">
          {navLinks.map((link) => (
            <Link
              key={link.name}
              href={link.path}
              onClick={() => setIsOpen(false)}
              className={cn(
                "p-2 text-lg font-medium border-b border-white/5",
                pathname === link.path ? "text-accent-green" : "text-white/70",
              )}
            >
              {link.name}
            </Link>
          ))}
          <Link
            href="/chat"
            className="mt-4 block max-w-max"
            onClick={() => setIsOpen(false)}
          >
            <GlassButton size="md">Talk to Agent</GlassButton>
          </Link>
        </div>
      </motion.div>
    </>
  );
}