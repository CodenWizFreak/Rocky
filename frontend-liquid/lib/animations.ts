import { Variants } from "framer-motion";

/** Single element: fade up into view */
export const fadeUp: Variants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.55, ease: [0.22, 1, 0.36, 1] },
  },
};

/** Container: staggers its children */
export const stagger: Variants = {
  hidden: {},
  visible: {
    transition: { staggerChildren: 0.08, delayChildren: 0.1 },
  },
};

/** Card spring lift on hover */
export const springLift: Variants = {
  rest: { y: 0 },
  hover: {
    y: -8,
    transition: { type: "spring", stiffness: 280, damping: 20 },
  },
};

/** Page-level enter/exit */
export const pageTransition: Variants = {
  initial: { opacity: 0, scale: 0.985 },
  animate: {
    opacity: 1,
    scale: 1,
    transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] },
  },
  exit: {
    opacity: 0,
    scale: 1.015,
    transition: { duration: 0.3, ease: [0.22, 1, 0.36, 1] },
  },
};

/** Slide in from left (sidebar, drawer) */
export const slideInLeft: Variants = {
  hidden: { opacity: 0, x: -24 },
  visible: {
    opacity: 1,
    x: 0,
    transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] },
  },
};

/** Fade in only — for overlays / backdrops */
export const fadeIn: Variants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { duration: 0.35 } },
  exit: { opacity: 0, transition: { duration: 0.25 } },
};