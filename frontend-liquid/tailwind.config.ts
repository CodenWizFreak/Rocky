import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        "bg-base": "#04040c",
        "bg-surface": "rgba(255,255,255,0.04)",
        "accent-green": "#00ff88",
        "accent-purple": "#a855f7",
        "accent-violet": "#7c3aed",
        "accent-glow": "#c084fc",
      },
      animation: {
        aurora: "aurora 4s infinite",
        float: "float 6s ease-in-out infinite",
        morph: "morph 12s infinite",
        "pulse-glow": "pulse-glow 2.5s infinite",
        ripple: "ripple 0.6s ease-out",
        "liquid-fill": "liquid-fill 0.4s cubic-bezier(.25,.46,.45,.94)",
        "typing-dot": "typing-dot 1s infinite alternate",
      },
      keyframes: {
        aurora: {
          "0%, 100%": { backgroundPosition: "0% 50%" },
          "50%": { backgroundPosition: "100% 50%" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-12px)" },
        },
        morph: {
          "0%, 100%": { borderRadius: "60% 40% 30% 70% / 60% 30% 70% 40%" },
          "50%": { borderRadius: "30% 60% 70% 40% / 50% 60% 30% 60%" },
        },
        "pulse-glow": {
          "0%, 100%": { boxShadow: "0 0 15px #a855f7" },
          "50%": { boxShadow: "0 0 15px #00ff88" },
        },
        ripple: {
          "0%": { transform: "scale(0)", opacity: "0.4" },
          "100%": { transform: "scale(2.5)", opacity: "0" },
        },
        "liquid-fill": {
          "0%": { transform: "scaleX(0)" },
          "100%": { transform: "scaleX(1)" },
        },
        "typing-dot": {
          "0%": { transform: "translateY(0)" },
          "100%": { transform: "translateY(-6px)" },
        },
      },
    },
  },
  plugins: [],
};
export default config;
