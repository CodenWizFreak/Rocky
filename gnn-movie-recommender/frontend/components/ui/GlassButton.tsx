"use client";

import React, { useState } from "react";
import { cn } from "@/lib/utils";

interface GlassButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md" | "lg";
  children: React.ReactNode;
}

export const GlassButton = React.forwardRef<
  HTMLButtonElement,
  GlassButtonProps
>(
  (
    {
      className,
      variant = "primary",
      size = "md",
      children,
      onClick,
      ...props
    },
    ref,
  ) => {
    const [isRippling, setIsRippling] = useState(false);

    const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
      setIsRippling(true);
      setTimeout(() => setIsRippling(false), 600);
      if (onClick) onClick(e);
    };

    const baseStyles =
      "relative inline-flex items-center justify-center font-medium overflow-hidden rounded-full transition-colors duration-300";

    const sizeStyles = {
      sm: "px-4 py-2 text-sm",
      md: "px-6 py-3 text-base",
      lg: "px-8 py-4 text-lg",
    };

    const variantStyles = {
      primary:
        "bg-accent-green text-bg-base hover:shadow-[0_0_20px_rgba(0,255,136,0.4)] border border-transparent",
      secondary:
        "bg-transparent text-white border border-accent-purple glass-mask hover:shadow-[0_0_20px_rgba(168,85,247,0.3)]",
      ghost: "bg-transparent text-white hover:bg-white/5",
    };

    return (
      <button
        ref={ref}
        onClick={handleClick}
        className={cn(
          baseStyles,
          sizeStyles[size],
          variantStyles[variant],
          "group",
          className,
        )}
        {...props}
      >
        <span className="relative z-10">{children}</span>

        {variant !== "ghost" && (
          <div className="absolute inset-0 z-0 bg-white/20 origin-left scale-x-0 transition-transform duration-400 ease-[cubic-bezier(0.25,0.46,0.45,0.94)] group-hover:scale-x-100" />
        )}

        {isRippling && (
          <span className="absolute inset-0 z-20 flex items-center justify-center">
            <span className="w-full h-full bg-white/30 rounded-full animate-ripple aspect-square" />
          </span>
        )}
      </button>
    );
  },
);

GlassButton.displayName = "GlassButton";
