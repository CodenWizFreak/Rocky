import type { Metadata } from "next";
import { Inter, Space_Grotesk } from "next/font/google";
import "./globals.css";
import { ParticleField } from "@/components/ui/ParticleField";
import { AuroraOrbs } from "@/components/ui/AuroraOrbs";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space-grotesk",
});

export const metadata: Metadata = {
  title: "GNN Movie Recommender",
  description:
    "A knowledge-graph powered recommendation system that can selectively forget user preferences on request.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${inter.variable} ${spaceGrotesk.variable} font-sans antialiased text-white selection:bg-accent-green selection:text-bg-base overflow-x-hidden`}
      >
        <ParticleField />
        <AuroraOrbs />
        {children}
      </body>
    </html>
  );
}
