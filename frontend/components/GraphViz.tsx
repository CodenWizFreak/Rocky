"use client";

import React, { useRef, useEffect, useState } from "react";
import { motion } from "framer-motion";
import * as d3 from "d3";
import { Share2, X, Loader2 } from "lucide-react";
import { fetchGraph, type GraphNode, type GraphLink } from "@/lib/api";

interface Props {
  userId: number;
  onClose: () => void;
}

interface SimNode extends GraphNode, d3.SimulationNodeDatum {}
interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  type: string;
  rating?: number;
}

export default function GraphViz({ userId, onClose }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [graphData, setGraphData] = useState<{ nodes: SimNode[]; links: SimLink[] } | null>(null);
  const [loading, setLoading] = useState(true);

  // Fetch graph data
  useEffect(() => {
    const load = async () => {
      try {
        const data = await fetchGraph(userId);
        setGraphData({
          nodes: data.nodes as SimNode[],
          links: data.links as SimLink[],
        });
      } catch {
        // ignore
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [userId]);

  // Render D3 graph AFTER both data is loaded AND SVG is mounted
  useEffect(() => {
    if (!graphData || !svgRef.current) return;

    const svgElement = svgRef.current;
    const width = svgElement.clientWidth || 900;
    const height = svgElement.clientHeight || 500;

    d3.select(svgElement).selectAll("*").remove();

    const svg = d3.select(svgElement)
      .attr("width", width)
      .attr("height", height);

    // Single container for zoom
    const container = svg.append("g");

    const nodeColors: Record<string, string> = {
      user: "#6366f1",
      movie: "#a855f7",
      genre: "#22d3ee",
      franchise: "#f43f5e",
    };

    const linkColors: Record<string, string> = {
      watched: "#6366f180",
      has_genre: "#22d3ee40",
      in_franchise: "#f43f5e40",
    };

    // Deep copy data so D3 can mutate it
    const nodes: SimNode[] = graphData.nodes.map((n) => ({ ...n }));
    const processedLinks: SimLink[] = graphData.links.map((l) => ({
      source: l.source as string,
      target: l.target as string,
      type: l.type,
      rating: l.rating,
    }));

    const simulation = d3
      .forceSimulation<SimNode>(nodes)
      .force(
        "link",
        d3.forceLink<SimNode, SimLink>(processedLinks)
          .id((d) => d.id)
          .distance(80)
      )
      .force("charge", d3.forceManyBody().strength(-150))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(30));

    // Links
    const link = container
      .append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(processedLinks)
      .join("line")
      .attr("stroke", (d) => linkColors[d.type] || "#ffffff20")
      .attr("stroke-width", (d) => (d.type === "watched" ? 2 : 1))
      .attr("stroke-dasharray", (d) => (d.type === "has_genre" ? "4,4" : "none"));

    // Nodes
    const node = container
      .append("g")
      .attr("class", "nodes")
      .selectAll<SVGGElement, SimNode>("g")
      .data(nodes)
      .join("g")
      .style("cursor", "grab");

    // Drag behaviour
    const dragBehavior = d3.drag<SVGGElement, SimNode>()
      .on("start", (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });

    (node as unknown as d3.Selection<SVGGElement, SimNode, SVGGElement, unknown>).call(dragBehavior);

    // Node circles
    node
      .append("circle")
      .attr("r", (d) => {
        if (d.type === "user") return 16;
        if (d.type === "movie") return 10;
        return 8;
      })
      .attr("fill", (d) => nodeColors[d.type] || "#666")
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.5)
      .attr("stroke-opacity", 0.2)
      .style("filter", (d) =>
        d.type === "user" ? "drop-shadow(0 0 8px rgba(99,102,241,0.6))" : "none"
      );

    // Node labels
    node
      .append("text")
      .text((d) => {
        if (d.type === "user") return `U${d.label.replace("User ", "")}`;
        if (d.type === "movie") {
          const title = d.label.replace(/ \(\d{4}\)$/, "");
          return title.length > 15 ? title.slice(0, 14) + "…" : title;
        }
        return d.label;
      })
      .attr("dy", (d) => (d.type === "user" ? 28 : 20))
      .attr("text-anchor", "middle")
      .attr("fill", "#9898b0")
      .attr("font-size", (d) => (d.type === "user" ? "10px" : "8px"))
      .attr("font-family", "Inter, system-ui, sans-serif")
      .attr("pointer-events", "none");

    // Tick update
    simulation.on("tick", () => {
      link
        .attr("x1", (d) => (d.source as SimNode).x!)
        .attr("y1", (d) => (d.source as SimNode).y!)
        .attr("x2", (d) => (d.target as SimNode).x!)
        .attr("y2", (d) => (d.target as SimNode).y!);

      node.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });

    // Zoom — only on the container
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on("zoom", (event) => {
        container.attr("transform", event.transform);
      });

    svg.call(zoom);

    // Cleanup
    return () => {
      simulation.stop();
    };
  }, [graphData]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-40 bg-black/80 backdrop-blur-sm flex items-center justify-center p-8"
    >
      <motion.div
        initial={{ scale: 0.95 }}
        animate={{ scale: 1 }}
        exit={{ scale: 0.95 }}
        className="glass-card w-full max-w-[1000px] h-[600px] p-6 relative overflow-hidden"
      >
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-semibold gradient-text flex items-center gap-2">
              <Share2 className="h-4 w-4 text-[var(--accent-cyan)]" strokeWidth={2} aria-hidden />
              Taste web — user {userId}
            </h3>
            <p className="text-[10px] text-[var(--text-muted)] mt-1">
              Nodes and edges, no astrophage — drag to explore, scroll to zoom
            </p>
          </div>
          <div className="flex items-center gap-4">
            {/* Legend */}
            <div className="flex items-center gap-3">
              {[
                { color: "#6366f1", label: "User" },
                { color: "#a855f7", label: "Movie" },
                { color: "#22d3ee", label: "Genre" },
                { color: "#f43f5e", label: "Franchise" },
              ].map(({ color, label }) => (
                <div key={label} className="flex items-center gap-1">
                  <div
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ background: color }}
                  />
                  <span className="text-[10px] text-[var(--text-muted)]">{label}</span>
                </div>
              ))}
            </div>
            <button
              type="button"
              onClick={onClose}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs bg-[var(--bg-card)] border border-[var(--border-subtle)]
                         hover:border-[var(--accent-rose)] hover:text-[var(--accent-rose)] transition-all"
            >
              <X className="h-3.5 w-3.5" strokeWidth={2} />
              Close
            </button>
          </div>
        </div>

        {loading ? (
          <div className="flex items-center justify-center h-[calc(100%-60px)]">
            <div className="text-center">
              <Loader2 className="w-8 h-8 text-[var(--accent-indigo)] animate-spin mx-auto mb-3" strokeWidth={2} aria-hidden />
              <p className="text-sm text-[var(--text-muted)]">Spinning up harmonics...</p>
            </div>
          </div>
        ) : (
          <svg
            ref={svgRef}
            className="w-full h-[calc(100%-60px)] rounded-xl"
            style={{ background: "rgba(10,10,15,0.5)" }}
          />
        )}
      </motion.div>
    </motion.div>
  );
}
