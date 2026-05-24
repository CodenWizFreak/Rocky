"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Maximize2, ZoomIn, ZoomOut, RotateCcw } from "lucide-react";

interface GraphNode {
  id: string;
  label: string;
  type: "user" | "recommended" | "liked" | "erased" | "genre";
  weight: number;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
}

interface GraphEdge {
  source: string;
  target: string;
  weight: number;
  type: "recommends" | "liked" | "erased" | "has_genre";
}

interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  genre_weights: Record<string, number>;
  stats: {
    liked_count: number;
    disliked_count: number;
    liked_genres: string[];
    disliked_genres: string[];
  };
}

interface Props {
  graphData: GraphData | null;
  onClose: () => void;
}

const NODE_CONFIG = {
  user:        { color: "#00ff88", radius: 28, glow: "#00ff88" },
  recommended: { color: "#a855f7", radius: 20, glow: "#a855f7" },
  liked:       { color: "#38bdf8", radius: 16, glow: "#38bdf8" },
  erased:      { color: "#ef4444", radius: 14, glow: "#ef4444" },
  genre:       { color: "#f59e0b", radius: 12, glow: "#f59e0b" },
};

const EDGE_CONFIG = {
  recommends: { color: "#a855f7", opacity: 0.8 },
  liked:      { color: "#38bdf8", opacity: 0.5 },
  erased:     { color: "#ef4444", opacity: 0.3 },
  has_genre:  { color: "#f59e0b", opacity: 0.3 },
};

function runForceLayout(nodes: GraphNode[], edges: GraphEdge[], width: number, height: number, iterations: number = 200) {
  // Initialize positions
  const positioned = nodes.map((n, i) => ({
    ...n,
    x: n.type === "user" ? width / 2 : width / 2 + (Math.random() - 0.5) * width * 0.8,
    y: n.type === "user" ? height / 2 : height / 2 + (Math.random() - 0.5) * height * 0.8,
    vx: 0,
    vy: 0,
  }));

  const nodeMap = new Map(positioned.map(n => [n.id, n]));

  for (let iter = 0; iter < iterations; iter++) {
    const alpha = 1 - iter / iterations;

    // Repulsion
    for (let i = 0; i < positioned.length; i++) {
      for (let j = i + 1; j < positioned.length; j++) {
        const a = positioned[i], b = positioned[j];
        const dx = b.x! - a.x!, dy = b.y! - a.y!;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = (3500 / (dist * dist)) * alpha;
        const fx = (dx / dist) * force, fy = (dy / dist) * force;
        a.vx! -= fx; a.vy! -= fy;
        b.vx! += fx; b.vy! += fy;
      }
    }

    // Attraction along edges
    for (const e of edges) {
      const src = nodeMap.get(e.source), tgt = nodeMap.get(e.target);
      if (!src || !tgt) continue;
      const dx = tgt.x! - src.x!, dy = tgt.y! - src.y!;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const idealDist = e.type === "has_genre" ? 120 : 180;
      const force = ((dist - idealDist) / dist) * 0.08 * alpha;
      const fx = dx * force, fy = dy * force;
      src.vx! += fx; src.vy! += fy;
      tgt.vx! -= fx; tgt.vy! -= fy;
    }

    // Center gravity
    for (const n of positioned) {
      if (n.type === "user") { n.x = width / 2; n.y = height / 2; continue; }
      n.vx! += (width / 2 - n.x!) * 0.01 * alpha;
      n.vy! += (height / 2 - n.y!) * 0.01 * alpha;
    }

    // Apply velocity with damping
    for (const n of positioned) {
      if (n.type === "user") continue;
      n.vx! *= 0.85; n.vy! *= 0.85;
      n.x! += n.vx!; n.y! += n.vy!;
      n.x! = Math.max(40, Math.min(width - 40, n.x!));
      n.y! = Math.max(40, Math.min(height - 40, n.y!));
    }
  }

  return positioned;
}

export function GnnVisualizer({ graphData, onClose }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const nodesRef = useRef<GraphNode[]>([]);
  const animFrameRef = useRef<number>(0);
  const [zoom, setZoom] = useState(1);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number } | null>(null);
  const panRef = useRef({ x: 0, y: 0 });
  const isDragging = useRef(false);
  const lastMouse = useRef({ x: 0, y: 0 });

  const layout = useCallback((data: GraphData, w: number, h: number) => {
    const laidOut = runForceLayout([...data.nodes], data.edges, w, h, 300);
    nodesRef.current = laidOut;
  }, []);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { width, height } = canvas;
    const data = graphData;
    if (!data) return;
    const nodes = nodesRef.current;
    const nodeMap = new Map(nodes.map(n => [n.id, n]));

    ctx.clearRect(0, 0, width, height);
    ctx.save();
    ctx.translate(panRef.current.x, panRef.current.y);
    ctx.scale(zoom, zoom);

    // Draw edges
    for (const edge of data.edges) {
      const src = nodeMap.get(edge.source);
      const tgt = nodeMap.get(edge.target);
      if (!src || !tgt || !src.x || !tgt.x) continue;
      const cfg = EDGE_CONFIG[edge.type] || EDGE_CONFIG.recommends;

      ctx.beginPath();
      ctx.moveTo(src.x, src.y!);
      ctx.lineTo(tgt.x, tgt.y!);

      const gradient = ctx.createLinearGradient(src.x, src.y!, tgt.x, tgt.y!);
      gradient.addColorStop(0, cfg.color + "88");
      gradient.addColorStop(1, cfg.color + "22");
      ctx.strokeStyle = gradient;
      ctx.lineWidth = Math.max(0.5, edge.weight * 2.5);
      ctx.globalAlpha = cfg.opacity;
      ctx.stroke();

      // Edge weight label
      if (edge.type === "recommends" && edge.weight > 0.1) {
        ctx.globalAlpha = 0.7;
        ctx.fillStyle = "#ffffff";
        ctx.font = "10px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(
          edge.weight.toFixed(2),
          (src.x + tgt.x) / 2,
          (src.y! + tgt.y!) / 2 - 6
        );
      }
      ctx.globalAlpha = 1;
    }

    // Draw nodes
    for (const node of nodes) {
      if (!node.x) continue;
      const cfg = NODE_CONFIG[node.type] || NODE_CONFIG.recommended;
      const r = cfg.radius;
      const isHovered = hoveredNode?.id === node.id;

      // Glow
      const glowRadius = isHovered ? r * 2.5 : r * 1.8;
      const glow = ctx.createRadialGradient(node.x, node.y!, 0, node.x, node.y!, glowRadius);
      glow.addColorStop(0, cfg.glow + (isHovered ? "55" : "33"));
      glow.addColorStop(1, cfg.glow + "00");
      ctx.beginPath();
      ctx.arc(node.x, node.y!, glowRadius, 0, Math.PI * 2);
      ctx.fillStyle = glow;
      ctx.fill();

      // Node circle
      ctx.beginPath();
      ctx.arc(node.x, node.y!, isHovered ? r * 1.15 : r, 0, Math.PI * 2);
      const nodeGrad = ctx.createRadialGradient(node.x - r * 0.3, node.y! - r * 0.3, 0, node.x, node.y!, r);
      nodeGrad.addColorStop(0, cfg.color + "ff");
      nodeGrad.addColorStop(1, cfg.color + "99");
      ctx.fillStyle = nodeGrad;
      ctx.fill();

      // Border
      ctx.strokeStyle = node.type === "erased" ? "#ef4444" : "#ffffff33";
      ctx.lineWidth = isHovered ? 2 : 1;
      ctx.stroke();

      // Erased: strikethrough
      if (node.type === "erased") {
        ctx.beginPath();
        ctx.moveTo(node.x - r, node.y!);
        ctx.lineTo(node.x + r, node.y!);
        ctx.strokeStyle = "#ef444488";
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Label
      ctx.globalAlpha = 1;
      const maxLabelLen = node.type === "user" ? 999 : 14;
      const label = node.label.length > maxLabelLen
        ? node.label.slice(0, maxLabelLen) + "…"
        : node.label;
      ctx.font = `${node.type === "user" ? "bold " : ""}${node.type === "genre" ? 9 : 11}px Inter, sans-serif`;
      ctx.textAlign = "center";
      ctx.fillStyle = "#ffffff";
      ctx.shadowColor = "#000000";
      ctx.shadowBlur = 4;
      ctx.fillText(label, node.x, node.y! + r + 14);
      ctx.shadowBlur = 0;
    }

    ctx.restore();
    animFrameRef.current = requestAnimationFrame(draw);
  }, [graphData, zoom, hoveredNode]);

  // Re-layout when data changes
  useEffect(() => {
    if (!graphData || !containerRef.current) return;
    const { offsetWidth: w, offsetHeight: h } = containerRef.current;
    layout(graphData, w, h);
  }, [graphData, layout]);

  // Animation loop
  useEffect(() => {
    animFrameRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animFrameRef.current);
  }, [draw]);

  // Resize canvas
  useEffect(() => {
    const resize = () => {
      const canvas = canvasRef.current;
      const container = containerRef.current;
      if (!canvas || !container) return;
      canvas.width  = container.offsetWidth;
      canvas.height = container.offsetHeight;
      if (graphData) layout(graphData, canvas.width, canvas.height);
    };
    resize();
    const ro = new ResizeObserver(resize);
    if (containerRef.current) ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, [graphData, layout]);

  // Mouse interaction
  const getHitNode = (mx: number, my: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const cx = ((mx - rect.left) * scaleX - panRef.current.x) / zoom;
    const cy = ((my - rect.top)  * scaleY - panRef.current.y) / zoom;

    for (const node of nodesRef.current) {
      if (!node.x) continue;
      const cfg = NODE_CONFIG[node.type];
      const dx = cx - node.x, dy = cy - node.y!;
      if (dx * dx + dy * dy < cfg.radius * cfg.radius * 1.5) return node;
    }
    return null;
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const node = getHitNode(e.clientX, e.clientY);
    setHoveredNode(node);
    if (node) setTooltip({ x: e.clientX, y: e.clientY });
    else setTooltip(null);

    if (isDragging.current) {
      panRef.current.x += e.clientX - lastMouse.current.x;
      panRef.current.y += e.clientY - lastMouse.current.y;
      lastMouse.current = { x: e.clientX, y: e.clientY };
    }
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    setZoom(z => Math.max(0.3, Math.min(3, z - e.deltaY * 0.001)));
  };

  const resetView = () => {
    setZoom(1);
    panRef.current = { x: 0, y: 0 };
    if (graphData && containerRef.current) {
      layout(graphData, containerRef.current.offsetWidth, containerRef.current.offsetHeight);
    }
  };

  const stats = graphData?.stats;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        transition={{ type: "spring", damping: 25, stiffness: 300 }}
        className="relative w-full max-w-5xl h-[85vh] rounded-2xl overflow-hidden border border-white/10"
        style={{ background: "linear-gradient(135deg, #04040c 0%, #0d0720 50%, #04040c 100%)" }}
      >
        {/* Header */}
        <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between px-5 py-4 border-b border-white/10 bg-black/30 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <div className="w-2.5 h-2.5 rounded-full bg-accent-green animate-pulse" />
            <h2 className="text-white font-semibold font-space-grotesk text-lg">
              Live GNN Graph Visualizer
            </h2>
            <span className="text-xs text-white/40 font-inter">
              {graphData?.nodes.length ?? 0} nodes · {graphData?.edges.length ?? 0} edges
            </span>
          </div>
          <div className="flex items-center gap-2">
            {/* Zoom controls */}
            <button
              onClick={() => setZoom(z => Math.min(3, z + 0.2))}
              className="p-2 rounded-lg text-white/60 hover:text-white hover:bg-white/10 transition-colors"
              title="Zoom in"
            >
              <ZoomIn size={16} />
            </button>
            <button
              onClick={() => setZoom(z => Math.max(0.3, z - 0.2))}
              className="p-2 rounded-lg text-white/60 hover:text-white hover:bg-white/10 transition-colors"
              title="Zoom out"
            >
              <ZoomOut size={16} />
            </button>
            <button
              onClick={resetView}
              className="p-2 rounded-lg text-white/60 hover:text-white hover:bg-white/10 transition-colors"
              title="Reset view"
            >
              <RotateCcw size={16} />
            </button>
            <div className="w-px h-5 bg-white/20 mx-1" />
            <button
              onClick={onClose}
              className="p-2 rounded-lg text-white/60 hover:text-white hover:bg-white/10 transition-colors"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Canvas */}
        <div
          ref={containerRef}
          className="absolute inset-0 top-[60px]"
          style={{ cursor: isDragging.current ? "grabbing" : hoveredNode ? "pointer" : "grab" }}
        >
          <canvas
            ref={canvasRef}
            className="w-full h-full"
            onMouseMove={handleMouseMove}
            onMouseLeave={() => { setHoveredNode(null); setTooltip(null); }}
            onMouseDown={(e) => { isDragging.current = true; lastMouse.current = { x: e.clientX, y: e.clientY }; }}
            onMouseUp={() => { isDragging.current = false; }}
            onWheel={handleWheel}
          />

          {/* Empty state */}
          {(!graphData || graphData.nodes.length === 0) && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-white/30">
              <div className="w-20 h-20 rounded-full border-2 border-dashed border-white/20 flex items-center justify-center mb-4">
                <span className="text-3xl">🕸️</span>
              </div>
              <p className="text-sm font-inter">Start chatting to populate the GNN graph</p>
            </div>
          )}
        </div>

        {/* Legend */}
        <div className="absolute bottom-4 left-4 flex flex-col gap-1.5 bg-black/40 backdrop-blur-sm rounded-xl p-3 border border-white/10">
          {Object.entries(NODE_CONFIG).map(([type, cfg]) => (
            <div key={type} className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: cfg.color, boxShadow: `0 0 6px ${cfg.glow}` }} />
              <span className="text-xs text-white/60 capitalize font-inter">{type}</span>
            </div>
          ))}
        </div>

        {/* Stats panel */}
        {stats && (
          <div className="absolute bottom-4 right-4 bg-black/40 backdrop-blur-sm rounded-xl p-3 border border-white/10 text-xs font-inter">
            <div className="text-white/40 mb-2 font-medium uppercase tracking-wider text-[10px]">Graph State</div>
            <div className="space-y-1">
              <div className="flex gap-3">
                <span className="text-accent-green">↑ {stats.liked_count}</span>
                <span className="text-white/40">liked</span>
              </div>
              <div className="flex gap-3">
                <span className="text-red-400">↓ {stats.disliked_count}</span>
                <span className="text-white/40">erased</span>
              </div>
              {stats.liked_genres.length > 0 && (
                <div className="text-yellow-400/70 mt-1">
                  ♥ {stats.liked_genres.slice(0, 2).join(", ")}
                </div>
              )}
              {stats.disliked_genres.length > 0 && (
                <div className="text-red-400/70">
                  ✕ {stats.disliked_genres.slice(0, 2).join(", ")}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Hover tooltip */}
        {hoveredNode && tooltip && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="fixed z-50 bg-bg-base/95 backdrop-blur-md border border-white/15 rounded-xl px-3 py-2 pointer-events-none"
            style={{ left: tooltip.x + 14, top: tooltip.y - 40 }}
          >
            <div className="text-white font-medium text-sm">{hoveredNode.label}</div>
            <div className="flex items-center gap-2 mt-0.5">
              <div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: NODE_CONFIG[hoveredNode.type]?.color }}
              />
              <span className="text-white/50 text-xs capitalize">{hoveredNode.type}</span>
              {hoveredNode.weight > 0 && (
                <span className="text-white/40 text-xs">· weight {hoveredNode.weight.toFixed(3)}</span>
              )}
            </div>
          </motion.div>
        )}

        {/* Zoom indicator */}
        <div className="absolute top-[72px] right-4 text-xs text-white/30 font-inter">
          {Math.round(zoom * 100)}%
        </div>
      </motion.div>
    </motion.div>
  );
}
