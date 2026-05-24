import { NextRequest } from 'next/server';

export const dynamic = 'force-dynamic';

export async function GET(_req: NextRequest) {
  const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
  try {
    const res = await fetch(`${backendUrl}/graph`, { cache: 'no-store' });
    const data = await res.json();
    return Response.json(data);
  } catch {
    return Response.json({
      nodes: [], edges: [], genre_weights: {},
      stats: { liked_count: 0, disliked_count: 0, liked_genres: [], disliked_genres: [] }
    });
  }
}