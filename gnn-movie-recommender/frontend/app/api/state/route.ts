import { NextRequest } from 'next/server';

export async function GET(_req: NextRequest) {
  const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
  const res = await fetch(`${backendUrl}/state`, { cache: 'no-store' });
  const data = await res.json();
  return Response.json(data);
}
