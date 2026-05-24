import { NextRequest } from 'next/server';

export const dynamic = 'force-dynamic';

export async function POST(_req: NextRequest) {
  const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
  const res = await fetch(`${backendUrl}/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ confirm: true }),
  });
  const data = await res.json();
  return Response.json(data);
}