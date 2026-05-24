import { NextRequest } from 'next/server';

export const maxDuration = 60;

export async function POST(req: NextRequest) {
  try {
    const { message } = await req.json();

    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';

    const backendRes = await fetch(`${backendUrl}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });

    if (!backendRes.ok) {
      const err = await backendRes.text();
      return new Response(JSON.stringify({ error: err }), { status: 500 });
    }

    // Stream the SSE from Python directly to the client
    return new Response(backendRes.body, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
      },
    });
  } catch (error: any) {
    console.error('[/api/chat]', error);
    return new Response(JSON.stringify({ error: 'Backend unreachable' }), { status: 500 });
  }
}
