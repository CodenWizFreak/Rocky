import { openai } from '@ai-sdk/openai';
import { streamText } from 'ai';

export const maxDuration = 30;

export async function POST(req: Request) {
  try {
    const { messages } = await req.json();

    const result = await streamText({
      model: openai('gpt-4o'),
      system: "You are the GNN Recommender System AI. You can recommend movies, and when requested to 'forget' or 'unlearn' a specific movie or genre, you confirm the operation is sent to the backend knowledge graph and retrains the shard. Keep responses brief, analytical, and professional.",
      messages,
    });

    return result.toTextStreamResponse();
  } catch (error) {
    return new Response(JSON.stringify({ error: 'Failed to connect to AI Provider' }), { status: 500 });
  }
}