/**
 * API Client — Typed fetch wrappers for Rocky (graph recs + selective forgetting)
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Types ──

export interface UserInfo {
  user_id: number;
  age: number;
  gender: string;
  occupation: string;
  num_movies_watched: number;
}

export interface MovieInfo {
  movie_id: number;
  title: string;
  genres: string[];
  franchise: string | null;
  rating?: number;
  score?: number;
  reason?: string | null;
}

export interface Recommendation {
  movie_id: number;
  title: string;
  genres: string[];
  franchise: string | null;
  score: number;
  reason: string | null;
}

export interface RecommendResponse {
  user_id: number;
  shard_id: number;
  recommendations: Recommendation[];
  unlearn_applied: boolean;
  excluded_franchises: string[];
}

export interface HistoryResponse {
  user_id: number;
  user_info: {
    user_id: number;
    age: number;
    gender: string;
    occupation: string;
  };
  history: MovieInfo[];
  total_watched: number;
}

export interface UnlearnRequest {
  user_id: number;
  movie_id: number;
  scope: "movie" | "franchise";
}

export interface UnlearnResponse {
  success: boolean;
  level: number;
  method: string;
  user_id: number;
  movie_id: number;
  movie_title: string | null;
  scope: string;
  franchise: string | null;
  shard_id: number;
  message: string;
}

export interface ExplainResponse {
  user_id: number;
  movie_id: number;
  movie_title: string;
  explanation: string;
  excluded: boolean;
  exclusion_reason: string | null;
}

export interface GraphNode {
  id: string;
  type: "user" | "movie" | "genre" | "franchise";
  label: string;
  genres?: string[];
  franchise?: string;
  occupation?: string;
}

export interface GraphLink {
  source: string;
  target: string;
  type: "watched" | "has_genre" | "in_franchise";
  rating?: number;
}

export interface GraphResponse {
  user_id: number;
  nodes: GraphNode[];
  links: GraphLink[];
}

export interface MovieSearchResult {
  movie_id: number;
  title: string;
  genres: string[];
  franchise: string | null;
}

export interface DislikesResponse {
  user_id: number;
  disliked_movies: { movie_id: number; title: string; franchise: string | null }[];
  disliked_franchises: string[];
}

// ── API Functions ──

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `API error: ${res.status}`);
  }

  return res.json();
}

export async function fetchUsers(limit = 20): Promise<{ users: UserInfo[]; total: number }> {
  return apiFetch(`/users?limit=${limit}`);
}

export async function fetchHistory(userId: number): Promise<HistoryResponse> {
  return apiFetch(`/history/${userId}`);
}

export async function fetchRecommendations(userId: number, k = 10): Promise<RecommendResponse> {
  return apiFetch(`/recommend/${userId}?k=${k}`);
}

export async function triggerUnlearn(request: UnlearnRequest): Promise<UnlearnResponse> {
  return apiFetch("/unlearn", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function fetchExplanation(userId: number, movieId: number): Promise<ExplainResponse> {
  return apiFetch(`/explain/${userId}/${movieId}`);
}

export async function fetchGraph(userId: number, depth = 1): Promise<GraphResponse> {
  return apiFetch(`/graph/${userId}?depth=${depth}`);
}

export async function searchMovies(query: string): Promise<{ query: string; results: MovieSearchResult[] }> {
  return apiFetch(`/movies/search?q=${encodeURIComponent(query)}`);
}

export async function fetchDislikes(userId: number): Promise<DislikesResponse> {
  return apiFetch(`/dislikes/${userId}`);
}

export async function clearDislikes(userId: number): Promise<{ success: boolean; message: string }> {
  return apiFetch(`/dislikes/${userId}`, { method: "DELETE" });
}

export async function healthCheck(): Promise<{
  status: string;
  service: string;
  systems?: string;
}> {
  return apiFetch("/health");
}
