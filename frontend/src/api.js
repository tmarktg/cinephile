const BASE = import.meta.env.VITE_API_URL || "";

export async function getRecommendations({ query, k = 10, filters = null }) {
  const body = { query, k };
  if (filters) body.filters = filters;

  const res = await fetch(`${BASE}/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }

  return res.json();
}

export async function getSimilar(movieId, k = 10) {
  const res = await fetch(`${BASE}/similar/${movieId}?k=${k}`);
  if (!res.ok) {
    throw new Error(`API error ${res.status}`);
  }
  return res.json();
}
