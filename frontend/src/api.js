const BASE = import.meta.env.VITE_API_URL || "";

export async function* streamRecommendations({ query, k = 10, filters = null, sessionId = null }) {
  const body = { query, k };
  if (filters) body.filters = filters;
  if (sessionId) body.session_id = sessionId;

  const res = await fetch(`${BASE}/recommend/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${await res.text()}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try { yield JSON.parse(line.slice(6)); } catch { /* skip malformed */ }
      }
    }
  }
}

export async function getRecommendations({ query, k = 10, filters = null, sessionId = null }) {
  const body = { query, k };
  if (filters) body.filters = filters;
  if (sessionId) body.session_id = sessionId;

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
