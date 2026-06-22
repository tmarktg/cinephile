import { useState, useCallback, useRef, useEffect } from "react";
import MovieCard from "./components/MovieCard.jsx";
import { getRecommendations, getSimilar } from "./api.js"; // streamRecommendations commented out

export default function App() {
  const [query, setQuery] = useState("");
  const [turns, setTurns] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading] = useState(false);
  // const [streamText, setStreamText] = useState("");  // unused while streaming is commented out
  const [error, setError] = useState(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    if (turns.length > 0) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [turns]);

  const search = useCallback(async (q) => {
    if (!q.trim()) return;
    setLoading(true);
    setError(null);
    setQuery("");
    try {
      const data = await getRecommendations({ query: q, sessionId });
      setSessionId(data.session_id);
      setTurns((prev) => [
        ...prev,
        { label: q, results: data.results, degraded: data.degraded },
      ]);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  // Streaming search — commented out
  // const search = useCallback(async (q) => {
  //   if (!q.trim()) return;
  //   setLoading(true);
  //   setStreamText("");
  //   setError(null);
  //   setQuery("");
  //   try {
  //     for await (const event of streamRecommendations({ query: q, sessionId })) {
  //       if (event.type === "chunk") {
  //         setStreamText((prev) => prev + event.text);
  //       } else if (event.type === "thinking") {
  //         setStreamText("…");
  //       } else if (event.type === "done") {
  //         setSessionId(event.session_id);
  //         setStreamText("");
  //         setTurns((prev) => [
  //           ...prev,
  //           { label: q, results: event.results, degraded: event.degraded },
  //         ]);
  //       }
  //     }
  //   } catch (e) {
  //     setError(e.message);
  //   } finally {
  //     setLoading(false);
  //     setStreamText("");
  //   }
  // }, [sessionId]);

  const handleSubmit = (e) => {
    e.preventDefault();
    search(query);
  };

  const handleSimilar = useCallback(async (movieId, movieTitle) => {
    setLoading(true);
    setError(null);
    try {
      const data = await getSimilar(movieId);
      setTurns((prev) => [
        ...prev,
        { label: `Similar to ${movieTitle}`, results: data.results, degraded: false },
      ]);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleNewConversation = () => {
    setTurns([]);
    setSessionId(null);
    setQuery("");
    setError(null);
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Cinephile</h1>
        <p className="subtitle">Describe what you want to watch</p>
      </header>

      <main>
        <div className="search-bar">
          <form className="search-form" onSubmit={handleSubmit}>
            <input
              className="search-input"
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={
                turns.length > 0
                  ? "Refine, follow up, or try something new…"
                  : "something dark and philosophical but not too slow…"
              }
              autoFocus
            />
            <button className="search-btn" type="submit" disabled={loading}>
              {loading ? "Searching…" : "Search"}
            </button>
          </form>
          {turns.length > 0 && (
            <button className="new-conversation-btn" onClick={handleNewConversation}>
              New conversation
            </button>
          )}
        </div>

        {/* Stream preview — commented out
        {streamText && (
          <div className="stream-preview">
            <pre className="stream-text">{streamText}</pre>
          </div>
        )}
        */}

        {error && <div className="error-banner">Error: {error}</div>}

        {turns.length > 0 && (
          <div className="conversation">
            {turns.map((turn, i) => (
              <section key={i} className="results-section">
                {i > 0 && <div className="turn-divider" />}
                <div className="results-header">
                  <span className="context-label">{turn.label}</span>
                  {turn.degraded && (
                    <span
                      className="degraded-badge"
                      title="LLM unavailable — showing vector-ranked results"
                    >
                      Degraded mode
                    </span>
                  )}
                </div>
                <div className="results-grid">
                  {turn.results.map((movie) => (
                    <MovieCard
                      key={movie.tmdb_id}
                      movie={movie}
                      onSimilar={handleSimilar}
                    />
                  ))}
                </div>
              </section>
            ))}
            <div ref={bottomRef} />
          </div>
        )}

        {!loading && !streamText && turns.length === 0 && !error && (
          <div className="empty-state">
            <p>Ask for anything. "A slow burn heist in 1970s New York."</p>
          </div>
        )}
      </main>
    </div>
  );
}
