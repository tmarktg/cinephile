import { useState, useCallback } from "react";
import MovieCard from "./components/MovieCard.jsx";
import { getRecommendations, getSimilar } from "./api.js";

export default function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [degraded, setDegraded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [contextLabel, setContextLabel] = useState(null);

  const search = useCallback(async (q) => {
    if (!q.trim()) return;
    setLoading(true);
    setError(null);
    setContextLabel(q);
    try {
      const data = await getRecommendations({ query: q });
      setResults(data.results);
      setDegraded(data.degraded);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    search(query);
  };

  const handleSimilar = useCallback(async (movieId) => {
    setLoading(true);
    setError(null);
    setContextLabel(`Similar to movie #${movieId}`);
    try {
      const data = await getSimilar(movieId);
      setResults(data.results);
      setDegraded(false);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Cinephile</h1>
        <p className="subtitle">Describe what you want to watch</p>
      </header>

      <main>
        <form className="search-form" onSubmit={handleSubmit}>
          <input
            className="search-input"
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="something dark and philosophical but not too slow…"
            autoFocus
          />
          <button className="search-btn" type="submit" disabled={loading}>
            {loading ? "Searching…" : "Search"}
          </button>
        </form>

        {error && <div className="error-banner">Error: {error}</div>}

        {results.length > 0 && (
          <section className="results-section">
            <div className="results-header">
              {contextLabel && (
                <span className="context-label">{contextLabel}</span>
              )}
              {degraded && (
                <span className="degraded-badge" title="LLM unavailable — showing vector-ranked results">
                  Degraded mode
                </span>
              )}
            </div>
            <div className="results-grid">
              {results.map((movie) => (
                <MovieCard
                  key={movie.tmdb_id}
                  movie={movie}
                  onSimilar={handleSimilar}
                />
              ))}
            </div>
          </section>
        )}

        {!loading && results.length === 0 && !error && (
          <div className="empty-state">
            <p>Ask for anything. "A slow burn heist in 1970s New York."</p>
          </div>
        )}
      </main>
    </div>
  );
}
