const POSTER_BASE = "https://image.tmdb.org/t/p/w342";

export default function MovieCard({ movie, onSimilar }) {
  const posterUrl = movie.poster_path
    ? `${POSTER_BASE}${movie.poster_path}`
    : null;

  return (
    <article className="movie-card">
      <div className="poster">
        {posterUrl ? (
          <img src={posterUrl} alt={movie.title} loading="lazy" />
        ) : (
          <div className="poster-placeholder">
            <span>{movie.title[0]}</span>
          </div>
        )}
      </div>
      <div className="card-body">
        <div className="card-header">
          <h3 className="title">{movie.title}</h3>
          <div className="meta">
            {movie.year && <span className="year">{movie.year}</span>}
            {movie.runtime && <span className="runtime">{movie.runtime}m</span>}
          </div>
        </div>
        {movie.genres?.length > 0 && (
          <div className="genres">
            {movie.genres.map((g) => (
              <span key={g} className="genre-tag">
                {g}
              </span>
            ))}
          </div>
        )}
        {movie.reason && <p className="reason">{movie.reason}</p>}
        <button
          className="similar-btn"
          onClick={() => onSimilar(movie.tmdb_id)}
          title="Find similar movies"
        >
          Similar
        </button>
      </div>
    </article>
  );
}
