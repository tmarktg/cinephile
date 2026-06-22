"""Retrieval eval harness. Run: python -m app.eval"""
from app.embeddings import embed_query
from app.retrieval import search

# Hand-written queries with known-relevant TMDB IDs.
# These cover a range of mood/genre/style descriptors to stress the embedding space.
TEST_QUERIES = [
    {
        "query": "dark psychological thriller about identity and obsession",
        "relevant_ids": [550, 274, 311, 49026, 77338],  # Fight Club, The Silence of the Lambs, etc.
    },
    {
        "query": "heartwarming animated family film",
        "relevant_ids": [862, 585, 10681, 12, 920],  # Toy Story, Monsters Inc, WALL-E, Finding Nemo
    },
    {
        "query": "epic science fiction space opera",
        "relevant_ids": [11, 1891, 1892, 140607, 181808],  # Star Wars
    },
    {
        "query": "romantic comedy with witty dialogue",
        "relevant_ids": [13475, 9273, 105, 4951, 13808],
    },
    {
        "query": "gritty crime drama set in a city",
        "relevant_ids": [769, 240, 238, 278, 680],  # Goodfellas, Godfather, Shawshank, Pulp Fiction
    },
    {
        "query": "slow burn existential drama about loneliness",
        "relevant_ids": [194, 372058, 77338, 313369, 120],
    },
    {
        "query": "action-packed superhero blockbuster",
        "relevant_ids": [299536, 299534, 284053, 284054, 271110],  # Avengers, etc.
    },
    {
        "query": "horror film with supernatural elements and dread",
        "relevant_ids": [694, 346364, 493922, 475557, 539],
    },
    {
        "query": "war film showing the human cost of conflict",
        "relevant_ids": [857, 637, 424, 1150, 11324],  # Apocalypse Now, Schindler's List
    },
    {
        "query": "quirky indie drama with offbeat humor",
        "relevant_ids": [102651, 293863, 14777, 8909, 244786],
    },
    {
        "query": "heist movie with clever twists",
        "relevant_ids": [161, 9340, 489, 290859, 522],  # Ocean's Eleven, etc.
    },
    {
        "query": "coming of age story about teenagers finding themselves",
        "relevant_ids": [9377, 4960, 62, 399174, 103],
    },
    {
        "query": "mind-bending sci-fi about time travel or alternate realities",
        "relevant_ids": [27205, 157336, 264660, 438631, 109445],  # Inception, Interstellar
    },
    {
        "query": "historical epic about ancient civilizations",
        "relevant_ids": [1452, 98, 339408, 197, 256835],
    },
    {
        "query": "documentary style realist drama about working class life",
        "relevant_ids": [244786, 14531, 22970, 76341, 399],
    },
    {
        "query": "musical with show-stopping dance numbers",
        "relevant_ids": [313369, 14836, 420818, 1585, 4348],  # La La Land, etc.
    },
    {
        "query": "dark comedy with satirical bite",
        "relevant_ids": [680, 550, 539, 769, 475557],
    },
    {
        "query": "animated film for adults with mature themes",
        "relevant_ids": [129, 4935, 14160, 315162, 508439],  # Spirited Away, etc.
    },
]


def recall_at_k(relevant: set[int], retrieved: list[int], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(relevant & top_k) / len(relevant)


def run_eval(k: int = 10) -> None:
    print(f"\nRetrieval Eval — recall@{k}\n{'='*50}")
    print(f"{'Query':<50} {'recall@' + str(k):>10}")
    print("-" * 62)

    scores = []
    for item in TEST_QUERIES:
        query = item["query"]
        relevant = set(item["relevant_ids"])
        vector = embed_query(query)
        candidates = search(vector, k=k)
        retrieved_ids = [c["tmdb_id"] for c in candidates]
        score = recall_at_k(relevant, retrieved_ids, k)
        scores.append(score)
        label = query[:48] + ".." if len(query) > 50 else query
        print(f"{label:<50} {score:>10.2f}")

    avg = sum(scores) / len(scores) if scores else 0.0
    print("-" * 62)
    print(f"{'Average':<50} {avg:>10.2f}")
    print(f"\nTotal queries: {len(TEST_QUERIES)}, k={k}\n")


if __name__ == "__main__":
    import sys
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_eval(k=k)
