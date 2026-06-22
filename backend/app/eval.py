"""Retrieval eval harness. Run: python -m app.eval"""
from app.embeddings import embed_query
from app.retrieval import search

# Hand-written queries with known-relevant TMDB IDs.
# These cover a range of mood/genre/style descriptors to stress the embedding space.
# Relevant IDs were seeded by hand then expanded with LLM-as-judge (score ≥ 4/5)
# via scripts/expand_eval_ids.py — making recall@k meaningful against what's
# actually in the collection rather than a narrow pre-curated list.
TEST_QUERIES = [
    {"query": "dark psychological thriller about identity and obsession", "relevant_ids": [274, 311, 539, 550, 807, 1018, 49026, 77338, 419430, 1339713, 1439584, 1639398]},
    {"query": "heartwarming animated family film", "relevant_ids": [12, 585, 862, 920, 1267, 9836, 10681, 13417, 28178, 150540, 467914, 529203, 592983, 1500099]},
    {"query": "epic science fiction space opera", "relevant_ids": [11, 62, 563, 841, 1891, 1892, 54138, 140607, 181808, 181812, 188927, 339964, 438631, 693134, 1170608]},
    {"query": "romantic comedy with witty dialogue", "relevant_ids": [105, 509, 4951, 9273, 9603, 10184, 13475, 13808, 50546, 884605, 1034716, 1511549, 1515615]},
    {"query": "gritty crime drama set in a city", "relevant_ids": [101, 187, 238, 240, 278, 388, 680, 769, 6977, 479718, 581528, 1128650, 1245700, 1306845, 1426964, 1472951]},
    {"query": "slow burn existential drama about loneliness", "relevant_ids": [120, 194, 77338, 103663, 313369, 334541, 371462, 372058, 858017]},
    {"query": "action-packed superhero blockbuster", "relevant_ids": [1724, 1726, 1979, 24428, 141052, 271110, 284053, 284054, 299534, 299536, 383498, 464052, 594767, 969681, 1003596, 1003598]},
    {"query": "horror film with supernatural elements and dread", "relevant_ids": [539, 694, 8329, 9392, 256274, 346364, 475557, 493922, 864370, 1092936, 1219739, 1363387]},
    {"query": "war film showing the human cost of conflict", "relevant_ids": [424, 637, 857, 1150, 9567, 11324, 25237, 49046, 228150, 530915]},
    {"query": "quirky indie drama with offbeat humor", "relevant_ids": [8909, 14777, 20453, 37735, 102651, 244786, 293863, 1391074, 1560230]},
    {"query": "heist movie with clever twists", "relevant_ids": [161, 163, 489, 522, 629, 9340, 9654, 75656, 290859, 291805, 425274, 941109, 1171145]},
    {"query": "coming of age story about teenagers finding themselves", "relevant_ids": [62, 103, 4960, 9377, 15804, 37735, 299710, 398818, 399174, 449176, 575813, 812037, 851644, 1008953, 1515615, 1542352]},
    {"query": "mind-bending sci-fi about time travel or alternate realities", "relevant_ids": [1903, 27205, 49530, 58244, 109445, 157336, 220289, 264660, 438631, 577922, 1032892, 1119449, 1377650, 1590097]},
    {"query": "historical epic about ancient civilizations", "relevant_ids": [98, 197, 652, 665, 1452, 1966, 256835, 339408, 856289, 1196943, 1368337]},
    {"query": "documentary style realist drama about working class life", "relevant_ids": [399, 1402, 14531, 22970, 76341, 244786, 441168, 1097714, 1520769]},
    {"query": "musical with show-stopping dance numbers", "relevant_ids": [88, 1585, 4348, 14836, 313369, 420818, 611213]},
    {"query": "dark comedy with satirical bite", "relevant_ids": [539, 550, 680, 769, 4247, 419430, 475557, 639988, 1628448]},
    {"query": "animated film for adults with mature themes", "relevant_ids": [129, 4935, 14160, 76726, 315162, 508439]},
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
