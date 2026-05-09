from retrieval_augmented_generation.sample_rag.rag_supabase import search_documents

queries = [
    "What is photosynthesis?",
    "Who was Albert Einstein?",
    "What is gravity?",
    "How do volcanoes form?",
    "What is a computer?",
    "What is World War II?",
]

for query in queries:
    print("\n==============================")
    print("QUERY:", query)
    print("==============================")

    results = search_documents(
        query=query,
        match_count=5,
        similarity_threshold=0.4,
    )

    for i, r in enumerate(results, start=1):
        print(f"\n#{i}")
        print("Title:", r["title"])
        print("Similarity:", round(r["similarity"], 4))
        print("Content:", r["content"][:500])