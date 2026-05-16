ROUTING_EVAL_QUERIES = [

    # =========================================================
    # HYBRID QUERIES
    # Exact entities, dates, acronyms, numbers, identifiers
    # Wikipedia-compatible
    # =========================================================

    {"query": "What happened in the 1998 FIFA World Cup?", "expected_route": "hybrid"},
    {"query": "Who won the 1984 Summer Olympics men's marathon?", "expected_route": "hybrid"},
    {"query": "What caused the 2008 financial crisis?", "expected_route": "hybrid"},
    {"query": "Explain the significance of Apollo 11", "expected_route": "hybrid"},
    {"query": "What is ISO 8601 date format?", "expected_route": "hybrid"},
    {"query": "What does HTTP 404 mean?", "expected_route": "hybrid"},
    {"query": "Who was president of the United States in 1969?", "expected_route": "hybrid"},
    {"query": "What happened during World War II?", "expected_route": "hybrid"},
    {"query": "Explain RAID 5 and RAID 6 differences", "expected_route": "hybrid"},
    {"query": "What is the meaning of DNA?", "expected_route": "hybrid"},
    {"query": "What is the significance of Area 51?", "expected_route": "hybrid"},
    {"query": "What happened at Chernobyl in 1986?", "expected_route": "hybrid"},
    {"query": "What does CPU stand for?", "expected_route": "hybrid"},
    {"query": "Explain the role of NATO during the Cold War", "expected_route": "hybrid"},
    {"query": "What happened in the 1929 stock market crash?", "expected_route": "hybrid"},
    {"query": "Who discovered DNA structure in 1953?", "expected_route": "hybrid"},
    {"query": "What is the significance of Unix?", "expected_route": "hybrid"},
    {"query": "What caused the Y2K scare?", "expected_route": "hybrid"},
    {"query": "What is the purpose of the CPU cache?", "expected_route": "hybrid"},
    {"query": "Who won the Nobel Prize in Physics in 1921?", "expected_route": "hybrid"},
    {"query": "What is the meaning of RAM in computers?", "expected_route": "hybrid"},
    {"query": "What happened during the Cuban Missile Crisis?", "expected_route": "hybrid"},
    {"query": "Explain the significance of the Hubble Space Telescope", "expected_route": "hybrid"},
    {"query": "What does UNESCO stand for?", "expected_route": "hybrid"},
    {"query": "What happened during the 2011 Tōhoku earthquake and tsunami?", "expected_route": "hybrid"},


    # =========================================================
    # SEMANTIC QUERIES
    # Conceptual, broad, paraphrased, meaning-driven
    # =========================================================

    {"query": "Why do civilizations collapse over time?", "expected_route": "semantic"},
    {"query": "How do economic recessions affect society?", "expected_route": "semantic"},
    {"query": "Why are historical events interpreted differently?", "expected_route": "semantic"},
    {"query": "How does evolution shape living organisms?", "expected_route": "semantic"},
    {"query": "Why do empires rise and fall?", "expected_route": "semantic"},
    {"query": "How does climate affect human civilization?", "expected_route": "semantic"},
    {"query": "Why are some scientific discoveries controversial?", "expected_route": "semantic"},
    {"query": "How do technological advances change society?", "expected_route": "semantic"},
    {"query": "Why do wars have long-term economic consequences?", "expected_route": "semantic"},
    {"query": "How does language evolve over time?", "expected_route": "semantic"},
    {"query": "Why are some leaders remembered more favorably than others?", "expected_route": "semantic"},
    {"query": "How do religions influence cultures?", "expected_route": "semantic"},
    {"query": "Why do societies create myths and legends?", "expected_route": "semantic"},
    {"query": "How does geography influence political power?", "expected_route": "semantic"},
    {"query": "Why do financial bubbles occur repeatedly?", "expected_route": "semantic"},
    {"query": "How do inventions spread across the world?", "expected_route": "semantic"},
    {"query": "Why do humans explore space?", "expected_route": "semantic"},
    {"query": "How does propaganda influence public opinion?", "expected_route": "semantic"},
    {"query": "Why do some technologies become obsolete?", "expected_route": "semantic"},
    {"query": "How do pandemics change societies?", "expected_route": "semantic"},
    {"query": "Why do different cultures develop different traditions?", "expected_route": "semantic"},
    {"query": "How do political systems evolve?", "expected_route": "semantic"},
    {"query": "Why do revolutions happen?", "expected_route": "semantic"},
    {"query": "How do scientific theories change over time?", "expected_route": "semantic"},
    {"query": "Why do humans preserve historical monuments?", "expected_route": "semantic"},
]