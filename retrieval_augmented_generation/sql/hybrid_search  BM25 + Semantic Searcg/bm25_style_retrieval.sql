CREATE OR REPLACE FUNCTION match_chunks_lexical(
  query_text text,
  match_count int DEFAULT 50,
  target_chunking_strategy text DEFAULT NULL
)
RETURNS TABLE (
  chunk_id uuid,
  document_id uuid,
  chunk_index int,
  title text,
  section_path text,
  content text,
  metadata jsonb,
  lexical_rank real
)
LANGUAGE plpgsql
AS $$
BEGIN
RETURN QUERY
    WITH query AS (
    SELECT websearch_to_tsquery('english', query_text) AS q
  )
SELECT
    dc.id AS chunk_id,
    dc.document_id,
    dc.chunk_index,
    dc.title,
    dc.section_path,
    dc.content,
    dc.metadata,
    ts_rank_cd(dc.search_vector, query.q) AS lexical_rank
FROM document_chunks dc, query
WHERE dc.search_vector @@ query.q
  AND (
    target_chunking_strategy IS NULL
   OR dc.metadata->>'chunking_strategy' = target_chunking_strategy
    )
ORDER BY lexical_rank DESC
    LIMIT match_count;
END;
$$;