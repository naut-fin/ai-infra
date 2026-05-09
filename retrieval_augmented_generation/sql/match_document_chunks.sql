CREATE OR REPLACE FUNCTION match_document_chunks(
  query_embedding vector(1536),
  match_count int,
  ef_search int
)
RETURNS TABLE (
  chunk_id uuid,
  document_id uuid,
  chunk_index int,
  title text,
  content text,
  metadata jsonb,
  similarity double precision
)
LANGUAGE plpgsql
AS $$
BEGIN
  PERFORM set_config('hnsw.ef_search', ef_search::text, true);

RETURN QUERY
SELECT
    dc.id AS chunk_id,
    dc.document_id,
    dc.chunk_index,
    d.title,
    dc.content,
    dc.metadata,
    1 - (dc.embedding <=> query_embedding) AS similarity
FROM document_chunks dc
         JOIN documents d ON d.id = dc.document_id
WHERE dc.embedding IS NOT NULL
ORDER BY dc.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;