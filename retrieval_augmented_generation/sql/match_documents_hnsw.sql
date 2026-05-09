CREATE OR REPLACE FUNCTION match_document_hnsw(
query_embedding vector(1536),
match_count int,
ef_search int
)
RETURNS TABLE (
chunk_id uuid,
document_id uuid,
title text,
content text,
similarity double precision
)
LANGUAGE plpgsql
AS $$
BEGIN
  PERFORM set_config('hnsw.ef_search', ef_search::text, true);

RETURN QUERY
Select
    dc.id as chunk_id,
    dc.document_id,
    d.title,
    dc.content,
    1 - (dc.embedding <=> query_embedding) as similarity
from document_chunks dc
         join documents d on d.id = dc.document_id
where dc.embedding is not null
order by dc.embedding <=> query_embedding
  limit match_count;
END;
$$;
