CREATE INDEX IF NOT EXISTS document_chunks_search_vector_gin_idx
    ON document_chunks
    USING gin (search_vector);

CREATE INDEX IF NOT EXISTS document_chunks_title_idx
ON document_chunks (title);

CREATE INDEX IF NOT EXISTS document_chunks_section_path_idx
    ON document_chunks (section_path);

CREATE INDEX IF NOT EXISTS document_chunks_document_chunk_idx
    ON document_chunks (document_id, chunk_index);