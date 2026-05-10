ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS title text,
    ADD COLUMN IF NOT EXISTS section_path text,
    ADD COLUMN IF NOT EXISTS search_text text,
    ADD COLUMN IF NOT EXISTS search_vector tsvector;