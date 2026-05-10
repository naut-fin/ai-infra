UPDATE document_chunks
SET
    title = COALESCE(
            metadata->>'article_title',
    metadata->>'title',
    title
  ),
    section_path = COALESCE(
            metadata->>'section_path',
    metadata->>'section_title',
    section_path
  );

UPDATE document_chunks
SET search_text =
        concat_ws(
                E'\n',
                title,
                section_path,
                content
        );


UPDATE document_chunks
SET search_vector =
        setweight(to_tsvector('english', COALESCE(title, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(section_path, '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(content, '')), 'C');