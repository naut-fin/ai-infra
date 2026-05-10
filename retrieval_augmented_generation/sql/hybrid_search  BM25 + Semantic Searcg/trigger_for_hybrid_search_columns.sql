CREATE OR REPLACE FUNCTION update_document_chunks_search_fields()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  NEW.title := COALESCE(
    NEW.title,
    NEW.metadata->>'article_title',
    NEW.metadata->>'title'
  );

  NEW.section_path := COALESCE(
    NEW.section_path,
    NEW.metadata->>'section_path',
    NEW.metadata->>'section_title'
  );

  NEW.search_text := concat_ws(
    E'\n',
    NEW.title,
    NEW.section_path,
    NEW.content
  );

  NEW.search_vector :=
    setweight(to_tsvector('english', COALESCE(NEW.title, '')), 'A') ||
    setweight(to_tsvector('english', COALESCE(NEW.section_path, '')), 'B') ||
    setweight(to_tsvector('english', COALESCE(NEW.content, '')), 'C');

RETURN NEW;
END;
$$;


DROP TRIGGER IF EXISTS trg_update_document_chunks_search_fields
ON document_chunks;

CREATE TRIGGER trg_update_document_chunks_search_fields
    BEFORE INSERT OR UPDATE OF title, section_path, content, metadata
                     ON document_chunks
                         FOR EACH ROW
                         EXECUTE FUNCTION update_document_chunks_search_fields();