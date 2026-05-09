CREATE TABLE public.document_chunks (
                                        id uuid NOT NULL DEFAULT gen_random_uuid(),
                                        document_id uuid NOT NULL,
                                        chunk_index integer(32,0) NOT NULL,
                                        content text NOT NULL,
                                        token_count integer(32,0),
                                        metadata jsonb DEFAULT '{}'::jsonb,
                                        embedding vector,
                                        created_at timestamp with time zone DEFAULT now()
);

CREATE INDEX document_chunks_document_id_idx ON public.document_chunks USING btree (document_id)

CREATE UNIQUE INDEX document_chunks_pkey ON public.document_chunks USING btree (id)


<!--- hnsw index >
CREATE INDEX document_chunks_embedding_hnsw_idx ON public.document_chunks USING hnsw (embedding vector_cosine_ops)