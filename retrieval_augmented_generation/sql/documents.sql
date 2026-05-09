CREATE TABLE public.documents (
                                  id uuid NOT NULL DEFAULT gen_random_uuid(),
                                  title text NOT NULL,
                                  source_url text,
                                  metadata jsonb DEFAULT '{}'::jsonb,
                                  created_at timestamp with time zone DEFAULT now()
);

CREATE UNIQUE INDEX documents_pkey ON public.documents USING btree (id)