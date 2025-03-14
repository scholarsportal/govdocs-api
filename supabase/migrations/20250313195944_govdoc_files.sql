create enum if not exists govdoc_document_status as ('Not Processed', 'Processing Queued', 'Processing', 'Processing Error', 'Completed');

create table if not exists documents (
  id uuid primary key default gen_uuid_v4(),
  title text not null,
  ia_link text not null,
  barcode bigint not null,
  status govdoc_document_status not null default 'Not Processed',
  status_message text,
  status_progress integer default 0,
  created_at timestamp with time zone default now() not null
);
