-- Create a storage bucket for internet archive rendered images
insert into storage.buckets (id, name, public)
values ('ia_bucket', 'ia_bucket', true);

-- Policies for ia_bucket
-- Only admins can upload/modify/delete files in the ia_bucket
-- create policy "Admin users can upload to ia_bucket"
-- on storage.objects for insert to authenticated with check (
--   bucket_id = 'ia_bucket' and
--   (select position_title from app_users where id = auth.uid()) = 'admin'
-- );

-- create policy "Admin users can update in ia_bucket"
-- on storage.objects for update to authenticated using (
--   bucket_id = 'ia_bucket' and
--   (select position_title from app_users where id = auth.uid()) = 'admin'
-- );

-- create policy "Admin users can delete from ia_bucket"
-- on storage.objects for delete to authenticated using (
--   bucket_id = 'ia_bucket' and
--   (select position_title from app_users where id = auth.uid()) = 'admin'
-- );

-- -- Anyone can view the rendered images
-- create policy "Anyone can view ia_bucket"
-- on storage.objects for select to authenticated using (
--   bucket_id = 'ia_bucket'
-- );

-- Temporary changes, prod should use above policies
create policy "Admin users can upload to ia_bucket"
on storage.objects for insert to authenticated, anon with check (
  bucket_id = 'ia_bucket' 
);

create policy "Admin users can update in ia_bucket"
on storage.objects for update to authenticated, anon using (
  bucket_id = 'ia_bucket' 
);

create policy "Admin users can delete from ia_bucket"
on storage.objects for delete to authenticated, anon using (
  bucket_id = 'ia_bucket'
);

-- Anyone can view the rendered images
create policy "Anyone can view ia_bucket"
on storage.objects for select to authenticated, anon using (
  bucket_id = 'ia_bucket'
);

-- Create a table to track documents processing status
create table document_processing (
  id serial primary key,
  document_id uuid references documents(id) not null,
  status text not null check (status in ('pending', 'processing', 'completed', 'error')),
  pages_processed int not null default 0,
  total_pages int,
  error_message text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter publication supabase_realtime add table document_processing;

create index idx_document_processing_document_id on document_processing(document_id);
create index idx_document_processing_status on document_processing(status);
