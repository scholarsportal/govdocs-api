create table if not exists documents (
  id uuid primary key default gen_uuid_v4(),
  title text not null,
  ia_link text not null,
  barcode bigint not null,
  created_at timestamp with time zone default now() not null
);

create index if not exists idx_documents_barcode on documents(barcode);

create enum if not exists ocr_status as ('pending', 'processing', 'completed', 'error');
create enum if not exists ocr_model as ('tesseract', 'marker', 'gotocr', 'olmocr');

create table if not exists ocr_jobs (
  id bigseral primary key,
  document_id uuid references documents(id) not null,
  page_number integer not null,
  ocr_output text not null,
  ocr_model ocr_model not null,
  ocr_config jsonb not null, 
  status ocr_status not null,
  created_at timestamp with time zone default now() not null
);

create index if not exists idx_ocr_jobs_document_id on ocr_jobs(document_id);
create index if not exists idx_ocr_document_id_page_number on ocr_jobs(document_id, page_number);
create index if not exists idx_ocr_model on ocr_jobs(ocr_model);


create table if not exists ocr_evaluation_metrics (
  id bigserial primary key,
  document_id uuid references documents(id) not null,
  ocr_job_id bigint references ocr_jobs(id) not null,

  format_quality integer check (format_quality >= 0 and format_quality <= 5),
  format_quality_comment text,
  
  output_vs_ground_truth integer check (output_vs_ground_truth >= 0 and output_vs_ground_truth <= 5),
  output_vs_ground_truth_comment text,
  
  table_parsing_capabilities integer check (table_parsing_capabilities >= 0 and table_parsing_capabilities <= 5),
  table_parsing_capabilities_comment text,
  
  hallucination integer check (hallucination >= 0 and hallucination <= 5),
  hallucination_comment text,
  
  evaluators_overall_comment text not null,
  
  evaluation_submitted boolean default false not null,
  evaluation_submitted_by uuid references auth.users(id) not null,
  
  created_at timestamp with time zone default now() not null,

  constraint submit_check check (
    evaluation_submitted = true and
    (format_quality is not null and format_quality_comment is not null) and
    (output_vs_ground_truth is not null and output_vs_ground_truth_comment is not null) and
    (table_parsing_capabilities is not null and table_parsing_capabilities_comment is not null) and
    (hallucination is not null and hallucination_comment is not null)
  )
);

create index if not exists idx_ocr_evaluation_metrics_document_id on ocr_evaluation_metrics(document_id);