-- Seed data for research_registry.
-- Two data sources: investor slides (PDF) and Pillar 3 disclosures (XLSX)
-- for the Big 6 Canadian banks, Q4 2025 and Q1 2026.

-- Clean old entries
DELETE FROM research_registry WHERE data_source NOT IN ('investor_slides', 'pillar3_disclosure');

-- Investor Slides
INSERT INTO research_registry (
    data_source, display_name, source_summary, source_description, enabled,
    batch_size, max_selected_files,
    top_chunks_in_catalog_selection, top_chunks_in_metadata_research,
    max_parallel_files, max_chunks_per_file,
    max_pages_for_full_context, max_primary_section_page_count,
    max_subsection_page_count, max_neighbour_chunks, max_gap_fill_pages,
    enable_db_wide_deep_research, enable_dense_table_retrieval,
    filter_1_label, filter_1_description,
    filter_2_label, filter_2_description,
    filter_3_label, filter_3_description,
    metadata_context_fields, sample_questions
) VALUES (
    'investor_slides',
    'Investor Presentations',
    'Quarterly investor presentation slides from the Big 6 Canadian banks (RBC, TD, BMO, CIBC, BNS, NBC).',
    'PDF investor presentations covering quarterly financial results, capital metrics, segment performance, credit quality, and strategic outlook for each of the Big 6 Canadian banks. Each file is a single bank''s quarterly results deck.',
    true,
    12,   -- batch_size (all 12 files fit in one batch)
    6,    -- max_selected_files (up to 6 banks)
    1,    -- top_chunks_in_catalog_selection
    3,    -- top_chunks_in_metadata_research
    5,    -- max_parallel_files
    20,   -- max_chunks_per_file
    10,   -- max_pages_for_full_context (slides can be 30-40 pages)
    8,    -- max_primary_section_page_count
    4,    -- max_subsection_page_count
    2,    -- max_neighbour_chunks
    2,    -- max_gap_fill_pages
    true, -- enable_db_wide_deep_research
    false, -- enable_dense_table_retrieval (PDFs have no dense tables)
    'Reporting Period',
    'Year and quarter subfolder (e.g., 2025_Q4, 2026_Q1). Set to match the quarter referenced in the user''s query. Leave unset to search across all available periods.',
    'Bank',
    'Big 6 Canadian bank code: RBC, TD, BMO, CIBC, BNS (Scotiabank), NBC (National Bank). Set to match the bank in the user''s query. Leave unset for cross-bank comparisons or when the user asks about multiple banks.',
    '',
    '',
    ARRAY['document_summary', 'document_description'],
    '[
        "What is RBC''s CET1 ratio for Q1 2026?",
        "Compare net income across the Big 6 banks for Q1 2026",
        "How did TD''s credit losses change from Q4 2025 to Q1 2026?",
        "What is BMO''s return on equity?",
        "Which bank has the highest efficiency ratio?"
    ]'::jsonb
) ON CONFLICT (data_source) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    source_summary = EXCLUDED.source_summary,
    source_description = EXCLUDED.source_description,
    filter_1_label = EXCLUDED.filter_1_label,
    filter_1_description = EXCLUDED.filter_1_description,
    filter_2_label = EXCLUDED.filter_2_label,
    filter_2_description = EXCLUDED.filter_2_description,
    max_pages_for_full_context = EXCLUDED.max_pages_for_full_context,
    max_selected_files = EXCLUDED.max_selected_files,
    batch_size = EXCLUDED.batch_size,
    enable_dense_table_retrieval = EXCLUDED.enable_dense_table_retrieval,
    sample_questions = EXCLUDED.sample_questions,
    updated_at = CURRENT_TIMESTAMP;

-- Pillar 3 Regulatory Disclosures
INSERT INTO research_registry (
    data_source, display_name, source_summary, source_description, enabled,
    batch_size, max_selected_files,
    top_chunks_in_catalog_selection, top_chunks_in_metadata_research,
    max_parallel_files, max_chunks_per_file,
    max_pages_for_full_context, max_primary_section_page_count,
    max_subsection_page_count, max_neighbour_chunks, max_gap_fill_pages,
    enable_db_wide_deep_research, enable_dense_table_retrieval,
    filter_1_label, filter_1_description,
    filter_2_label, filter_2_description,
    filter_3_label, filter_3_description,
    metadata_context_fields, sample_questions
) VALUES (
    'pillar3_disclosure',
    'Pillar 3 Regulatory Disclosures',
    'Quarterly Pillar 3 supplementary regulatory capital disclosure workbooks from the Big 6 Canadian banks (RBC, TD, BMO, CIBC, BNS, NBC).',
    'XLSX workbooks containing Basel III/IV Pillar 3 quantitative tables covering capital composition, risk-weighted assets, credit risk, market risk, operational risk, leverage ratio, liquidity coverage, and net stable funding ratio for each of the Big 6 Canadian banks.',
    true,
    12,   -- batch_size
    6,    -- max_selected_files
    1,    -- top_chunks_in_catalog_selection
    3,    -- top_chunks_in_metadata_research
    5,    -- max_parallel_files
    20,   -- max_chunks_per_file
    10,   -- max_pages_for_full_context
    8,    -- max_primary_section_page_count
    4,    -- max_subsection_page_count
    2,    -- max_neighbour_chunks
    2,    -- max_gap_fill_pages
    true, -- enable_db_wide_deep_research
    true, -- enable_dense_table_retrieval (XLSX with dense regulatory tables)
    'Reporting Period',
    'Year and quarter subfolder (e.g., 2025_Q4, 2026_Q1). Set to match the quarter referenced in the user''s query. Leave unset to search across all available periods.',
    'Bank',
    'Big 6 Canadian bank code: RBC, TD, BMO, CIBC, BNS (Scotiabank), NBC (National Bank). Set to match the bank in the user''s query. Leave unset for cross-bank comparisons or when the user asks about multiple banks.',
    '',
    '',
    ARRAY['document_summary', 'document_description', 'document_usage'],
    '[
        "What is RBC''s Pillar 3 CET1 ratio for Q1 2026?",
        "Compare total capital ratios across the Big 6 banks",
        "What are TD''s risk-weighted assets by category?",
        "How does BMO''s leverage ratio compare to CIBC''s?",
        "What is the liquidity coverage ratio for Scotiabank?"
    ]'::jsonb
) ON CONFLICT (data_source) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    source_summary = EXCLUDED.source_summary,
    source_description = EXCLUDED.source_description,
    filter_1_label = EXCLUDED.filter_1_label,
    filter_1_description = EXCLUDED.filter_1_description,
    filter_2_label = EXCLUDED.filter_2_label,
    filter_2_description = EXCLUDED.filter_2_description,
    max_pages_for_full_context = EXCLUDED.max_pages_for_full_context,
    max_selected_files = EXCLUDED.max_selected_files,
    batch_size = EXCLUDED.batch_size,
    enable_dense_table_retrieval = EXCLUDED.enable_dense_table_retrieval,
    sample_questions = EXCLUDED.sample_questions,
    updated_at = CURRENT_TIMESTAMP;
