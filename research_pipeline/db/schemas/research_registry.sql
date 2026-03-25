-- Per-source research configuration.
-- One row per data_source value in the documents table.

CREATE TABLE IF NOT EXISTS research_registry (
    data_source                      TEXT PRIMARY KEY,
    display_name                     TEXT NOT NULL,
    source_summary                   TEXT NOT NULL,
    source_description               TEXT NOT NULL,
    enabled                          BOOLEAN NOT NULL DEFAULT true,

    -- Retrieval config
    batch_size                       INTEGER NOT NULL DEFAULT 15,
    max_selected_files               INTEGER NOT NULL DEFAULT 10,
    top_chunks_in_catalog_selection  INTEGER NOT NULL DEFAULT 1,
    top_chunks_in_metadata_research  INTEGER NOT NULL DEFAULT 3,
    max_parallel_files               INTEGER NOT NULL DEFAULT 5,
    max_chunks_per_file              INTEGER NOT NULL DEFAULT 20,
    max_pages_for_full_context       INTEGER NOT NULL DEFAULT 6,
    max_primary_section_page_count   INTEGER NOT NULL DEFAULT 6,
    max_subsection_page_count        INTEGER NOT NULL DEFAULT 3,
    max_neighbour_chunks             INTEGER NOT NULL DEFAULT 2,
    max_gap_fill_pages               INTEGER NOT NULL DEFAULT 2,
    enable_db_wide_deep_research     BOOLEAN NOT NULL DEFAULT true,

    -- Dense table config
    enable_dense_table_retrieval     BOOLEAN NOT NULL DEFAULT true,

    -- Per-source subfolder filters (up to 3 levels)
    -- When set, the filter resolver LLM uses these to auto-resolve or
    -- ask the user for filter values after data source selection.
    filter_1_label                   TEXT NOT NULL DEFAULT '',
    filter_1_description             TEXT NOT NULL DEFAULT '',
    filter_2_label                   TEXT NOT NULL DEFAULT '',
    filter_2_description             TEXT NOT NULL DEFAULT '',
    filter_3_label                   TEXT NOT NULL DEFAULT '',
    filter_3_description             TEXT NOT NULL DEFAULT '',

    -- Context fields
    metadata_context_fields          TEXT[] NOT NULL DEFAULT ARRAY['document_summary'],
    sample_questions                 JSONB,

    created_at                       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at                       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
