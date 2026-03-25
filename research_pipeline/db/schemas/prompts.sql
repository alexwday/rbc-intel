-- Agent prompts with versioning.
-- Loaded into memory at startup by prompt_loader.py.

CREATE TABLE IF NOT EXISTS prompts (
    id              SERIAL PRIMARY KEY,
    model           VARCHAR(50)  NOT NULL DEFAULT 'research',
    layer           VARCHAR(50)  NOT NULL,
    name            VARCHAR(100) NOT NULL,
    version         VARCHAR(20)  NOT NULL DEFAULT '1.0.0',
    description     TEXT,
    system_prompt   TEXT,
    user_prompt     TEXT,
    tool_definition JSONB,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_prompts_model_layer_name_version
        UNIQUE (model, layer, name, version)
);
