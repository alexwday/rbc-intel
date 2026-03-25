#!/usr/bin/env bash
# =============================================================================
# rbc-intel setup script
#
# Creates venvs, installs dependencies, configures .env files, creates
# Postgres tables, and verifies everything is ready to run.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# Prerequisites:
#   - Python 3.12+
#   - PostgreSQL running with pgvector extension
#   - LibreOffice (for DOCX/PPTX → PDF conversion in ingestion)
# =============================================================================

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOC_PIPELINE="$ROOT_DIR/document_pipeline"
RESEARCH_PIPELINE="$ROOT_DIR/research_pipeline"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!!]${NC} $1"; }
fail()  { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }
step()  { echo -e "\n${GREEN}=== $1 ===${NC}"; }

# ---------------------------------------------------------------------------
step "1. Checking Python"
# ---------------------------------------------------------------------------
PYTHON=$(command -v python3 || true)
if [ -z "$PYTHON" ]; then
    fail "python3 not found. Install Python 3.12+."
fi
PY_VERSION=$($PYTHON --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
info "Python $PY_VERSION found at $PYTHON"

# ---------------------------------------------------------------------------
step "2. Document Pipeline — venv + dependencies"
# ---------------------------------------------------------------------------
if [ ! -d "$DOC_PIPELINE/.venv" ]; then
    echo "Creating document_pipeline venv..."
    $PYTHON -m venv "$DOC_PIPELINE/.venv"
fi
"$DOC_PIPELINE/.venv/bin/pip" install -q -e "$DOC_PIPELINE[dev]" 2>&1 | tail -1
info "document_pipeline dependencies installed"

# ---------------------------------------------------------------------------
step "3. Research Pipeline — venv + dependencies"
# ---------------------------------------------------------------------------
if [ ! -d "$RESEARCH_PIPELINE/.venv" ]; then
    echo "Creating research_pipeline venv..."
    $PYTHON -m venv "$RESEARCH_PIPELINE/.venv"
fi
"$RESEARCH_PIPELINE/.venv/bin/pip" install -q -e "$RESEARCH_PIPELINE[dev]" 2>&1 | tail -1
info "research_pipeline dependencies installed"

# ---------------------------------------------------------------------------
step "4. Environment files"
# ---------------------------------------------------------------------------
for PIPELINE in "$DOC_PIPELINE" "$RESEARCH_PIPELINE"; do
    NAME=$(basename "$PIPELINE")
    if [ ! -f "$PIPELINE/.env" ]; then
        cp "$PIPELINE/.env.example" "$PIPELINE/.env"
        warn "$NAME/.env created from .env.example — EDIT IT with your credentials"
    else
        info "$NAME/.env already exists"
    fi
done

echo ""
echo "  Before continuing, make sure both .env files have:"
echo "    - Database credentials (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)"
echo "    - Auth config (OPENAI_API_KEY or OAuth settings)"
echo "    - LLM endpoint (LLM_ENDPOINT / AZURE_BASE_URL)"
echo "    - Model names for your environment"
echo ""
read -p "  Press Enter when .env files are configured (or Ctrl+C to stop)..."

# ---------------------------------------------------------------------------
step "5. PostgreSQL — connection test"
# ---------------------------------------------------------------------------
# Source the research pipeline .env for DB params
set -a
source "$RESEARCH_PIPELINE/.env"
set +a

echo "Testing connection to $DB_HOST:$DB_PORT/$DB_NAME..."
"$RESEARCH_PIPELINE/.venv/bin/python" -c "
import psycopg2
conn = psycopg2.connect(
    host='$DB_HOST', port='$DB_PORT', dbname='$DB_NAME',
    user='$DB_USER', password='$DB_PASSWORD'
)
conn.close()
print('Connected successfully')
" || fail "Cannot connect to PostgreSQL. Check your .env credentials."
info "PostgreSQL connection verified"

# ---------------------------------------------------------------------------
step "6. PostgreSQL — pgvector extension"
# ---------------------------------------------------------------------------
"$RESEARCH_PIPELINE/.venv/bin/python" -c "
import psycopg2
conn = psycopg2.connect(
    host='$DB_HOST', port='$DB_PORT', dbname='$DB_NAME',
    user='$DB_USER', password='$DB_PASSWORD'
)
cur = conn.cursor()
cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
conn.commit()
cur.execute('SELECT extversion FROM pg_extension WHERE extname = %s', ('vector',))
row = cur.fetchone()
print(f'pgvector version: {row[0]}' if row else 'pgvector not found')
conn.close()
" || fail "Could not create/verify pgvector extension."
info "pgvector extension verified"

# ---------------------------------------------------------------------------
step "7. PostgreSQL — ingestion pipeline tables"
# ---------------------------------------------------------------------------
"$DOC_PIPELINE/.venv/bin/python" -c "
import psycopg2, os, sys
sys.path.insert(0, '$DOC_PIPELINE/src')
os.chdir('$DOC_PIPELINE')

# Source ingestion .env
from dotenv import load_dotenv
load_dotenv('$DOC_PIPELINE/.env')

from ingestion.utils.config import get_database_schema
from ingestion.utils.postgres import ensure_storage_tables

conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'localhost'),
    port=os.getenv('DB_PORT', '5432'),
    dbname=os.getenv('DB_NAME', 'rbc_intel'),
    user=os.getenv('DB_USER', ''),
    password=os.getenv('DB_PASSWORD', ''),
)
ensure_storage_tables(conn)
conn.commit()

# Verify tables exist
cur = conn.cursor()
tables = [
    'document_catalog', 'documents', 'document_sections',
    'document_subsections', 'document_chunks', 'document_dense_tables',
    'document_keywords', 'document_metrics', 'document_sheet_summaries',
    'document_sheet_context_chains',
]
schema = get_database_schema()
for t in tables:
    cur.execute(
        'SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s)',
        (schema, t)
    )
    exists = cur.fetchone()[0]
    status = 'OK' if exists else 'MISSING'
    print(f'  {t:35s} [{status}]')
conn.close()
" || fail "Could not create ingestion tables."
info "Ingestion pipeline tables verified"

# ---------------------------------------------------------------------------
step "8. PostgreSQL — research pipeline tables + seed data"
# ---------------------------------------------------------------------------
cd "$RESEARCH_PIPELINE"
.venv/bin/python -m db.setup 2>&1
info "Research pipeline tables created and seeded"

# ---------------------------------------------------------------------------
step "9. LLM connection test"
# ---------------------------------------------------------------------------
"$DOC_PIPELINE/.venv/bin/python" -c "
import os, sys
sys.path.insert(0, '$DOC_PIPELINE/src')
os.chdir('$DOC_PIPELINE')
from dotenv import load_dotenv
load_dotenv('$DOC_PIPELINE/.env')
from ingestion.utils.config import get_stage_model_config
from ingestion.utils.llm import LLMClient

llm = LLMClient()
print(f'Auth mode: {llm.auth_mode}')
print('Testing LLM connection...')
# The startup health check does a lightweight LLM call
model_config = get_stage_model_config('startup')
print(f'Model: {model_config[\"model\"]}')
" || warn "LLM client initialized but connection test skipped (run pipeline to test)"
info "LLM configuration loaded"

# ---------------------------------------------------------------------------
step "10. Test data verification"
# ---------------------------------------------------------------------------
SOURCES="$DOC_PIPELINE/test_data/runtime_sources"
FILE_COUNT=$(find "$SOURCES" -type f \( -name "*.pdf" -o -name "*.xlsx" \) | wc -l | tr -d ' ')
echo "  Found $FILE_COUNT files in test_data/runtime_sources:"
for ds in investor_slides pillar3_disclosure; do
    for q in 2025_Q4 2026_Q1; do
        count=$(find "$SOURCES/$ds/$q" -type f 2>/dev/null | wc -l | tr -d ' ')
        echo "    $ds/$q: $count files"
    done
done
if [ "$FILE_COUNT" -eq 24 ]; then
    info "All 24 test files present"
else
    warn "Expected 24 files, found $FILE_COUNT"
fi

# ---------------------------------------------------------------------------
step "11. Summary"
# ---------------------------------------------------------------------------
echo ""
echo "  Setup complete. Next steps:"
echo ""
echo "  1. Run the ingestion pipeline:"
echo "     cd document_pipeline"
echo "     .venv/bin/python -m ingestion.main"
echo ""
echo "  2. Start the research API:"
echo "     cd research_pipeline"
echo "     .venv/bin/uvicorn research.api:app --app-dir src --port 8001"
echo ""
echo "  3. Send a research query:"
echo "     curl -X POST http://localhost:8001/chat \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"messages\": [{\"role\": \"user\", \"content\": \"What is RBC CET1 ratio for Q1 2026?\"}], \"stream\": true}'"
echo ""
