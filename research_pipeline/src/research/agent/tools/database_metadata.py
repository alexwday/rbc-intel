"""Cached access to data source metadata from research_registry."""

import logging
import time
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from ...connections.postgres import get_database_session, get_database_schema

logger = logging.getLogger(__name__)


class DataSourceNotFoundError(Exception):
    """Exception raised when a data source is not found in the registry."""


class DatabaseMetadataCache:
    """Repository for data source metadata with a configurable cache TTL."""

    def __init__(self, cache_ttl_seconds: int = 300):
        """Initialize the repository.

        Args:
            cache_ttl_seconds: Seconds to retain cached metadata (default 5 minutes).
        """
        self._cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._cache_timestamp: Optional[float] = None
        self._cache_ttl = cache_ttl_seconds

    def _is_cache_valid(self) -> bool:
        """Check whether cached metadata is still within the TTL."""
        if self._cache is None or self._cache_timestamp is None:
            return False
        return (time.time() - self._cache_timestamp) < self._cache_ttl

    def invalidate_cache(self) -> None:
        """Force cache refresh on next query."""
        self._cache = None
        self._cache_timestamp = None

    def _fetch_from_database(self) -> Dict[str, Dict[str, Any]]:
        """Fetch enabled data sources from research_registry.

        Returns:
            Mapping of data_source to configuration details.

        Raises:
            RuntimeError: If the database query fails.
        """
        schema = get_database_schema()
        try:
            with get_database_session() as session:
                rows = (
                    session.execute(
                        text(
                            f"""
                            SELECT
                                data_source,
                                display_name,
                                source_summary,
                                source_description,
                                batch_size,
                                max_selected_files,
                                top_chunks_in_catalog_selection,
                                top_chunks_in_metadata_research,
                                max_pages_for_full_context,
                                enable_db_wide_deep_research,
                                enable_dense_table_retrieval,
                                max_parallel_files,
                                max_chunks_per_file,
                                max_primary_section_page_count,
                                max_subsection_page_count,
                                max_neighbour_chunks,
                                max_gap_fill_pages,
                                filter_1_label,
                                filter_1_description,
                                filter_2_label,
                                filter_2_description,
                                filter_3_label,
                                filter_3_description,
                                metadata_context_fields,
                                sample_questions,
                                enabled
                            FROM {schema}.research_registry
                            WHERE enabled = true
                            ORDER BY data_source
                        """
                        )
                    )
                    .mappings()
                    .all()
                )

            data_sources = {}
            for row in rows:
                data_sources[row["data_source"]] = {
                    "display_name": row["display_name"],
                    "description": row["source_summary"],
                    "source_description": row["source_description"],
                    "batch_size": row["batch_size"],
                    "max_selected_files": row["max_selected_files"],
                    "top_chunks_in_catalog_selection": row[
                        "top_chunks_in_catalog_selection"
                    ],
                    "top_chunks_in_metadata_research": row[
                        "top_chunks_in_metadata_research"
                    ],
                    "max_pages_for_full_context": row[
                        "max_pages_for_full_context"
                    ],
                    "enable_db_wide_deep_research": row[
                        "enable_db_wide_deep_research"
                    ],
                    "enable_dense_table_retrieval": row[
                        "enable_dense_table_retrieval"
                    ],
                    "max_parallel_files": row["max_parallel_files"],
                    "max_chunks_per_file": row["max_chunks_per_file"],
                    "max_primary_section_page_count": row[
                        "max_primary_section_page_count"
                    ],
                    "max_subsection_page_count": row[
                        "max_subsection_page_count"
                    ],
                    "max_neighbour_chunks": row["max_neighbour_chunks"],
                    "max_gap_fill_pages": row["max_gap_fill_pages"],
                    "filter_1_label": row["filter_1_label"] or "",
                    "filter_1_description": row["filter_1_description"] or "",
                    "filter_2_label": row["filter_2_label"] or "",
                    "filter_2_description": row["filter_2_description"] or "",
                    "filter_3_label": row["filter_3_label"] or "",
                    "filter_3_description": row["filter_3_description"] or "",
                    "metadata_context_fields": row["metadata_context_fields"]
                    or ["document_summary"],
                    "sample_questions": row["sample_questions"] or [],
                    "enabled": row["enabled"],
                }

            logger.info(
                "Loaded %d data sources from research_registry",
                len(data_sources),
            )
            return data_sources
        except Exception as exc:
            logger.error(
                "Failed to fetch data source metadata: %s",
                exc,
                exc_info=True,
            )
            raise RuntimeError(
                f"Data source metadata fetch failed: {exc}"
            ) from exc

    def get_all_data_sources(
        self, use_cache: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Return enabled data sources with full configuration.

        Args:
            use_cache: Whether to return cached data when valid.

        Returns:
            Dict mapping data_source to configuration.
        """
        if use_cache and self._is_cache_valid():
            return self._cache

        self._cache = self._fetch_from_database()
        self._cache_timestamp = time.time()
        return self._cache

    def get_data_source_config(
        self, data_source: str
    ) -> Optional[Dict[str, Any]]:
        """Return configuration for a specific data source.

        Args:
            data_source: Data source identifier.

        Returns:
            Configuration dict, or None if not found.
        """
        data_sources = self.get_all_data_sources()
        return data_sources.get(data_source)

    def get_research_config(self, data_source: str) -> Dict[str, Any]:
        """Return research configuration for a specific data source.

        Args:
            data_source: Data source identifier.

        Returns:
            Research configuration dict.

        Raises:
            DataSourceNotFoundError: If data source is not found.
        """
        ds_config = self.get_data_source_config(data_source)
        if ds_config is None:
            raise DataSourceNotFoundError(
                f"Data source '{data_source}' not found in research_registry"
            )
        return {
            "batch_size": ds_config["batch_size"],
            "max_selected_files": ds_config["max_selected_files"],
            "top_chunks_in_catalog_selection": ds_config[
                "top_chunks_in_catalog_selection"
            ],
            "top_chunks_in_metadata_research": ds_config[
                "top_chunks_in_metadata_research"
            ],
            "max_pages_for_full_context": ds_config[
                "max_pages_for_full_context"
            ],
            "enable_db_wide_deep_research": ds_config[
                "enable_db_wide_deep_research"
            ],
            "enable_dense_table_retrieval": ds_config[
                "enable_dense_table_retrieval"
            ],
            "max_parallel_files": ds_config["max_parallel_files"],
            "max_chunks_per_file": ds_config["max_chunks_per_file"],
            "max_primary_section_page_count": ds_config[
                "max_primary_section_page_count"
            ],
            "max_subsection_page_count": ds_config[
                "max_subsection_page_count"
            ],
            "max_neighbour_chunks": ds_config["max_neighbour_chunks"],
            "max_gap_fill_pages": ds_config["max_gap_fill_pages"],
            "metadata_context_fields": ds_config["metadata_context_fields"],
        }

    def is_data_source_enabled(self, data_source: str) -> bool:
        """Check whether a data source exists and is enabled.

        Args:
            data_source: Data source identifier.

        Returns:
            True if the data source exists and is enabled.
        """
        ds_config = self.get_data_source_config(data_source)
        return ds_config is not None and ds_config.get("enabled", False)


_REPOSITORY_CACHE: Dict[str, DatabaseMetadataCache] = {}


def get_metadata_repository() -> DatabaseMetadataCache:
    """Return the singleton repository instance."""
    if "instance" not in _REPOSITORY_CACHE:
        _REPOSITORY_CACHE["instance"] = DatabaseMetadataCache()
    return _REPOSITORY_CACHE["instance"]


def get_filter_metadata_for_sources(
    data_source_names: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Return filter metadata for data sources that have filters configured.

    Args:
        data_source_names: Data source identifiers to check.

    Returns:
        Dict mapping data_source to its filter metadata (label, description)
        for each filter level that has a non-empty label. Sources with no
        filters configured are omitted.
    """
    repo = get_metadata_repository()
    result: Dict[str, Dict[str, Any]] = {}

    for ds_name in data_source_names:
        ds_config = repo.get_data_source_config(ds_name)
        if not ds_config:
            continue

        filters = {}
        for level in (1, 2, 3):
            label = ds_config.get(f"filter_{level}_label", "")
            desc = ds_config.get(f"filter_{level}_description", "")
            if label:
                filters[f"filter_{level}"] = {
                    "label": label,
                    "description": desc,
                }

        if filters:
            result[ds_name] = {
                "display_name": ds_config.get("display_name", ds_name),
                "filters": filters,
            }

    return result


def fetch_filter_values_for_source(
    data_source: str,
) -> Dict[str, List[str]]:
    """Query distinct filter values for a data source from the documents table.

    Args:
        data_source: Data source identifier.

    Returns:
        Dict mapping filter key to sorted list of distinct non-empty values.
    """
    schema = get_database_schema()
    values: Dict[str, List[str]] = {}

    try:
        with get_database_session() as session:
            for level in (1, 2, 3):
                col = f"filter_{level}"
                rows = (
                    session.execute(
                        text(
                            f"SELECT DISTINCT {col} FROM {schema}.documents "
                            f"WHERE data_source = :ds AND {col} != '' "
                            f"ORDER BY {col}"
                        ),
                        {"ds": data_source},
                    )
                    .scalars()
                    .all()
                )
                if rows:
                    values[col] = list(rows)
    except Exception as exc:
        logger.error(
            "Error fetching filter values for %s: %s",
            data_source,
            exc,
        )

    return values


def fetch_available_data_sources() -> Dict[str, Dict[str, Any]]:
    """Return available data sources for API consumers.

    Returns:
        Dict mapping data_source to configuration with sample questions.
    """
    enriched: Dict[str, Dict[str, Any]] = {}
    for ds_name, ds_info in (
        get_metadata_repository().get_all_data_sources().items()
    ):
        enriched[ds_name] = {
            **ds_info,
            "questions": ds_info.get("sample_questions") or [],
        }

    return enriched
