"""Tests for the module layout."""

import importlib


def test_import_ingestion_package():
    """The project exposes an importable ingestion package."""
    package = importlib.import_module("ingestion")
    module = importlib.import_module("ingestion.main")
    assert package.__name__ == "ingestion"
    assert callable(module.main)


def test_import_subpackages():
    """Core subpackages are importable."""
    importlib.import_module("ingestion.connections")
    importlib.import_module("ingestion.utils")
    importlib.import_module("ingestion.stages")
