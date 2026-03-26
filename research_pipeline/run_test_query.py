#!/usr/bin/env python3
"""Run test queries against the /research endpoint.

Usage:
    python run_test_query.py --list                 # list available queries
    python run_test_query.py rbc_cet1_q1            # run a specific query
    python run_test_query.py rbc_cet1_q1 td_capital_trend  # run multiple
    python run_test_query.py --all                  # run all queries

Requires the research API to be running on localhost:8001.
"""

import argparse
import json
import sys
import time

import requests
import yaml

API_URL = "http://localhost:8001/research"
QUERIES_FILE = "test_queries.yaml"


def load_queries(path: str = QUERIES_FILE) -> dict:
    """Load test queries from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return {q["name"]: q for q in data.get("queries", [])}


def list_queries(queries: dict) -> None:
    """Print available query names with combo counts."""
    print("Available test queries:\n")
    for name, q in queries.items():
        combos = q.get("combos", [])
        sources = set(c["data_source"] for c in combos)
        banks = set(c.get("bank", "?") for c in combos)
        periods = set(c.get("period", "?") for c in combos)
        print(
            f"  {name:30s}  "
            f"{len(combos)} combo(s)  "
            f"sources={','.join(sources)}  "
            f"banks={','.join(banks)}  "
            f"periods={','.join(periods)}"
        )
    print(f"\n{len(queries)} queries total")


def run_query(name: str, query: dict) -> None:
    """Execute a single test query and print results."""
    print(f"\n{'=' * 70}")
    print(f"QUERY: {name}")
    print(f"{'=' * 70}")
    print(f"\nStatement: {query['research_statement'].strip()[:200]}...")
    print(f"Combos: {len(query['combos'])}")
    for c in query["combos"]:
        print(f"  - {c['data_source']}/{c.get('period', '?')}/{c.get('bank', '?')}")
    print()

    payload = {
        "research_statement": query["research_statement"].strip(),
        "combos": query["combos"],
    }

    start = time.time()
    try:
        response = requests.post(
            API_URL,
            json=payload,
            stream=True,
            timeout=300,
        )
        response.raise_for_status()

        full_response = []
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                print(chunk, end="", flush=True)
                full_response.append(chunk)

        elapsed = time.time() - start
        print(f"\n\n--- Completed in {elapsed:.1f}s ---\n")

    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to API. Is the server running?")
        print(f"  Expected at: {API_URL}")
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Run research test queries")
    parser.add_argument(
        "queries",
        nargs="*",
        help="Query names to run (from test_queries.yaml)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available queries"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all queries"
    )
    parser.add_argument(
        "--file",
        default=QUERIES_FILE,
        help=f"Path to queries YAML (default: {QUERIES_FILE})",
    )
    args = parser.parse_args()

    all_queries = load_queries(args.file)

    if args.list:
        list_queries(all_queries)
        return

    if args.all:
        names = list(all_queries.keys())
    elif args.queries:
        names = args.queries
    else:
        list_queries(all_queries)
        print("\nSpecify a query name, --all, or --list")
        return

    for name in names:
        if name not in all_queries:
            print(f"Unknown query: {name}")
            print(f"Available: {', '.join(all_queries.keys())}")
            sys.exit(1)
        run_query(name, all_queries[name])


if __name__ == "__main__":
    main()
