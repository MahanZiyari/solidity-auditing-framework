"""Phase 1 reference implementation: run the Slither and ANTLR extractors together.

This is the single entry point Phase 2 agents talk to. Given a contract file or a
dataset directory it runs both extractors, writes one document per extractor and
returns a manifest that pairs them:

    <output>/slither/<name>_metadata.json   from slither_metadata_extractor
    <output>/ast/<name>_ast.json            from antlr_ast_extractor
    <output>/manifest.json                  one row per contract, both statuses

The two extractors are independent on purpose: Slither needs a working solc and
can crash on some sources, while ANTLR always parses. A failure in one never
stops the other, and the manifest records both outcomes.

Command line::

    python phase1_runner.py contracts/SimpleVault.sol -o phase1_output/example
    python phase1_runner.py datasets/cve -o phase1_output/cve --validate
    python phase1_runner.py datasets/cve -o phase1_output/cve --include-tree --compact

From another Python file::

    from phase1_runner import run_phase1

    manifest = run_phase1("datasets/cve", output_dir="phase1_output/cve")
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from antlr_ast_extractor import extract_ast, validate_ast_document  # noqa: E402
from slither_metadata_extractor import (  # noqa: E402
    build_planner_prompt,
    extract_metadata,
    validate_metadata,
)

METADATA_SUFFIX = "_metadata.json"
AST_SUFFIX = "_ast.json"

MANIFEST_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["tool", "target", "options", "counts", "contracts"],
    "properties": {
        "tool": {"const": "phase1_runner"},
        "target": {"type": "string"},
        "options": {"type": "object"},
        "counts": {
            "type": "object",
            "required": [
                "contracts",
                "both_success",
                "slither_failed",
                "antlr_failed",
                "antlr_partial",
            ],
            "properties": {
                "contracts": {"type": "integer"},
                "both_success": {"type": "integer"},
                "slither_failed": {"type": "integer"},
                "antlr_failed": {"type": "integer"},
                "antlr_partial": {"type": "integer"},
            },
        },
        "contracts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["contract_file", "relative_path", "slither", "antlr"],
                "properties": {
                    "contract_file": {"type": "string"},
                    "relative_path": {"type": "string"},
                    "slither": {"type": ["object", "null"]},
                    "antlr": {"type": ["object", "null"]},
                },
            },
        },
    },
}


def _collect_targets(path: Path) -> List[Path]:
    return sorted(path.rglob("*.sol")) if path.is_dir() else [path]


def _ensure_venv_on_path() -> None:
    """Put the running interpreter's scripts directory on PATH.

    ``solc-select`` installs its ``solc`` shim next to the interpreter, so this is
    what activating a virtual environment already does. Without it the Slither half
    fails when the CLI is started as ``.venv/Scripts/python phase1_runner.py``.
    """
    if shutil.which("solc-select"):
        return
    scripts = Path(sys.executable).parent
    if any((scripts / name).exists() for name in ("solc-select", "solc-select.exe")):
        os.environ["PATH"] = str(scripts) + os.pathsep + os.environ.get("PATH", "")


def _output_name(root: Path, contract_file: Path, suffix: str) -> str:
    relative = contract_file.relative_to(root).with_suffix("")
    return f"{'__'.join(relative.parts)}{suffix}"


def _write_json(data: Dict[str, Any], output_path: Path, compact: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        if compact:
            json.dump(data, handle, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(data, handle, ensure_ascii=False, indent=2)


def run_slither(contract_file: Path, output_path: Path, options: Dict[str, Any]) -> Dict[str, Any]:
    """Run the Slither extractor and return a manifest row for it."""
    metadata = extract_metadata(
        contract_file,
        output_path,
        solc_version=options.get("solc_version"),
        solc_path=options.get("solc_path"),
        include_informational=bool(options.get("include_informational")),
    )
    summary = metadata.get("summary") or {}
    structural = metadata.get("structural_metadata") or {}
    return {
        "status": metadata.get("status"),
        "solc_version": metadata.get("solc_version"),
        "contract_name": structural.get("contract_name") if isinstance(structural, dict) else None,
        "total_findings": summary.get("total_findings", 0),
        "high_impact": summary.get("high_impact", 0),
        "medium_impact": summary.get("medium_impact", 0),
        "low_informational_impact": summary.get("low_informational_impact", 0),
        "output": output_path.as_posix(),
        "error": metadata.get("error"),
    }


def run_antlr(contract_file: Path, output_path: Path, options: Dict[str, Any]) -> Dict[str, Any]:
    """Run the ANTLR extractor and return a manifest row for it."""
    document = extract_ast(
        contract_file,
        output_path,
        include_tree=bool(options.get("include_tree")),
        include_tokens=bool(options.get("include_tokens")),
        compact=bool(options.get("compact")),
    )
    summary = document.get("summary") or {}
    return {
        "status": document.get("status"),
        "contracts": summary.get("contracts", 0),
        "functions": summary.get("functions", 0),
        "state_variables": summary.get("state_variables", 0),
        "syntax_errors": summary.get("syntax_errors", 0),
        "ast_nodes": summary.get("ast_nodes"),
        "include_tree": bool(options.get("include_tree")),
        "output": output_path.as_posix(),
        "error": document.get("error"),
    }


def run_phase1(
    target: str | Path,
    output_dir: Optional[str | Path] = None,
    *,
    solc_version: Optional[str] = None,
    solc_path: Optional[str] = None,
    include_informational: bool = False,
    include_tree: bool = False,
    include_tokens: bool = False,
    compact: bool = False,
    with_slither: bool = True,
    with_antlr: bool = True,
    on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run both Phase 1 extractors over one contract or a whole dataset directory.

    Returns the manifest dict (also written to ``<output_dir>/manifest.json`` when
    ``output_dir`` is given). ``on_result(record)`` is called after each contract and
    can collect progress or validation messages. ``with_slither`` / ``with_antlr``
    disable one extractor, in which case its manifest row is ``null``.
    """
    target_path = Path(target)
    if not target_path.exists():
        raise FileNotFoundError(f"Target not found: {target_path}")

    root = target_path if target_path.is_dir() else target_path.parent
    options = {
        "solc_version": solc_version,
        "solc_path": solc_path,
        "include_informational": include_informational,
        "include_tree": include_tree,
        "include_tokens": include_tokens,
        "compact": compact,
    }

    out_root = Path(output_dir) if output_dir is not None else None
    records: List[Dict[str, Any]] = []

    for contract_file in _collect_targets(target_path):
        record: Dict[str, Any] = {
            "contract_file": contract_file.name,
            "relative_path": contract_file.relative_to(root).as_posix(),
            "slither": None,
            "antlr": None,
        }
        if with_slither:
            name = _output_name(root, contract_file, METADATA_SUFFIX)
            record["slither"] = run_slither(
                contract_file, out_root / "slither" / name if out_root else Path(name), options
            )
        if with_antlr:
            name = _output_name(root, contract_file, AST_SUFFIX)
            record["antlr"] = run_antlr(
                contract_file, out_root / "ast" / name if out_root else Path(name), options
            )
        if on_result is not None:
            on_result(record)
        records.append(record)

    counts = {
        "contracts": len(records),
        "both_success": sum(
            1
            for record in records
            if all(
                row is None or row["status"] == "success"
                for row in (record["slither"], record["antlr"])
            )
        ),
        "slither_failed": sum(
            1 for record in records if record["slither"] and record["slither"]["status"] == "failed"
        ),
        "antlr_failed": sum(
            1 for record in records if record["antlr"] and record["antlr"]["status"] == "failed"
        ),
        "antlr_partial": sum(
            1 for record in records if record["antlr"] and record["antlr"]["status"] == "partial"
        ),
    }
    manifest = {
        "tool": "phase1_runner",
        "target": target_path.as_posix(),
        "options": options,
        "counts": counts,
        "contracts": records,
    }
    if out_root is not None:
        _write_json(manifest, out_root / "manifest.json", compact=compact)
    return manifest


def validate_manifest(manifest: Dict[str, Any]) -> List[str]:
    """Validate against MANIFEST_SCHEMA; returns error messages (empty when valid)."""
    try:
        import jsonschema
    except ImportError:
        return ["jsonschema is not installed; run `pip install jsonschema`"]
    validator = jsonschema.Draft202012Validator(MANIFEST_SCHEMA)
    return [
        f"{'/'.join(map(str, error.path)) or '<root>'}: {error.message}"
        for error in validator.iter_errors(manifest)
    ]


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: run the Slither and ANTLR extractors over one contract or a dataset."
    )
    parser.add_argument("target", help="Solidity file, or a directory to process every .sol file")
    parser.add_argument("-o", "--output", help="Output directory (default: phase1_output)")
    parser.add_argument("--solc-version", help="Force this solc version instead of resolving the pragma")
    parser.add_argument("--solc", dest="solc_path", help="Path to a solc binary to use")
    parser.add_argument("--include-informational", action="store_true", help="Keep informational Slither findings")
    parser.add_argument("--include-tree", action="store_true", help="Also keep the full ANTLR syntax tree")
    parser.add_argument("--include-tokens", action="store_true", help="Keep every token as a tree node (needs --include-tree)")
    parser.add_argument("--compact", action="store_true", help="Write JSON without pretty-printing whitespace")
    parser.add_argument("--skip-slither", action="store_true", help="Run only the ANTLR extractor")
    parser.add_argument("--skip-antlr", action="store_true", help="Run only the Slither extractor")
    parser.add_argument("--prompt", action="store_true", help="Print the Phase 2 planner prompt (single file only)")
    parser.add_argument("--validate", action="store_true", help="Validate every written document against its schema")
    parser.add_argument("--print-schema", action="store_true", help="Print the manifest JSON schema and exit")
    return parser.parse_args(argv)


def _status(pair: str, row: Optional[Dict[str, Any]]) -> str:
    if row is None:
        return f"{pair}=skipped"
    detail = row.get("solc_version") if pair == "slither" else row.get("syntax_errors")
    label = row["status"]
    return f"{pair}={label}" + (f"({detail})" if detail else "")


def main(argv: Optional[List[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parse_args(argv)
    if args.print_schema:
        print(json.dumps(MANIFEST_SCHEMA, indent=2))
        return 0

    _ensure_venv_on_path()
    if not args.skip_slither and not shutil.which("solc-select"):
        print(
            "solc-select is not on PATH; the Slither extractor will fail. "
            "Activate the virtual environment or install solc-select.",
            file=sys.stderr,
        )

    target = Path(args.target)
    if not target.exists():
        print(f"Target not found: {target}", file=sys.stderr)
        return 2
    if args.include_tokens and not args.include_tree:
        print("--include-tokens requires --include-tree", file=sys.stderr)
        return 2
    if args.prompt and target.is_dir():
        print("--prompt only works for a single contract file", file=sys.stderr)
        return 2

    output_dir = Path(args.output or "phase1_output")
    schema_errors: List[str] = []
    prompts: List[str] = []

    def report(record: Dict[str, Any]) -> None:
        slither_row, antlr_row = record["slither"], record["antlr"]
        print(f"{record['relative_path']}: {_status('slither', slither_row)}, {_status('antlr', antlr_row)}")
        for name, row in (("slither", slither_row), ("antlr", antlr_row)):
            if row and row.get("error"):
                print(f"    {name} error: {row['error']}", file=sys.stderr)
        if args.validate:
            if slither_row:
                document = _load(Path(slither_row["output"]))
                for error in validate_metadata(document)[:5]:
                    schema_errors.append(f"{record['relative_path']} slither: {error}")
            if antlr_row:
                document = _load(Path(antlr_row["output"]))
                for error in validate_ast_document(document)[:5]:
                    schema_errors.append(f"{record['relative_path']} antlr: {error}")
        if args.prompt and slither_row:
            metadata = _load(Path(slither_row["output"]))
            ast = _load(Path(antlr_row["output"])) if antlr_row else None
            prompts.append(build_planner_prompt(metadata, ast))

    manifest = run_phase1(
        target,
        output_dir,
        solc_version=args.solc_version,
        solc_path=args.solc_path,
        include_informational=args.include_informational,
        include_tree=args.include_tree,
        include_tokens=args.include_tokens,
        compact=args.compact,
        with_slither=not args.skip_slither,
        with_antlr=not args.skip_antlr,
        on_result=report,
    )

    for error in validate_manifest(manifest):
        schema_errors.append(f"manifest: {error}")

    counts = manifest["counts"]
    print(
        f"\nPhase 1 on {counts['contracts']} contract(s): "
        f"{counts['both_success']} clean, "
        f"{counts['slither_failed']} Slither failure(s), "
        f"{counts['antlr_failed']} ANTLR failure(s), "
        f"{counts['antlr_partial']} partial parse(s)"
    )
    print(f"Manifest: {(output_dir / 'manifest.json').as_posix()}")
    if schema_errors:
        for error in schema_errors[:10]:
            print(f"    schema: {error}", file=sys.stderr)
        print(f"    {len(schema_errors)} schema error(s) in total", file=sys.stderr)
    if prompts:
        print("\n=== SYSTEM ===\n" + prompts[0]["system"])
        print("\n=== USER ===\n" + prompts[0]["user"])

    failures = counts["slither_failed"] + counts["antlr_failed"]
    return 1 if schema_errors or failures else 0


def _load(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    sys.exit(main())
