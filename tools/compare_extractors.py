"""Cross-check the Slither and ANTLR phase-1 outputs.

Compares, per contract, the multiset of function names and state-variable
names reported by ``slither_metadata_extractor.py`` and
``antlr_ast_extractor.py``.  Also reports output sizes for the different
AST serialisation options.

Usage:
    python tools/compare_extractors.py phase1_output
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def slither_functions(contract: dict, declared_only: bool = False) -> Counter:
    names = []
    for function in contract.get("functions") or []:
        if declared_only and function.get("declared_in") != contract.get("contract_name"):
            continue
        is_ctor = function.get("is_constructor") or function.get("name") == "constructor"
        names.append("constructor" if is_ctor else function.get("name"))
    return Counter(names)


def ast_functions(contract: dict) -> Counter:
    names = []
    for function in contract.get("functions") or []:
        is_ctor = (
            function.get("kind") == "constructor"
            or function.get("is_legacy_constructor")
        )
        if is_ctor:
            names.append("constructor")
        elif function.get("kind") in ("fallback", "receive"):
            names.append(function["kind"])
        else:
            names.append(function.get("name"))
    return Counter(names)


def slither_state_vars(contract: dict) -> Counter:
    return Counter(v.get("name") for v in contract.get("state_variables") or [])


def ast_state_vars(contract: dict) -> Counter:
    return Counter(v.get("name") for v in contract.get("state_variables") or [])


def ast_lineage(contract: dict, by_name: dict) -> list[dict]:
    """Contract followed by every ancestor declared in the same file."""
    ordered = []
    seen = set()

    def walk(node: dict) -> None:
        if node.get("name") in seen:
            return
        seen.add(node.get("name"))
        ordered.append(node)
        for parent in node.get("inheritance") or []:
            parent_name = parent.get("name") if isinstance(parent, dict) else parent
            if parent_name in by_name:
                walk(by_name[parent_name])

    walk(contract)
    return ordered


def pair_files(output_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for metadata in sorted(output_dir.glob("*/*_metadata.json")):
        dataset = metadata.parent.name
        stem = metadata.name[: -len("_metadata.json")]
        ast = output_dir / "ast" / dataset / f"{stem}_ast.json"
        if ast.exists():
            pairs.append((metadata, ast))
    return pairs


def compare(metadata_path: Path, ast_path: Path, report) -> dict:
    metadata = load(metadata_path)
    ast = load(ast_path)
    result = {
        "file": metadata_path.name[: -len("_metadata.json")],
        "skipped": None,
        "contracts_missing_in_ast": [],
        "contracts_missing_in_slither": [],
        "function_mismatches": [],
        "effective_function_mismatches": [],
        "state_var_mismatches": [],
    }

    structural = metadata.get("structural_metadata")
    if isinstance(structural, dict):
        structural = [structural]
    elif not isinstance(structural, list):
        structural = []
    if metadata.get("status") != "success" or not structural:
        result["skipped"] = metadata.get("status")
        return result
    slither_contracts = {c.get("contract_name"): c for c in structural}
    ast_contracts = {c.get("name"): c for c in ast.get("contracts") or []}

    result["contracts_missing_in_ast"] = sorted(set(slither_contracts) - set(ast_contracts))

    for name in sorted(set(slither_contracts) & set(ast_contracts)):
        slither_fns = slither_functions(slither_contracts[name], declared_only=True)
        ast_fns = ast_functions(ast_contracts[name])
        if slither_fns != ast_fns:
            result["function_mismatches"].append(
                {
                    "contract": name,
                    "only_slither": sorted((slither_fns - ast_fns).elements()),
                    "only_antlr": sorted((ast_fns - slither_fns).elements()),
                }
            )

        lineage = ast_lineage(ast_contracts[name], ast_contracts)
        effective_fns = Counter()
        effective_vars = Counter()
        for node in lineage:
            effective_fns.update(ast_functions(node))
            effective_vars.update(ast_state_vars(node))

        private_inherited = {
            var.get("name")
            for node in lineage[1:]
            for var in node.get("state_variables") or []
            if var.get("visibility") == "private"
        }

        def classify(names, counts, private=()):
            if not names:
                return "none"
            if all(counts[n] > 1 for n in names):
                return "shadowing"
            if all(counts[n] == 1 and n in private for n in names):
                return "private_inherited"
            return "unexplained"

        slither_all_fns = slither_functions(slither_contracts[name])
        if slither_all_fns != effective_fns:
            extra = sorted((effective_fns - slither_all_fns).elements())
            result["effective_function_mismatches"].append(
                {
                    "contract": name,
                    "only_slither": sorted((slither_all_fns - effective_fns).elements()),
                    "only_antlr": extra,
                    "cause": classify(extra, effective_fns),
                }
            )

        slither_vars = slither_state_vars(slither_contracts[name])
        if slither_vars != effective_vars:
            extra = sorted((effective_vars - slither_vars).elements())
            result["state_var_mismatches"].append(
                {
                    "contract": name,
                    "only_slither": sorted((slither_vars - effective_vars).elements()),
                    "only_antlr": extra,
                    "cause": classify(extra, effective_vars, private_inherited),
                }
            )
    return result


def size_report(output_dir: Path) -> None:
    ast_files = sorted(output_dir.glob("ast/*/*_ast.json"))
    if not ast_files:
        return
    sizes = sorted(p.stat().st_size for p in ast_files)
    with_tree = sum(1 for p in ast_files if load(p).get("ast") is not None)
    print(f"\nAST files: {len(ast_files)} ({with_tree} with a syntax tree)")
    print(f"  total : {sum(sizes) / 1e6:.2f} MB")
    print(f"  median: {sizes[len(sizes) // 2] / 1e3:.1f} KB")
    print(f"  max   : {sizes[-1] / 1e3:.1f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, nargs="?", default=Path("phase1_output"))
    parser.add_argument("--json", type=Path, help="write the full report to this file")
    args = parser.parse_args()

    pairs = pair_files(args.output_dir)
    print(f"Matched {len(pairs)} Slither/ANTLR pairs")

    reports = []
    clean = 0
    skipped = 0
    for metadata_path, ast_path in pairs:
        result = compare(metadata_path, ast_path, None)
        reports.append(result)
        if result["skipped"]:
            skipped += 1
            continue
        if not any(
            result[key]
            for key in (
                "contracts_missing_in_ast",
                "function_mismatches",
                "effective_function_mismatches",
                "state_var_mismatches",
            )
        ):
            clean += 1

    compared = len(pairs) - skipped
    print(f"Skipped (Slither failed): {skipped}")
    print(f"Fully identical (declared + inherited): {clean}/{compared}")

    for key, label in (
        ("effective_function_mismatches", "inherited functions"),
        ("state_var_mismatches", "state variables"),
    ):
        causes = Counter(entry["cause"] for r in reports for entry in r[key])
        print(f"\n{label} differences by cause: {dict(causes)}")
        for result in reports:
            bad = [e for e in result[key] if e["cause"] == "unexplained"]
            if bad:
                print(f"  UNEXPLAINED {result['file']}: {json.dumps(bad, ensure_ascii=False)}")

    for key, label in (
        ("contracts_missing_in_ast", "contracts only in Slither"),
        ("function_mismatches", "declared function mismatches"),
        ("effective_function_mismatches", "inherited function mismatches"),
        ("state_var_mismatches", "state variable mismatches"),
    ):
        offenders = [r for r in reports if r[key]]
        print(f"\n{label}: {len(offenders)} file(s)")
        for result in offenders[:15]:
            print(f"  {result['file']}: {json.dumps(result[key], ensure_ascii=False)}")
        if len(offenders) > 15:
            print(f"  ... {len(offenders) - 15} more")

    size_report(args.output_dir)

    if args.json:
        with args.json.open("w", encoding="utf-8") as handle:
            json.dump(reports, handle, ensure_ascii=False, indent=2)
        print(f"\nFull report written to {args.json}")


if __name__ == "__main__":
    main()
