"""Phase 1: structural metadata extraction for Solidity smart contracts.

Compiles a Solidity file with Slither, runs its static detectors and emits a
``contract_metadata.json`` document that is fed to the Phase 2 Planner Agent.

Use it from the command line::

    python slither_metadata_extractor.py contracts/SimpleVault.sol -o contract_metadata.json

or from another Python file::

    from slither_metadata_extractor import extract_metadata, extract_directory

    metadata = extract_metadata("contracts/SimpleVault.sol")
    results = extract_directory("datasets/smartbugs-curated/dataset", output_dir="phase1_output")
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

__all__ = [
    "METADATA_SCHEMA",
    "PLANNER_SYSTEM_PROMPT",
    "PLANNER_USER_PROMPT_TEMPLATE",
    "SlitherMetadataExtractor",
    "build_planner_prompt",
    "build_unified_payload",
    "extract_directory",
    "extract_metadata",
    "main",
    "validate_metadata",
]

PathLike = Union[str, os.PathLike]

IMPACT_RANK = {"High": 0, "Medium": 1, "Low": 2, "Informational": 3, "Optimization": 4}

LATEST_PATCH = {(0, 4): 26, (0, 5): 17, (0, 6): 12, (0, 7): 6, (0, 8): 30}
NO_PRAGMA_FALLBACK = ["0.4.26", "0.5.17", "0.6.12", "0.8.30"]
MAX_COMPILE_ATTEMPTS = 4

LINE_RANGE_SCHEMA: Dict[str, Any] = {
    "type": ["object", "null"],
    "required": ["start", "end"],
    "properties": {
        "start": {"type": "integer", "minimum": 1},
        "end": {"type": "integer", "minimum": 1},
    },
    "additionalProperties": False,
}

METADATA_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["contract_file", "status", "summary", "detectors_findings", "structural_metadata"],
    "properties": {
        "contract_file": {"type": "string"},
        "status": {"enum": ["success", "failed"]},
        "error": {"type": "string"},
        "solc_version": {"type": ["string", "null"]},
        "summary": {
            "type": "object",
            "required": ["total_findings", "high_impact", "medium_impact", "low_informational_impact"],
            "properties": {
                "total_findings": {"type": "integer", "minimum": 0},
                "high_impact": {"type": "integer", "minimum": 0},
                "medium_impact": {"type": "integer", "minimum": 0},
                "low_informational_impact": {"type": "integer", "minimum": 0},
            },
        },
        "detectors_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "check",
                    "impact",
                    "confidence",
                    "description",
                    "first_element",
                    "lines",
                    "problem_lines",
                ],
                "properties": {
                    "check": {"type": "string"},
                    "impact": {"type": "string"},
                    "confidence": {"type": "string"},
                    "description": {"type": "string"},
                    "first_element": {"type": "string"},
                    "lines": LINE_RANGE_SCHEMA,
                    "problem_lines": {"type": "array", "items": {"type": "integer", "minimum": 1}},
                },
            },
        },
        "structural_metadata": {
            "type": "object",
            "required": ["contract_name", "state_variables", "functions"],
            "properties": {
                "contract_name": {"type": ["string", "null"]},
                "kind": {"type": "string"},
                "inheritance": {"type": "array", "items": {"type": "string"}},
                "state_variables": {
                    "type": "array",
                    "items": {"type": "object", "required": ["name", "type", "visibility"]},
                },
                "modifiers": {"type": "array"},
                "functions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "name",
                            "visibility",
                            "is_payable",
                            "replaces",
                            "modifiers",
                            "state_variables_read",
                            "state_variables_written",
                            "calls",
                            "external_calls",
                            "lines",
                        ],
                        "properties": {
                            "lines": LINE_RANGE_SCHEMA,
                            "replaces": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Inherited functions this one overrides.",
                            },
                        },
                    },
                },
                "call_graph": {"type": "array"},
                "additional_contracts": {"type": "array"},
            },
        },
    },
}

PLANNER_SYSTEM_PROMPT = """You are an elite Smart Contract Security Audit Planner specializing in Solidity.
Your task is to analyze deterministic static analysis metadata (combining Slither semantic
findings with ANTLR syntactic facts) and to construct a focused, step-by-step Audit Plan.

### Core objectives
1. Identify high-risk functions: state-changing methods, Ether handlers, access-control
   gates, and structural anomalies.
2. Combine Slither detector findings with ANTLR syntactic facts (legacy constructors,
   implicit visibility, overridden functions).
3. Deconstruct the audit into 3 to 5 targeted, non-overlapping sub-tasks.
4. Emit a strictly structured JSON execution plan for the downstream agents.

### Rules
- Do NOT write exploit code or deliver final verdicts. Your role is PLANNING only.
- Prioritize High impact findings (reentrancy, access-control bypass, unchecked low-level
  calls, arithmetic).
- Prefer the `function_signals` table for planning: it lists the functions that actually
  execute on the deployed contract. Use `replaces` to see which inherited
  implementations each one overrides.
- If `capabilities` does not contain `call_graph` or `read_write_maps`, do NOT reason about
  reentrancy paths for this contract; say so in the rationale instead.
- Every step must provide: `step_number`, `focus_area`, `target_functions`,
  `key_rationale`, `recommended_checks`.

### Output format
Respond ONLY with a valid JSON object of this shape:

{
  "contract_name": "<string>",
  "risk_assessment_summary": "<string>",
  "audit_plan": [
    {
      "step_number": 1,
      "focus_area": "<string>",
      "target_functions": ["<function_name>"],
      "key_rationale": "<string>",
      "recommended_checks": ["<check_description>"]
    }
  ]
}
"""

PLANNER_USER_PROMPT_TEMPLATE = """Analyze the Unified Static Analysis Payload below for the Solidity
contract "{contract_name}" and produce a 3-to-5 step Audit Plan.

### Unified Static Analysis Payload
```json
{payload_json}
```

### Source context
Primary contract: {primary_contract}
Solidity version: {solc_version}

Respond with the JSON audit plan matching the required schema."""

MAX_DESCRIPTION_CHARS = 400
PRIORITY_FINDING_KEYS = ("check", "impact", "confidence", "first_element", "lines", "problem_lines")


def _normalized_function_name(name: Optional[str], kind: Optional[str], contract: Optional[str]) -> str:
    """Give anonymous Solidity functions a usable label.

    ANTLR reports the constructor and the fallback with ``name: null``. Left as-is they
    become the string ``"None"`` in an f-string and the Planner cannot tell what it is
    reading.
    """
    if kind == "fallback" or name in (None, "fallback"):
        return "fallback"
    if kind == "receive" or name == "receive":
        return "receive"
    if kind == "constructor" or name == "constructor":
        return "constructor"
    if name is None and contract:
        return "constructor"
    return name or "unnamed"


def _shorten(text: Any, limit: int = MAX_DESCRIPTION_CHARS) -> str:
    """Trim long prose while keeping the first and last clause readable."""
    if not isinstance(text, str) or len(text) <= limit:
        return text if isinstance(text, str) else ""
    head = text[: limit - 40].rstrip()
    return f"{head} ... [truncated]"

_PRAGMA_RE = re.compile(r"pragma\s+solidity\s+([^;]+);")
_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+)")


def detect_pragma_version(source: str) -> Optional[str]:
    """Return the lowest concrete solc version allowed by the first ``pragma solidity``."""
    pragma = _PRAGMA_RE.search(source)
    if not pragma:
        return None
    expression = pragma.group(1)
    for match in _VERSION_RE.finditer(expression):
        prefix = expression[: match.start()].rstrip()
        if prefix.endswith("<") or (prefix.endswith(">") and not prefix.endswith(">=")):
            continue
        return match.group(1)
    return None


def candidate_versions(source: str) -> List[str]:
    """Compiler versions to try, in order.

    The pragma's lower bound comes first, then the newest release the pragma allows,
    then the remaining releases in ascending order. Trying the newest early handles
    contracts that use syntax newer than their pragma claims (e.g. ``emit`` under
    ``^0.4.16``). A file without a pragma gets ``NO_PRAGMA_FALLBACK``.
    """
    lower = detect_pragma_version(source)
    if lower is None:
        return [] if _PRAGMA_RE.search(source) else list(NO_PRAGMA_FALLBACK)
    expression = _PRAGMA_RE.search(source).group(1).strip()
    if re.fullmatch(r"=?\s*\d+\.\d+\.\d+", expression):
        return [lower]
    major, minor, patch = (int(part) for part in lower.split("."))
    latest = LATEST_PATCH.get((major, minor), patch)
    upper = re.search(r"<(=?)\s*(\d+)\.(\d+)\.(\d+)", expression)
    versions = []
    for candidate in range(patch, latest + 1):
        if upper:
            bound = tuple(int(part) for part in upper.group(2, 3, 4))
            current = (major, minor, candidate)
            if current > bound or (current == bound and not upper.group(1)):
                break
        versions.append(f"{major}.{minor}.{candidate}")
    if len(versions) > 2:
        versions = [versions[0], versions[-1]] + versions[1:-1]
    return versions


def prepare_solc(version: str) -> bool:
    """Install ``version`` with solc-select and point the solc shim at it."""
    if not shutil.which("solc-select"):
        return False
    result = subprocess.run(["solc-select", "install", version], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        os.environ.pop("SOLC_VERSION", None)
        return False
    os.environ["SOLC_VERSION"] = version
    return True


def _name(obj: Any) -> str:
    if obj is None:
        return "unknown"
    return str(getattr(obj, "name", obj))


def _unique(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(items))


def _line_range(lines: Iterable[int]) -> Optional[Dict[str, int]]:
    """Collapse a list of line numbers into ``{"start": first, "end": last}`` (None if empty)."""
    numbers = sorted(set(lines))
    if not numbers:
        return None
    return {"start": numbers[0], "end": numbers[-1]}


def _source_lines(obj: Any) -> Optional[Dict[str, int]]:
    mapping = getattr(obj, "source_mapping", None)
    if mapping is None:
        return None
    lines = mapping.get("lines", []) if isinstance(mapping, dict) else getattr(mapping, "lines", [])
    return _line_range(lines or [])


def _source_file(obj: Any) -> Optional[str]:
    mapping = getattr(obj, "source_mapping", None)
    filename = getattr(mapping, "filename", None)
    absolute = getattr(filename, "absolute", None)
    return os.path.normcase(os.path.abspath(absolute)) if absolute else None


def _internal_callees(function: Any) -> List[Any]:
    from slither.core.declarations import Function, Modifier

    callees = []
    for call in getattr(function, "internal_calls", []):
        target = getattr(call, "function", call)
        if isinstance(target, Function) and not isinstance(target, Modifier):
            callees.append(target)
    return callees


def _high_level_targets(function: Any) -> List[str]:
    targets = []
    for contract, call in getattr(function, "high_level_calls", []):
        target = call if hasattr(call, "canonical_name") else getattr(call, "function", None)
        if target is not None and hasattr(target, "canonical_name"):
            targets.append(str(target.canonical_name))
        else:
            member = getattr(call, "function_name", None) or _name(call)
            targets.append(f"{_name(contract)}.{member}")
    return _unique(targets)


def _low_level_targets(function: Any) -> List[str]:
    targets = []
    for item in getattr(function, "low_level_calls", []):
        if isinstance(item, tuple):
            destination, member = item
            targets.append(f"{_name(destination)}.{member}")
            continue
        member = getattr(item, "function_name", "")
        called = getattr(getattr(item, "expression", None), "called", None)
        receiver = getattr(called, "expression", None)
        if receiver is not None:
            targets.append(f"{receiver}.{member}")
        else:
            targets.append(f"{_name(getattr(item, 'destination', None))}.{member}")
    return _unique(targets)


def _contract_kind(contract: Any) -> str:
    kind = getattr(contract, "contract_kind", None)
    if kind:
        return str(kind)
    if getattr(contract, "is_interface", False):
        return "interface"
    if getattr(contract, "is_library", False):
        return "library"
    return "contract"


class SlitherMetadataExtractor:
    """Extracts Slither findings and structural metadata for the Phase 2 Planner Agent."""

    def __init__(
        self,
        target_contract_path: PathLike,
        solc_version: Optional[str] = None,
        solc_path: Optional[str] = None,
        include_informational: bool = False,
    ):
        if not os.path.isfile(target_contract_path):
            raise FileNotFoundError(f"Contract file not found: {target_contract_path}")
        self.contract_path = os.fspath(target_contract_path)
        self.solc_version = solc_version
        self.solc_path = solc_path
        self.include_informational = include_informational

    def _load_slither(self) -> Any:
        previous_solc_version = os.environ.get("SOLC_VERSION")
        try:
            return self._compile_with_fallback()
        finally:
            if previous_solc_version is None:
                os.environ.pop("SOLC_VERSION", None)
            else:
                os.environ["SOLC_VERSION"] = previous_solc_version

    def _compile_with_fallback(self) -> Any:
        from slither import Slither

        kwargs: Dict[str, Any] = {"compile_force_framework": "solc", "cwd": tempfile.gettempdir()}
        if self.solc_path:
            kwargs["solc"] = self.solc_path
            return Slither(os.path.abspath(self.contract_path), **kwargs)

        if self.solc_version:
            candidates = [self.solc_version]
        else:
            source = Path(self.contract_path).read_text(encoding="utf-8", errors="replace")
            candidates = candidate_versions(source) or [None]

        first_error: Optional[Exception] = None
        attempts = 0
        for version in candidates:
            if attempts == MAX_COMPILE_ATTEMPTS:
                break
            if version is not None and not prepare_solc(version):
                continue
            attempts += 1
            try:
                slither = Slither(os.path.abspath(self.contract_path), **kwargs)
            except Exception as exc:
                first_error = first_error or exc
                continue
            self.solc_version = version
            return slither
        if first_error is None:
            raise RuntimeError(f"No usable solc version could be installed from {candidates}")
        raise first_error

    def _run_detectors(self, slither: Any) -> List[Dict[str, Any]]:
        from slither.detectors import all_detectors
        from slither.detectors.abstract_detector import AbstractDetector, DetectorClassification

        excluded = {DetectorClassification.OPTIMIZATION}
        if not self.include_informational:
            excluded.add(DetectorClassification.INFORMATIONAL)

        for detector in vars(all_detectors).values():
            if (
                isinstance(detector, type)
                and issubclass(detector, AbstractDetector)
                and detector is not AbstractDetector
                and detector.IMPACT not in excluded
            ):
                slither.register_detector(detector)

        results = [finding for group in slither.run_detectors() for finding in group]
        results.sort(key=lambda finding: (IMPACT_RANK.get(finding.get("impact"), 9), finding.get("check", "")))
        return results

    @staticmethod
    def _format_finding(finding: Dict[str, Any]) -> Dict[str, Any]:
        """Shape one Slither finding.

        ``lines`` is the start/end range of the first element (the function or contract
        the finding is about). ``problem_lines`` are the exact statement lines Slither
        points at (its "node" elements); empty when the finding is about a whole
        function or contract rather than specific statements.
        """
        elements = finding.get("elements") or []
        first = elements[0] if elements else {}
        problem_lines = sorted(
            {
                line
                for element in elements
                if element.get("type") == "node"
                for line in (element.get("source_mapping") or {}).get("lines") or []
            }
        )
        return {
            "check": finding.get("check", ""),
            "impact": finding.get("impact", ""),
            "confidence": finding.get("confidence", ""),
            "description": finding.get("description", ""),
            "first_element": str(first.get("name") or "N/A"),
            "lines": _line_range((first.get("source_mapping") or {}).get("lines") or []),
            "problem_lines": problem_lines,
        }

    @staticmethod
    def _describe_function(function: Any, replaced: Optional[List[str]] = None) -> Dict[str, Any]:
        return {
            "name": function.name,
            "signature": function.full_name,
            "declared_in": _name(getattr(function, "contract_declarer", None)),
            "visibility": function.visibility,
            "is_payable": bool(function.payable),
            "is_constructor": bool(function.is_constructor),
            "is_view": bool(function.view),
            "is_pure": bool(function.pure),
            "modifiers": _unique(_name(modifier) for modifier in function.modifiers),
            "state_variables_read": _unique(_name(var) for var in function.state_variables_read),
            "state_variables_written": _unique(_name(var) for var in function.state_variables_written),
            "calls": _unique(_name(callee) for callee in _internal_callees(function)),
            "external_calls": _high_level_targets(function) + _low_level_targets(function),
            "sends_eth": bool(function.can_send_eth()),
            "replaces": replaced or [],
            "lines": _source_lines(function),
        }

    def _describe_contract(self, contract: Any) -> Dict[str, Any]:
        functions = [
            function
            for function in contract.functions
            if not function.is_shadowed and function.name not in ("slitherConstructorVariables", "slitherConstructorConstantVariables")
        ]
        overridden_map: Dict[str, List[str]] = {}
        for function in contract.functions:
            if not function.is_shadowed:
                continue
            for candidate in functions:
                if candidate.name == function.name and len(candidate.parameters) == len(function.parameters):
                    overridden_map.setdefault(candidate.canonical_name, []).append(function.canonical_name)
                    break
        call_graph = []
        for function in functions:
            for callee in _internal_callees(function):
                call_graph.append({"from": function.canonical_name, "to": callee.canonical_name, "type": "internal"})
            for target in _high_level_targets(function):
                call_graph.append({"from": function.canonical_name, "to": target, "type": "external"})
            for target in _low_level_targets(function):
                call_graph.append({"from": function.canonical_name, "to": target, "type": "low_level"})

        return {
            "contract_name": contract.name,
            "kind": _contract_kind(contract),
            "inheritance": [_name(parent) for parent in contract.inheritance],
            "state_variables": [
                {
                    "name": var.name,
                    "type": str(var.type),
                    "visibility": var.visibility,
                    "is_constant": bool(getattr(var, "is_constant", False)),
                    "is_immutable": bool(getattr(var, "is_immutable", False)),
                }
                for var in contract.state_variables
            ],
            "modifiers": [
                {
                    "name": modifier.name,
                    "state_variables_read": _unique(_name(var) for var in modifier.state_variables_read),
                }
                for modifier in contract.modifiers
            ],
            "functions": [
                self._describe_function(function, replaced=overridden_map.get(function.canonical_name, []))
                for function in functions
            ],
            "call_graph": call_graph,
        }

    def _target_contracts(self, slither: Any) -> List[Any]:
        target = os.path.normcase(os.path.abspath(self.contract_path))
        in_file = [contract for contract in slither.contracts if _source_file(contract) in (target, None)]
        candidates = in_file or list(slither.contracts)
        stem = Path(self.contract_path).stem
        derived = set(slither.contracts_derived)

        def priority(contract: Any) -> tuple:
            return (
                contract.name != stem,
                _contract_kind(contract) != "contract",
                contract not in derived,
                -len(contract.functions),
            )

        return sorted(candidates, key=priority)

    def extract_structured_metadata(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "contract_file": os.path.basename(self.contract_path),
            "status": "failed",
            "solc_version": self.solc_version,
            "summary": {"total_findings": 0, "high_impact": 0, "medium_impact": 0, "low_informational_impact": 0},
            "detectors_findings": [],
            "structural_metadata": {"contract_name": None, "state_variables": [], "functions": []},
        }

        try:
            try:
                slither = self._load_slither()
            finally:
                metadata["solc_version"] = self.solc_version
            contracts = self._target_contracts(slither)
            if contracts:
                structural = self._describe_contract(contracts[0])
                structural["additional_contracts"] = [self._describe_contract(other) for other in contracts[1:]]
                metadata["structural_metadata"] = structural
            findings = [self._format_finding(finding) for finding in self._run_detectors(slither)]
        except Exception as exc:
            metadata["error"] = f"{type(exc).__name__}: {exc}"
            return metadata

        summary = metadata["summary"]
        summary["total_findings"] = len(findings)
        for finding in findings:
            if finding["impact"] == "High":
                summary["high_impact"] += 1
            elif finding["impact"] == "Medium":
                summary["medium_impact"] += 1
            else:
                summary["low_informational_impact"] += 1
        metadata["detectors_findings"] = findings
        metadata["status"] = "success"
        return metadata

    def save_metadata_to_json(self, output_path: PathLike = "contract_metadata.json") -> Dict[str, Any]:
        data = self.extract_structured_metadata()
        _write_json(data, output_path)
        return data


def _write_json(data: Dict[str, Any], output_path: PathLike) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=4)


def extract_metadata(
    contract_path: PathLike,
    output_path: Optional[PathLike] = None,
    *,
    solc_version: Optional[str] = None,
    solc_path: Optional[str] = None,
    include_informational: bool = False,
) -> Dict[str, Any]:
    """Extract Phase 1 metadata for one Solidity file and return it as a dict.

    Compilation and analysis errors do not raise; they come back as
    ``status == "failed"`` with an ``error`` message. Only a missing file raises
    ``FileNotFoundError``. When ``output_path`` is given, the JSON is also written there.
    """
    extractor = SlitherMetadataExtractor(
        contract_path,
        solc_version=solc_version,
        solc_path=solc_path,
        include_informational=include_informational,
    )
    metadata = extractor.extract_structured_metadata()
    if output_path is not None:
        _write_json(metadata, output_path)
    return metadata


def extract_directory(
    directory: PathLike,
    output_dir: Optional[PathLike] = None,
    *,
    solc_version: Optional[str] = None,
    solc_path: Optional[str] = None,
    include_informational: bool = False,
    on_result: Optional[Callable[[Path, Dict[str, Any]], None]] = None,
) -> Dict[Path, Dict[str, Any]]:
    """Extract metadata for every ``.sol`` file under ``directory`` (recursively).

    Returns ``{contract_path: metadata}`` in sorted path order. When ``output_dir``
    is given, each result is written as ``<sub__dirs__name>_metadata.json``.
    ``on_result(contract_path, metadata)`` is called after each contract, e.g. for progress.
    """
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    results: Dict[Path, Dict[str, Any]] = {}
    for contract_file in _collect_targets(root):
        output_path = None
        if output_dir is not None:
            output_path = Path(output_dir) / _batch_output_name(root, contract_file)
        metadata = extract_metadata(
            contract_file,
            output_path,
            solc_version=solc_version,
            solc_path=solc_path,
            include_informational=include_informational,
        )
        results[contract_file] = metadata
        if on_result is not None:
            on_result(contract_file, metadata)
    return results


def _batch_output_name(root: Path, contract_file: Path) -> str:
    relative = contract_file.relative_to(root).with_suffix("")
    return f"{'__'.join(relative.parts)}_metadata.json"


def validate_metadata(metadata: Dict[str, Any]) -> List[str]:
    """Validate against METADATA_SCHEMA; returns error messages (empty when valid)."""
    try:
        import jsonschema
    except ImportError:
        return ["jsonschema is not installed; run `pip install jsonschema`"]
    validator = jsonschema.Draft202012Validator(METADATA_SCHEMA)
    return [f"{'/'.join(map(str, error.path)) or '<root>'}: {error.message}" for error in validator.iter_errors(metadata)]


def build_unified_payload(metadata: Dict[str, Any], ast: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge the Slither metadata document with the ANTLR AST document.

    The two extractors answer different questions, so this does not interleave them.
    It keeps the semantic facts and the syntactic facts side by side and adds one
    merged table, ``function_signals``, with a row per function that actually executes
    on the deployed contract. ``ast`` is optional; without it the syntactic section
    reports itself as unavailable instead of pretending the contract is clean.
    """
    structural = metadata.get("structural_metadata")
    structural = structural if isinstance(structural, dict) else {}
    findings = metadata.get("detectors_findings") or []
    ast = ast or {}

    contracts = ast.get("contracts") or []
    implicit_visibility: List[Dict[str, Any]] = []
    legacy_constructors: List[Dict[str, Any]] = []
    unimplemented: List[Dict[str, Any]] = []
    contract_shapes: List[Dict[str, Any]] = []

    for contract in contracts:
        contract_name = contract.get("name")
        contract_shapes.append(
            {
                "name": contract_name,
                "kind": contract.get("kind"),
                "inherits": [
                    parent.get("name") if isinstance(parent, dict) else parent
                    for parent in contract.get("inheritance") or []
                ],
                "lines": contract.get("lines"),
            }
        )
        for function in contract.get("functions") or []:
            kind = function.get("kind")
            if not function.get("visibility_explicit", True):
                implicit_visibility.append(
                    {
                        "contract": contract_name,
                        "function": _normalized_function_name(
                            function.get("name"), kind, contract_name
                        ),
                        "kind": kind,
                    }
                )
            if function.get("is_legacy_constructor"):
                legacy_constructors.append(
                    {
                        "contract": contract_name,
                        "function": function.get("name"),
                        "visibility": function.get("visibility"),
                        "visibility_explicit": function.get("visibility_explicit"),
                    }
                )
            if not function.get("has_body", True):
                unimplemented.append({"contract": contract_name, "function": function.get("name")})

    def findings_for(name: Optional[str]) -> List[Dict[str, Any]]:
        if not name:
            return []
        return [
            {
                "check": finding.get("check"),
                "impact": finding.get("impact"),
                "confidence": finding.get("confidence"),
                "problem_lines": finding.get("problem_lines") or [],
            }
            for finding in findings
            if finding.get("first_element") == name
        ]

    signals: List[Dict[str, Any]] = []
    for function in structural.get("functions") or []:
        kind = "constructor" if function.get("is_constructor") else (
            "fallback" if function.get("name") in (None, "fallback") else None
        )
        plain_name = None if function.get("name") in ("constructor", "fallback") else function.get("name")
        label = _normalized_function_name(
            "constructor" if function.get("is_constructor") else plain_name,
            kind,
            function.get("declared_in"),
        )
        flags: List[str] = []
        if function.get("replaces"):
            flags.append("overrides_ancestor")
        if function.get("external_calls"):
            flags.append("external_call")
        if function.get("sends_eth"):
            flags.append("sends_eth")
        if function.get("is_payable"):
            flags.append("payable")
        if function.get("state_variables_written"):
            flags.append("writes_state")
        if function.get("modifiers"):
            flags.append("guarded_by_modifier")
        touches_state = bool(
            function.get("state_variables_read")
            or function.get("state_variables_written")
            or function.get("calls")
            or function.get("external_calls")
        )
        if not touches_state:
            flags.append("stateless")
        if not function.get("is_view") and not function.get("is_pure") and touches_state:
            flags.append("state_changing")
        matched = findings_for(function.get("name"))
        signals.append(
            {
                "function": label,
                "canonical_signature": function.get("signature"),
                "declared_in": function.get("declared_in"),
                "visibility": function.get("visibility"),
                "flags": flags,
                "state_variables_read": function.get("state_variables_read") or [],
                "state_variables_written": function.get("state_variables_written") or [],
                "external_calls": function.get("external_calls") or [],
                "modifiers": function.get("modifiers") or [],
                "replaces": function.get("replaces") or [],
                "lines": function.get("lines"),
                "finding_count": len(matched),
                "findings": matched,
            }
        )

    capabilities = ["syntactic_facts"] if ast else []
    if metadata.get("status") == "success":
        capabilities.append("semantic_facts")
        if structural.get("call_graph"):
            capabilities.append("call_graph")
        if any(s["state_variables_read"] or s["state_variables_written"] for s in signals):
            capabilities.append("read_write_maps")

    truncated = False
    trimmed_findings: List[Dict[str, Any]] = []
    for finding in findings:
        entry = {key: finding.get(key) for key in PRIORITY_FINDING_KEYS if key in finding}
        description = finding.get("description", "")
        entry["description"] = _shorten(description)
        truncated = truncated or entry["description"] != description
        trimmed_findings.append(entry)

    return {
        "payload_version": "1.0",
        "audit_subject": {
            "primary_contract": structural.get("contract_name"),
            "source_file": metadata.get("contract_file"),
            "kind": structural.get("kind"),
            "solc_version": metadata.get("solc_version"),
            "inheritance_chain": structural.get("inheritance") or [],
            "declared_contracts": [c.get("name") for c in contracts],
            "slither_status": metadata.get("status"),
            "antlr_status": ast.get("status"),
            "degraded": metadata.get("status") != "success",
            "capabilities": capabilities,
            "truncated": truncated,
        },
        "semantic_facts": (
            {
                "summary": metadata.get("summary") or {},
                "detectors_findings": trimmed_findings,
                "call_graph": structural.get("call_graph") or [],
                "state_variables": structural.get("state_variables") or [],
            }
            if metadata.get("status") == "success"
            else None
        ),
        "syntactic_facts": (
            {
                "syntax_errors": ast.get("syntax_errors") or [],
                "implicit_visibility": implicit_visibility,
                "legacy_constructors": legacy_constructors,
                "unimplemented_functions": unimplemented,
                "contract_shapes": contract_shapes,
            }
            if ast
            else None
        ),
        "function_signals": signals if metadata.get("status") == "success" else [],
    }


def build_planner_prompt(
    metadata: Dict[str, Any], ast: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """Build the English System and User prompts for the Planner agent.

    Returns ``{"system": ..., "user": ...}`` ready for an LLM call. Pass the ANTLR
    document as ``ast`` to enable the merged ``function_signals`` table; without it
    the payload still builds and marks itself as degraded.
    """
    payload = build_unified_payload(metadata, ast)
    subject = payload["audit_subject"]
    user_prompt = PLANNER_USER_PROMPT_TEMPLATE.format(
        contract_name=subject["primary_contract"] or subject["source_file"] or "UnknownContract",
        payload_json=json.dumps(payload, ensure_ascii=False, indent=2),
        primary_contract=subject["primary_contract"],
        solc_version=subject["solc_version"],
    )
    return {"system": PLANNER_SYSTEM_PROMPT.strip(), "user": user_prompt.strip()}


def _collect_targets(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(path.rglob("*.sol"))
    return [path]


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: extract Slither-based structural metadata from Solidity contracts."
    )
    parser.add_argument("target", help="Solidity file, or a directory to process every .sol file (batch mode)")
    parser.add_argument("-o", "--output", help="Output JSON file (single mode) or directory (batch mode)")
    parser.add_argument("--solc-version", help="Force a solc version (otherwise detected from pragma)")
    parser.add_argument("--solc", help="Path to a specific solc binary")
    parser.add_argument("--include-informational", action="store_true", help="Also run informational detectors")
    parser.add_argument("--validate", action="store_true", help="Validate output against the JSON schema")
    parser.add_argument("--prompt", help="Also write the Phase 2 Planner prompt to this file (single mode)")
    parser.add_argument("--print-schema", action="store_true", help="Print the output JSON schema and exit")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parse_args(argv)

    if args.print_schema:
        print(json.dumps(METADATA_SCHEMA, indent=2))
        return 0

    target = Path(args.target)
    if not target.exists():
        print(f"Target not found: {target}", file=sys.stderr)
        return 2

    options = {
        "solc_version": args.solc_version,
        "solc_path": args.solc,
        "include_informational": args.include_informational,
    }
    counters = {"failures": 0, "schema_errors": 0}

    def report(contract_file: Path, metadata: Dict[str, Any], output_path: Path) -> None:
        summary = metadata["summary"]
        status = metadata["status"]
        print(
            f"[{status}] {contract_file} -> {output_path} "
            f"(high={summary['high_impact']}, medium={summary['medium_impact']}, "
            f"low/info={summary['low_informational_impact']})"
        )
        if status != "success":
            counters["failures"] += 1
            print(f"    error: {metadata.get('error')}", file=sys.stderr)
        if args.validate:
            errors = validate_metadata(metadata)
            counters["schema_errors"] += bool(errors)
            for error in errors:
                print(f"    schema: {error}", file=sys.stderr)

    if target.is_dir():
        output_dir = Path(args.output or "phase1_output")
        results = extract_directory(
            target,
            output_dir,
            on_result=lambda path, data: report(path, data, output_dir / _batch_output_name(target, path)),
            **options,
        )
        rate = counters["failures"] / len(results) * 100 if results else 0.0
        print(f"\nProcessed {len(results)} contracts: {counters['failures']} failed (crash rate {rate:.2f}%)")
    else:
        output_path = Path(args.output or "contract_metadata.json")
        metadata = extract_metadata(target, output_path, **options)
        report(target, metadata, output_path)
        if args.prompt:
            prompts = build_planner_prompt(metadata)
            Path(args.prompt).write_text(
                f"### SYSTEM\n{prompts['system']}\n\n### USER\n{prompts['user']}\n",
                encoding="utf-8",
            )

    return 1 if counters["failures"] or counters["schema_errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
