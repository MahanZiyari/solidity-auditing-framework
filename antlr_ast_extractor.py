"""Phase 1: Abstract Syntax Tree extraction for Solidity contracts with ANTLR4.

Parses a Solidity file with the ANTLR grammar in ``grammar/Solidity.g4`` and emits
a JSON document with a contract outline (contracts, state variables, functions,
modifiers, events, ...). The full syntax tree is opt-in with ``include_tree=True``,
because it makes the document roughly 50x larger.

Unlike the Slither extractor this needs no compiler, so it works for every
Solidity version. Files with syntax errors still get a tree: the errors are
listed and ``status`` is ``"partial"``.

Use it from the command line::

    python antlr_ast_extractor.py contracts/SimpleVault.sol -o SimpleVault_ast.json
    python antlr_ast_extractor.py datasets/cve -o phase1_output/ast/cve --validate
    python antlr_ast_extractor.py datasets/cve -o phase1_output/ast/cve --include-tree --compact

or from another Python file::

    from antlr_ast_extractor import extract_ast, extract_ast_directory

    document = extract_ast("contracts/SimpleVault.sol")
    results = extract_ast_directory("datasets/cve", output_dir="phase1_output/ast/cve")
    with_tree = extract_ast("contracts/SimpleVault.sol", include_tree=True)

The parser in ``solidity_antlr/`` is generated from the grammar. To regenerate it
(``pip install antlr4-tools``; it downloads Java on first use)::

    cd grammar
    antlr4 -v 4.13.2 -Dlanguage=Python3 -visitor -no-listener -o ../solidity_antlr Solidity.g4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from antlr4 import CommonTokenStream, InputStream
from antlr4.atn.PredictionMode import PredictionMode
from antlr4.error.ErrorListener import ErrorListener
from antlr4.error.ErrorStrategy import BailErrorStrategy, DefaultErrorStrategy
from antlr4.error.Errors import ParseCancellationException
from antlr4.tree.Tree import TerminalNode

from solidity_antlr.SolidityLexer import SolidityLexer
from solidity_antlr.SolidityParser import SolidityParser

__all__ = [
    "AST_SCHEMA",
    "ParseResult",
    "extract_ast",
    "extract_ast_directory",
    "main",
    "parse_source",
    "validate_ast_document",
]

PathLike = Union[str, os.PathLike]

GRAMMAR_DIR = Path(__file__).resolve().parent / "grammar"
ANTLR_VERSION = "4.13.2"
RECURSION_LIMIT = 20000

LINE_RANGE_SCHEMA: Dict[str, Any] = {
    "type": ["object", "null"],
    "required": ["start", "end"],
    "properties": {"start": {"type": "integer", "minimum": 1}, "end": {"type": "integer", "minimum": 1}},
    "additionalProperties": False,
}

_NAMED_WITH_LINES: Dict[str, Any] = {
    "type": "object",
    "required": ["name", "lines"],
    "properties": {"name": {"type": ["string", "null"]}, "lines": LINE_RANGE_SCHEMA},
}

_FUNCTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "name",
        "kind",
        "is_legacy_constructor",
        "visibility",
        "visibility_explicit",
        "state_mutability",
        "modifiers",
        "parameters",
        "returns",
        "has_body",
        "lines",
    ],
    "properties": {
        "kind": {"enum": ["function", "constructor", "fallback", "receive"]},
        "visibility": {"enum": ["public", "external", "internal", "private"]},
        "lines": LINE_RANGE_SCHEMA,
    },
}

AST_NODE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["type", "lines"],
    "properties": {
        "type": {"type": "string"},
        "lines": LINE_RANGE_SCHEMA,
        "text": {"type": "string"},
        "children": {"type": "array", "items": {"$ref": "#/$defs/ast_node"}},
    },
}

AST_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$defs": {"ast_node": AST_NODE_SCHEMA},
    "type": "object",
    "required": [
        "contract_file",
        "status",
        "parser",
        "summary",
        "syntax_errors",
        "pragmas",
        "imports",
        "contracts",
        "free_functions",
        "ast",
    ],
    "properties": {
        "contract_file": {"type": "string"},
        "status": {"enum": ["success", "partial", "failed"]},
        "error": {"type": "string"},
        "parser": {"type": "object", "required": ["tool", "antlr_version", "grammar", "grammar_commit", "parse_seconds"]},
        "summary": {
            "type": "object",
            "required": ["contracts", "functions", "state_variables", "modifiers", "events", "syntax_errors", "ast_nodes"],
            "properties": {
                "ast_nodes": {
                    "type": ["integer", "null"],
                    "description": "Tree node count; null when the tree was not included.",
                }
            },
        },
        "syntax_errors": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["line", "column", "message"],
                "properties": {
                    "line": {"type": "integer"},
                    "column": {"type": "integer"},
                    "message": {"type": "string"},
                },
            },
        },
        "pragmas": {"type": "array", "items": {"type": "object", "required": ["name", "value", "lines"]}},
        "imports": {"type": "array", "items": {"type": "object", "required": ["path", "lines"]}},
        "contracts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "name",
                    "kind",
                    "is_abstract",
                    "inheritance",
                    "lines",
                    "state_variables",
                    "functions",
                    "modifiers",
                    "events",
                    "structs",
                    "enums",
                    "errors",
                    "using_for",
                ],
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "kind": {"enum": ["contract", "interface", "library"]},
                    "lines": LINE_RANGE_SCHEMA,
                    "state_variables": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["name", "type", "visibility", "is_constant", "is_immutable", "lines"],
                            "properties": {"lines": LINE_RANGE_SCHEMA},
                        },
                    },
                    "functions": {"type": "array", "items": _FUNCTION_SCHEMA},
                    "modifiers": {"type": "array", "items": _NAMED_WITH_LINES},
                    "events": {"type": "array", "items": _NAMED_WITH_LINES},
                    "structs": {"type": "array", "items": _NAMED_WITH_LINES},
                    "enums": {"type": "array", "items": _NAMED_WITH_LINES},
                    "errors": {"type": "array", "items": _NAMED_WITH_LINES},
                },
            },
        },
        "free_functions": {"type": "array", "items": _FUNCTION_SCHEMA},
        "ast": {"anyOf": [{"type": "null"}, {"$ref": "#/$defs/ast_node"}]},
    },
}


class _CollectingErrorListener(ErrorListener):
    def __init__(self) -> None:
        super().__init__()
        self.errors: List[Dict[str, Any]] = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):  # noqa: N802 - ANTLR API name
        self.errors.append({"line": int(line), "column": int(column), "message": str(msg)})


class ParseResult:
    """A parsed Solidity file: the parse tree plus what is needed to read it back."""

    def __init__(
        self,
        tree: Any,
        parser: SolidityParser,
        stream: InputStream,
        syntax_errors: List[Dict[str, Any]],
        parse_seconds: float,
    ):
        self.tree = tree
        self.parser = parser
        self.stream = stream
        self.syntax_errors = syntax_errors
        self.parse_seconds = parse_seconds


def _grammar_commit() -> Optional[str]:
    commit_file = GRAMMAR_DIR / "COMMIT"
    if commit_file.is_file():
        return commit_file.read_text(encoding="utf-8").strip() or None
    return None


def _build_parser(stream: InputStream, listener: Optional[ErrorListener]) -> SolidityParser:
    lexer = SolidityLexer(stream)
    lexer.removeErrorListeners()
    parser = SolidityParser(CommonTokenStream(lexer))
    parser.removeErrorListeners()
    if listener is not None:
        lexer.addErrorListener(listener)
        parser.addErrorListener(listener)
    return parser


def parse_source(source: str) -> ParseResult:
    """Parse Solidity source text into an ANTLR parse tree.

    Uses ANTLR's two-stage strategy: a fast SLL pass that stops at the first error,
    then a full LL pass with error recovery only if the fast pass fails. Both give
    the same tree for valid input; the second one also collects syntax errors.
    """
    if sys.getrecursionlimit() < RECURSION_LIMIT:
        sys.setrecursionlimit(RECURSION_LIMIT)
    source = source.lstrip("\ufeff")
    started = time.perf_counter()

    stream = InputStream(source)
    parser = _build_parser(stream, None)
    parser._interp.predictionMode = PredictionMode.SLL
    parser._errHandler = BailErrorStrategy()
    try:
        tree = parser.sourceUnit()
        return ParseResult(tree, parser, stream, [], time.perf_counter() - started)
    except ParseCancellationException:
        pass

    stream = InputStream(source)
    listener = _CollectingErrorListener()
    parser = _build_parser(stream, listener)
    parser._interp.predictionMode = PredictionMode.LL
    parser._errHandler = DefaultErrorStrategy()
    tree = parser.sourceUnit()
    return ParseResult(tree, parser, stream, listener.errors, time.perf_counter() - started)


def _lines(ctx: Any) -> Optional[Dict[str, int]]:
    """Start/end source lines of a parse-tree node (None when it has no tokens)."""
    start = getattr(ctx, "start", None)
    if start is None or start.line is None or start.type == -1:
        return None
    stop = getattr(ctx, "stop", None)
    if stop is None or stop.type == -1 or stop.tokenIndex < start.tokenIndex:
        stop = start
    end_line = stop.line + (stop.text or "").count("\n")
    return {"start": start.line, "end": max(start.line, end_line)}


class _OutlineBuilder:
    """Reads an ANTLR parse tree and builds the contract outline and the JSON AST."""

    def __init__(self, result: ParseResult, include_tokens: bool = True):
        self.result = result
        self.rule_names = result.parser.ruleNames
        self.include_tokens = include_tokens
        self.node_count = 0

    def text(self, ctx: Any) -> Optional[str]:
        """Original source text of ``ctx`` with whitespace collapsed to single spaces."""
        if ctx is None or ctx.start is None or ctx.stop is None:
            return None
        if ctx.stop.type == -1 or ctx.stop.stop < ctx.start.start:
            return None
        return " ".join(self.result.stream.getText(ctx.start.start, ctx.stop.stop).split())

    @staticmethod
    def identifier(ctx: Any) -> Optional[str]:
        if ctx is None or not hasattr(ctx, "identifier"):
            return None
        found = ctx.identifier()
        if isinstance(found, list):
            found = found[0] if found else None
        if found is None or found.getText().startswith("<missing"):
            return None
        return found.getText()

    def named(self, ctx: Any) -> Dict[str, Any]:
        return {"name": self.identifier(ctx), "lines": _lines(ctx)}

    def parameters(self, parameter_list: Any) -> List[Dict[str, Any]]:
        if parameter_list is None:
            return []
        result = []
        for param in parameter_list.parameter():
            location = param.storageLocation()
            result.append(
                {
                    "name": self.identifier(param),
                    "type": self.text(param.typeName()),
                    "storage_location": location.getText() if location is not None else None,
                }
            )
        return result

    def state_variable(self, ctx: Any) -> Dict[str, Any]:
        if ctx.PublicKeyword():
            visibility = "public"
        elif ctx.PrivateKeyword():
            visibility = "private"
        else:
            visibility = "internal"
        initial = ctx.expression()
        return {
            "name": self.identifier(ctx),
            "type": self.text(ctx.typeName()),
            "visibility": visibility,
            "is_constant": bool(ctx.ConstantKeyword()),
            "is_immutable": bool(ctx.ImmutableKeyword()),
            "initial_value": self.text(initial) if initial is not None else None,
            "lines": _lines(ctx),
        }

    def function(self, ctx: Any, contract_name: Optional[str], contract_kind: Optional[str]) -> Dict[str, Any]:
        descriptor = ctx.functionDescriptor()
        options = ctx.modifierList()
        if descriptor.ConstructorKeyword() is not None:
            kind, name = "constructor", None
        elif descriptor.FallbackKeyword() is not None:
            kind, name = "fallback", None
        elif descriptor.ReceiveKeyword() is not None:
            kind, name = "receive", None
        else:
            name = self.identifier(descriptor)
            kind = "function" if name is not None else "fallback"

        explicit = None
        for keywords, value in (
            (options.ExternalKeyword(), "external"),
            (options.PublicKeyword(), "public"),
            (options.InternalKeyword(), "internal"),
            (options.PrivateKeyword(), "private"),
        ):
            if keywords:
                explicit = value
                break
        visibility = explicit or ("external" if contract_kind == "interface" else "public")
        mutability = [item.getText() for item in options.stateMutability()]
        returns = ctx.returnParameters()
        return {
            "name": name,
            "kind": kind,
            "is_legacy_constructor": kind == "function" and name is not None and name == contract_name,
            "visibility": visibility,
            "visibility_explicit": explicit is not None,
            "state_mutability": mutability[0] if mutability else "nonpayable",
            "is_virtual": bool(options.VirtualKeyword()),
            "modifiers": [
                {"name": self.identifier(invocation), "arguments": self.text(invocation.expressionList())}
                for invocation in options.modifierInvocation()
            ],
            "parameters": self.parameters(ctx.parameterList()),
            "returns": self.parameters(returns.parameterList()) if returns is not None else [],
            "has_body": ctx.block() is not None,
            "lines": _lines(ctx),
        }

    def contract(self, ctx: Any) -> Dict[str, Any]:
        name = self.identifier(ctx)
        is_abstract = ctx.getChild(0).getText() == "abstract"
        kind_token = ctx.getChild(1 if is_abstract else 0).getText()
        kind = kind_token if kind_token in ("contract", "interface", "library") else "contract"
        outline: Dict[str, Any] = {
            "name": name,
            "kind": kind,
            "is_abstract": is_abstract,
            "inheritance": [
                {"name": self.text(spec.userDefinedTypeName()), "arguments": self.text(spec.expressionList())}
                for spec in ctx.inheritanceSpecifier()
            ],
            "lines": _lines(ctx),
            "state_variables": [],
            "functions": [],
            "modifiers": [],
            "events": [],
            "structs": [],
            "enums": [],
            "errors": [],
            "using_for": [],
        }
        for part in ctx.contractPart():
            if part.stateVariableDeclaration() is not None:
                outline["state_variables"].append(self.state_variable(part.stateVariableDeclaration()))
            elif part.functionDefinition() is not None:
                outline["functions"].append(self.function(part.functionDefinition(), name, kind))
            elif part.modifierDefinition() is not None:
                modifier = part.modifierDefinition()
                outline["modifiers"].append({**self.named(modifier), "parameters": self.parameters(modifier.parameterList())})
            elif part.eventDefinition() is not None:
                outline["events"].append(self.named(part.eventDefinition()))
            elif part.structDefinition() is not None:
                outline["structs"].append(self.named(part.structDefinition()))
            elif part.enumDefinition() is not None:
                outline["enums"].append(self.named(part.enumDefinition()))
            elif part.customErrorDefinition() is not None:
                outline["errors"].append(self.named(part.customErrorDefinition()))
            elif part.usingForDeclaration() is not None:
                declaration = part.usingForDeclaration()
                outline["using_for"].append({"declaration": self.text(declaration), "lines": _lines(declaration)})
        return outline

    def to_json_tree(self, ctx: Any) -> Optional[Dict[str, Any]]:
        """Convert the ANTLR parse tree into nested dicts.

        Rule nodes become ``{"type": <grammar rule>, "lines": ..., "children": [...]}``.
        Tokens become ``{"type": "token", "text": ..., "lines": ...}``. With
        ``include_tokens`` off, tokens are dropped and rule nodes made only of tokens
        (identifiers, literals, type names) keep their source as ``text`` instead.
        """
        if isinstance(ctx, TerminalNode):
            token = ctx.getSymbol()
            if token.type == -1:
                return None
            self.node_count += 1
            return {"type": "token", "text": token.text, "lines": {"start": token.line, "end": token.line}}

        self.node_count += 1
        node: Dict[str, Any] = {"type": self.rule_names[ctx.getRuleIndex()], "lines": _lines(ctx)}
        children: List[Dict[str, Any]] = []
        has_rule_child = False
        for child in ctx.children or []:
            if isinstance(child, TerminalNode):
                if self.include_tokens:
                    converted = self.to_json_tree(child)
                    if converted is not None:
                        children.append(converted)
                continue
            has_rule_child = True
            converted = self.to_json_tree(child)
            if converted is not None:
                children.append(converted)
        if not self.include_tokens and not has_rule_child:
            text = self.text(ctx)
            if text is not None:
                node["text"] = text
        if children:
            node["children"] = children
        return node


def _empty_document(contract_path: Path) -> Dict[str, Any]:
    return {
        "contract_file": contract_path.name,
        "status": "failed",
        "parser": {
            "tool": "antlr4",
            "antlr_version": ANTLR_VERSION,
            "grammar": "solidity-parser/antlr Solidity.g4",
            "grammar_commit": _grammar_commit(),
            "parse_seconds": 0.0,
        },
        "summary": {
            "contracts": 0,
            "functions": 0,
            "state_variables": 0,
            "modifiers": 0,
            "events": 0,
            "syntax_errors": 0,
            "ast_nodes": 0,
        },
        "syntax_errors": [],
        "pragmas": [],
        "imports": [],
        "contracts": [],
        "free_functions": [],
        "ast": None,
    }


def _build_document(contract_path: Path, result: ParseResult, include_tree: bool, include_tokens: bool) -> Dict[str, Any]:
    document = _empty_document(contract_path)
    builder = _OutlineBuilder(result, include_tokens=include_tokens)
    tree = result.tree

    document["pragmas"] = [
        {
            "name": pragma.pragmaName().getText(),
            "value": builder.text(pragma.pragmaValue()),
            "lines": _lines(pragma),
        }
        for pragma in tree.pragmaDirective()
        if pragma.pragmaName() is not None
    ]
    document["imports"] = [
        {
            "path": item.importPath().getText().strip("\"'") if item.importPath() is not None else None,
            "lines": _lines(item),
        }
        for item in tree.importDirective()
    ]
    document["contracts"] = [builder.contract(contract) for contract in tree.contractDefinition()]
    document["free_functions"] = [builder.function(function, None, None) for function in tree.functionDefinition()]
    document["syntax_errors"] = result.syntax_errors
    document["parser"]["parse_seconds"] = round(result.parse_seconds, 4)
    if include_tree:
        document["ast"] = builder.to_json_tree(tree)

    contracts = document["contracts"]
    document["summary"] = {
        "contracts": len(contracts),
        "functions": sum(len(c["functions"]) for c in contracts) + len(document["free_functions"]),
        "state_variables": sum(len(c["state_variables"]) for c in contracts),
        "modifiers": sum(len(c["modifiers"]) for c in contracts),
        "events": sum(len(c["events"]) for c in contracts),
        "syntax_errors": len(result.syntax_errors),
        "ast_nodes": builder.node_count if include_tree else None,
    }
    document["status"] = "partial" if result.syntax_errors else "success"
    return document


def _write_json(data: Dict[str, Any], output_path: PathLike, compact: bool = False) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        if compact:
            json.dump(data, handle, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(data, handle, ensure_ascii=False, indent=2)


def extract_ast(
    contract_path: PathLike,
    output_path: Optional[PathLike] = None,
    *,
    include_tree: bool = False,
    include_tokens: bool = False,
    compact: bool = False,
) -> Dict[str, Any]:
    """Parse one Solidity file with ANTLR and return the AST document as a dict.

    ``status`` is ``"success"`` (clean parse), ``"partial"`` (syntax errors, tree built
    with error recovery) or ``"failed"`` (the file could not be read or parsed at all,
    with ``error`` set). Only a missing file raises ``FileNotFoundError``.

    The default document is the contract outline, which is small enough to hand to an
    LLM. ``include_tree=True`` adds the full syntax tree (about 50x larger);
    ``include_tokens=True`` also keeps every token (keywords, punctuation) as its own
    node, roughly doubling the tree again, and requires ``include_tree=True``.
    ``compact`` drops the pretty-printing whitespace when writing to ``output_path``.
    """
    if include_tokens and not include_tree:
        raise ValueError("include_tokens=True requires include_tree=True")
    path = Path(contract_path)
    if not path.is_file():
        raise FileNotFoundError(f"Contract file not found: {path}")
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        document = _build_document(path, parse_source(source), include_tree, include_tokens)
    except Exception as exc:
        document = _empty_document(path)
        document["error"] = f"{type(exc).__name__}: {exc}"
    if output_path is not None:
        _write_json(document, output_path, compact=compact)
    return document


def _collect_targets(path: Path) -> List[Path]:
    return sorted(path.rglob("*.sol")) if path.is_dir() else [path]


def _batch_output_name(root: Path, contract_file: Path) -> str:
    relative = contract_file.relative_to(root).with_suffix("")
    return f"{'__'.join(relative.parts)}_ast.json"


def extract_ast_directory(
    directory: PathLike,
    output_dir: Optional[PathLike] = None,
    *,
    include_tree: bool = False,
    include_tokens: bool = False,
    compact: bool = False,
    on_result: Optional[Callable[[Path, Dict[str, Any]], None]] = None,
) -> Dict[Path, Dict[str, Any]]:
    """Parse every ``.sol`` file under ``directory`` (recursively).

    Returns ``{contract_path: document}`` in sorted path order. With ``output_dir``,
    each document is written as ``<sub__dirs__name>_ast.json``. ``on_result`` is
    called after each file, e.g. for progress output. ``include_tree``, ``include_tokens``
    and ``compact`` are forwarded to :func:`extract_ast`.
    """
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    results: Dict[Path, Dict[str, Any]] = {}
    for contract_file in _collect_targets(root):
        output_path = Path(output_dir) / _batch_output_name(root, contract_file) if output_dir is not None else None
        document = extract_ast(
            contract_file,
            output_path,
            include_tree=include_tree,
            include_tokens=include_tokens,
            compact=compact,
        )
        results[contract_file] = document
        if on_result is not None:
            on_result(contract_file, document)
    return results


def validate_ast_document(document: Dict[str, Any]) -> List[str]:
    """Validate against AST_SCHEMA; returns error messages (empty when valid)."""
    try:
        import jsonschema
    except ImportError:
        return ["jsonschema is not installed; run `pip install jsonschema`"]
    validator = jsonschema.Draft202012Validator(AST_SCHEMA)
    return [f"{'/'.join(map(str, error.path)) or '<root>'}: {error.message}" for error in validator.iter_errors(document)]


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1: extract the ANTLR syntax tree and outline of Solidity contracts.")
    parser.add_argument("target", help="Solidity file, or a directory to process every .sol file (batch mode)")
    parser.add_argument("-o", "--output", help="Output JSON file (single mode) or directory (batch mode)")
    parser.add_argument("--include-tree", action="store_true", help="Also keep the full syntax tree (about 50x larger)")
    parser.add_argument("--outline-only", action="store_true", help="Deprecated: the outline is now the default")
    parser.add_argument("--include-tokens", action="store_true", help="Keep every token as its own tree node (needs --include-tree)")
    parser.add_argument("--compact", action="store_true", help="Write JSON without pretty-printing whitespace")
    parser.add_argument("--validate", action="store_true", help="Validate output against the JSON schema")
    parser.add_argument("--print-schema", action="store_true", help="Print the output JSON schema and exit")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parse_args(argv)
    if args.print_schema:
        print(json.dumps(AST_SCHEMA, indent=2))
        return 0

    target = Path(args.target)
    if not target.exists():
        print(f"Target not found: {target}", file=sys.stderr)
        return 2

    if args.include_tokens and not args.include_tree:
        print("--include-tokens requires --include-tree", file=sys.stderr)
        return 2
    options = {
        "include_tree": args.include_tree,
        "include_tokens": args.include_tokens,
        "compact": args.compact,
    }
    counts = {"success": 0, "partial": 0, "failed": 0, "schema_errors": 0}

    def report(contract_file: Path, document: Dict[str, Any], output_path: Path) -> None:
        summary = document["summary"]
        counts[document["status"]] += 1
        nodes = f"nodes={summary['ast_nodes']}, " if summary["ast_nodes"] is not None else ""
        print(
            f"[{document['status']}] {contract_file} -> {output_path} "
            f"(contracts={summary['contracts']}, functions={summary['functions']}, "
            f"{nodes}errors={summary['syntax_errors']}, "
            f"{document['parser']['parse_seconds']:.2f}s)"
        )
        if document["status"] == "failed":
            print(f"    error: {document.get('error')}", file=sys.stderr)
        for error in document["syntax_errors"][:3]:
            print(f"    syntax: line {error['line']}:{error['column']} {error['message']}", file=sys.stderr)
        if args.validate:
            errors = validate_ast_document(document)
            counts["schema_errors"] += bool(errors)
            for error in errors[:5]:
                print(f"    schema: {error}", file=sys.stderr)

    if target.is_dir():
        output_dir = Path(args.output or "phase1_output/ast")
        results = extract_ast_directory(
            target,
            output_dir,
            on_result=lambda path, document: report(path, document, output_dir / _batch_output_name(target, path)),
            **options,
        )
        print(
            f"\nParsed {len(results)} files: {counts['success']} clean, "
            f"{counts['partial']} with syntax errors, {counts['failed']} failed"
        )
    else:
        output_path = Path(args.output or f"{target.stem}_ast.json")
        report(target, extract_ast(target, output_path, **options), output_path)

    return 1 if counts["failed"] or counts["schema_errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
