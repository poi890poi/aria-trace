"""Migrate a trusted user's Airtest files by inspecting syntax, without importing them."""

import ast
import builtins
import inspect
import io
from pathlib import Path
import re
import shutil
import tempfile
import tokenize
import json

from . import compat

MODULES = {
    "airtest.core.api": set(compat.__all__),
    "airtest.core.cv": {"Template"},
    "airtest.core.error": {"TargetNotFoundError"},
}


def convert_code(source):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    named = {}
    modules = {}
    wildcard = False
    problems = []

    def issue(n, reason):
        problems.append({"line": n.lineno, "reason": reason})

    # Replace only the module name on supported imports. Leave strings, comments and aliases intact.
    replacements = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module in MODULES and not n.level:
            names = MODULES[n.module]
            for a in n.names:
                if a.name == "*" and n.module == "airtest.core.api":
                    wildcard = True
                elif a.name not in names:
                    issue(n, "Unsupported import " + a.name)
                else:
                    named[a.asname or a.name] = a.name
            replacements.append((n, n.module))
        elif isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] in (
            "airtest",
            "poco",
        ):
            issue(n, "Unsupported module " + n.module)
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name in MODULES and a.asname and len(n.names) == 1:
                    modules[a.asname] = a.name
                    replacements.append((n, a.name))
                elif a.name.split(".")[0] in ("airtest", "poco"):
                    issue(n, "Unsupported import " + a.name)
    known = set(dir(builtins)) | set(named) | set(modules)
    device_variables = set()
    for n in ast.walk(tree):
        if (
            isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name)
        ):
            if named.get(n.value.func.id, n.value.func.id) in (
                "device",
                "connect_device",
            ):
                device_variables.update(
                    t.id for t in n.targets if isinstance(t, ast.Name)
                )
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            known.add(n.name)
        if isinstance(n, ast.arg):
            known.add(n.arg)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            known.add(n.id)
        if isinstance(n, ast.alias):
            known.add(n.asname or n.name.split(".")[0])
    for n in ast.walk(tree):
        if (
            wildcard
            and isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Load)
            and n.id not in known
            and n.id not in compat.__all__
            and n.id
            not in (
                "__file__",
                "__name__",
                "__package__",
                "__doc__",
                "__spec__",
                "__builtins__",
            )
        ):
            if not any(p["line"] == n.lineno and n.id in p["reason"] for p in problems):
                issue(n, "Unresolved symbol under wildcard import: " + n.id)
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        name = None
        if isinstance(f, ast.Name):
            name = named.get(f.id)
            if wildcard and f.id not in known:
                if f.id in compat.__all__:
                    name = f.id
                else:
                    issue(n, "Unresolved call under wildcard import: " + f.id)
            if f.id in ("exec", "eval", "__import__"):
                issue(n, "Dynamic Python requires review.")
        elif (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and f.value.id in modules
        ):
            if f.attr in MODULES[modules[f.value.id]]:
                name = f.attr
            else:
                issue(n, "Unsupported module member " + f.attr)
        elif (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and f.value.id in device_variables
        ):
            issue(n, "Direct device API requires review.")
        elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Call):
            receiver = f.value.func
            if isinstance(receiver, ast.Name) and named.get(
                receiver.id, receiver.id
            ) in ("device", "connect_device"):
                issue(n, "Direct device API requires review.")
        if name:
            func = getattr(compat, name, None)
            if any(k.arg is None for k in n.keywords) or any(
                isinstance(a, ast.Starred) for a in n.args
            ):
                issue(n, "Dynamic arguments to " + name)
            elif func and name != "TargetNotFoundError":
                try:
                    inspect.signature(func).bind(
                        *([None] * len(n.args)), **{k.arg: None for k in n.keywords}
                    )
                except TypeError as e:
                    issue(n, f"{name}: {e}")
    # Python AST columns are UTF-8 byte offsets. Replace from the right for semicolon-separated imports.
    for n, old in sorted(
        replacements,
        key=lambda item: (item[0].lineno, item[0].col_offset),
        reverse=True,
    ):
        index = n.lineno - 1
        prefix = lines[index].encode("utf8")[: n.col_offset].decode("utf8")
        rest = lines[index][len(prefix) :]
        match = re.match(r"(from|import)(\s+)" + re.escape(old) + r"\b", rest)
        if not match:
            issue(n, "Unusual import formatting requires review.")
            continue
        lines[index] = (
            prefix
            + rest[: match.start()]
            + match[1]
            + match[2]
            + "swiftest.compat"
            + rest[match.end() :]
        )
    return "".join(lines), problems


def convert(source, destination):
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Choose a new output directory.")
    root = source if source.is_dir() else source.parent
    if source.is_dir() and (destination == root or root in destination.parents):
        raise ValueError("Choose a destination outside the source directory.")
    if source.is_dir() and source.suffix != ".air":
        raise ValueError("Expected an .air directory or Python file.")
    files = sorted(root.rglob("*")) if source.is_dir() else [source]
    if not source.is_dir():
        # Only copy explicitly referenced adjacent image assets for single scripts.
        with tokenize.open(source) as stream:
            tree = ast.parse(stream.read())
        for n in ast.walk(tree):
            if (
                isinstance(n, ast.Constant)
                and isinstance(n.value, str)
                and Path(n.value).suffix.lower() in IMAGE_TYPES
            ):
                p = (root / n.value).resolve()
                if root in p.parents and p.is_file() and p not in files:
                    files.append(p)
    issues = []
    scripts = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        output = Path(tmp) / "project"
        output.mkdir()
        for path in files:
            if path.is_symlink() or "__pycache__" in path.parts or not path.is_file():
                continue
            rel = path.relative_to(root)
            target = output / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".py":
                raw = path.read_bytes()
                encoding = tokenize.detect_encoding(io.BytesIO(raw).readline)[0]
                content, warnings = convert_code(raw.decode(encoding))
                target.write_bytes(content.encode(encoding))
                scripts.append(str(rel))
                issues.extend(dict(file=str(rel), **warning) for warning in warnings)
            elif path.suffix.lower() in IMAGE_TYPES:
                shutil.copyfile(path, target)
            else:
                issues.append(
                    {
                        "file": str(rel),
                        "reason": "Non-image resource requires manual migration.",
                    }
                )
        if not scripts:
            raise ValueError("No Python scripts found.")
        report = {"scripts": scripts, "issues": issues}
        (output / "migration.json").write_text(
            json.dumps(report, indent=2), encoding="utf8"
        )
        output.rename(destination)
    return report


IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
