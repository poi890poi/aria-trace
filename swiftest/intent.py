"""Data-only Android tests. The same selectors and session policies power the Python API."""

from .targets import Absent, AnyOf, Dialog, Node, Pixel, Position, Template, Text, Variants
from .textmatch import TextPattern


def target(spec, lookup=None, base=None):
    if isinstance(spec, str) and lookup:
        return lookup(spec)
    if not isinstance(spec, dict) or len(spec) != 1:
        raise ValueError("A selector requires exactly one selector type.")
    name, opts = next(iter(spec.items()))
    if name == "position":
        return Position(*opts)
    if name == "pixel":
        return Pixel(*opts)
    if name == "node":
        values = dict(opts)
        for key in ("resource_id", "text", "description", "class_name"):
            if isinstance(values.get(key), dict):
                values[key] = TextPattern(**values[key])
        return Node(**values)
    if name == "text":
        return Text(opts) if isinstance(opts, (str, list)) else Text(**opts)
    if name == "template":
        values = {"filename": opts} if isinstance(opts, str) else dict(opts)
        if base:
            for key in ("filename", "mask"):
                if values.get(key):
                    values[key] = str((base / values[key]).resolve())
        return Template(**values)
    if name == "absent":
        return Absent(target(opts, lookup, base))
    if name == "any_of":
        return AnyOf(*(target(v, lookup, base) for v in opts))
    if name == "variants":
        return Variants(*(target(v, lookup, base) for v in opts))
    if name == "dialog":
        return Dialog(
            target(opts["parent"], lookup, base), target(opts["child"], lookup, base)
        )
    raise ValueError("Unknown selector: " + name)


def load_catalog(session, filename):
    import json
    from pathlib import Path

    path = Path(filename).resolve()
    catalog = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(catalog, dict) or set(catalog) - {"elements", "busy", "popups"}:
        raise ValueError("A UI catalog accepts elements, busy states and popups.")
    definitions = catalog.get("elements", {})
    active = set()

    def resolve(name):
        if name in session.targets:
            return session.targets[name]
        if name in active or name not in definitions:
            raise ValueError("Unknown or cyclic UI target: " + name)
        active.add(name)
        try:
            result = target(definitions[name], resolve, path.parent)
            session.register(name, result)
            return result
        finally:
            active.remove(name)

    for name in definitions:
        resolve(name)
    for state in catalog.get("busy", []):
        session.busy_while(
            target(state["while"], resolve, path.parent), state["allowance"]
        )
    for popup in catalog.get("popups", []):
        if set(popup) - {"name", "when", "dismiss", "max_occurrences", "timeout"}:
            raise ValueError("Unknown pop-up policy field.")
        session.on_popup(popup["name"], target(popup["when"], resolve, path.parent),
                         target(popup["dismiss"], resolve, path.parent),
                         **{k: popup[k] for k in ("max_occurrences", "timeout") if k in popup})


def execute(session, plan):
    def select(spec):
        return target(spec, session.resolve, session.base)

    if (
        not isinstance(plan, dict)
        or set(plan) - {"app", "activity", "steps"}
        or not isinstance(plan.get("steps"), list)
    ):
        error = ValueError("An intent accepts app, activity and a list of steps.")
        session.record_failure("validate plan", error)
        raise error
    # Validate every step before launching the app or dispatching the first input.
    calls = []
    for index, original in enumerate(plan["steps"], 1):
        label = f"step {index}"
        try:
            if not isinstance(original, dict):
                raise ValueError("Each step must be an object.")
            step = dict(original)
            label = step.pop("intent", label)
            if not isinstance(label, str) or not label.strip():
                raise ValueError("An intent label must be a nonempty string.")
            if "touch" in step and not set(step) - {
                "touch", "expect", "offset", "offset_unit", "required",
            }:
                opts = {k: v for k, v in step.items() if k not in ("touch", "expect")}
                if "expect" in step:
                    opts["expect"] = select(step["expect"])
                call = (session.touch, [select(step["touch"])], opts)
            elif len(step) == 1:
                op, value = next(iter(step.items()))
                if op in ("assert", "assert_absent"):
                    method = session.assert_exists if op == "assert" else session.assert_not_exists
                    call = (method, [select(value)], {})
                elif op == "wait_any" and isinstance(value, dict) and value:
                    if any(not isinstance(k, str) or not k for k in value):
                        raise ValueError("Condition names must be nonempty strings.")
                    call = (session.wait_any, [{k: select(v) for k, v in value.items()}], {})
                elif op == "swipe" and isinstance(value, list) and len(value) == 2:
                    call = (session.swipe, [select(v) for v in value], {})
                elif op in ("key", "text"):
                    call = (getattr(session, op), [value], {})
                else:
                    raise ValueError("Unknown intent operation: " + op)
            else:
                raise ValueError("Unexpected intent fields.")
            details = dict(step)
            if "text" in details:
                details["text"] = "<text omitted>"
            calls.append((label, index, details, call))
        except Exception as exc:
            session.record_failure(label, exc, step=index, phase="validation")
            raise
    if "app" in plan:
        session.open(plan["app"], plan.get("activity"))
    result = None
    for label, index, details, (method, args, options) in calls:
        with session.intent(label, step=index, operation=details):
            result = method(*args, **options)
    return result
