"""Print a compact aggregate report; keep raw telemetry in ignored storage."""
import argparse
import json
from pathlib import Path
import statistics


def metrics(root, result):
    task = result["task"]
    if result["implementation"] == "airtest":
        events = task.get("events", [])
        start = next((v["seconds"] for v in events if v.get("reason") == "minimap"), None)
        end = next((v["seconds"] for v in events if v["event"] == "arrival"), None)
        captures = task.get("captures")
        recognition = task.get("match_seconds")
    else:
        path = Path(task.get("evidence_directory", "")) / "events.jsonl"
        events = [json.loads(s) for s in path.read_text().splitlines()] if path.is_file() else []
        start = next((v["time"] for v in events if v.get("reason") == "minimap"), None)
        end = events[-1]["time"] if events and events[-1].get("state") == "world" else None
        captures = task.get("metrics", {}).get("captures")
        recognition = task.get("ocr_seconds")
    return dict(result, map_seconds=end-start if end is not None and start is not None else None,
                captures=captures, recognition_seconds=recognition)


def median_range(values, decimals=2):
    if not values:
        return "unavailable"
    return (f"{statistics.median(values):.{decimals}f} "
            f"({min(values):.{decimals}f}–{max(values):.{decimals}f})")


def report(root):
    results = [metrics(root, r) for r in json.loads((root / "results.json").read_text())]
    good = {name: [r for r in results if r["implementation"] == name and r["success"]]
            for name in ("testkit", "airtest")}
    text = ["# Genshin: SwifTest versus Airtest", "",
            "Measured November 7, 2026 on one Samsung SM-G781B (Android 13), USB-powered, "
            "fixed brightness 255/255, 2400×1080 landscape. Four scheduled runs per implementation. "
            "Each trial force-stopped the app, launched it, handled both entry prompts, and "
            "teleported to the same unlocked waypoint at the player's position.", "",
            "Timing includes a fresh Python process, imports, device setup, the task and cleanup. "
            "Helper installation, cooling and independent screenshots are outside timing. "
            "OS disk caches and installed helpers were warm.", "",
            "## Observed results", "",
            "Values are median (minimum–maximum), using successful trials only; failures are listed below.", "",
            "| Measurement | SwifTest | Airtest 1.4.3 |", "|---|---:|---:|"]
    for label, getter, decimals in [
        ("Full process, s", lambda r: r["process_seconds"], 2),
        ("Task, s", lambda r: r["task"]["task_seconds"], 2),
        ("Map tap to verified arrival, s", lambda r: r["map_seconds"], 2),
        ("Captured frames", lambda r: r["captures"], 0),
        ("Frame capture time, s", lambda r: r["task"].get("capture_seconds"), 2),
        ("OCR / template recognition time, s", lambda r: r["recognition_seconds"], 2),
    ]:
        cells = [median_range([getter(r) for r in good[n] if getter(r) is not None], decimals)
                 for n in good]
        text.append(f"| {label} | {' | '.join(cells)} |")
    for sensor, title in (("battery_c", "Battery"), ("AP_c", "Processor AP"), ("SKIN_c", "Skin")):
        for key, label in (("start", "start"), ("end", "end"), ("peak", "peak"), ("delta", "change")):
            cells = [median_range([r["temperature"][sensor][key] for r in good[n]
                                   if sensor in r["temperature"]], 1) for n in good]
            text.append(f"| {title} {label}, °C | {' | '.join(cells)} |")
    text += ["", "## Every trial", "",
             "| # | Implementation | Passed | Full process s | Map-to-arrival s | Battery start → end °C | AP start → end °C | Skin start → end °C | Cooling gate |",
             "|---:|---|---|---:|---:|---|---|---|---|"]
    for r in results:
        temperatures = [f"{r['temperature'][n]['start']:.1f} → {r['temperature'][n]['end']:.1f}"
                        if n in r["temperature"] else "missing" for n in ("battery_c", "AP_c", "SKIN_c")]
        map_time = f"{r['map_seconds']:.2f}" if r["map_seconds"] is not None else "—"
        text.append(f"| {r['index']} | {r['implementation']} | {r['success']} | "
                    f"{r['process_seconds']:.2f} | {map_time} | {' | '.join(temperatures)} | {r['rest_gate_met']} |")
    text += ["", "## Interpretation limits", "",
             "- These are different complete workflows. TestKit uses on-demand JPEG, local OCR and adaptive polling; "
             "Airtest uses Javacap, Maxtouch and bounded template matching. Results do not isolate framework overhead.",
             "- Temperature comes from Android battery and thermal HAL sensors, sampled every two seconds. "
             "Sensor lag, different starting temperatures, USB power and uncontrolled room temperature limit causal conclusions. "
             "This experiment does not measure energy in joules or establish a general thermal advantage.",
             "- This is a same-location teleport: a verified selected waypoint, one Teleport action, a loading transition, "
             "and two gameplay observations. It does not test travel to a different location.",
             "- The sample is four planned runs per tool on one phone and one UI language. No statistical-significance claim is made.",
             "- TestKit's current Chinese start screen was initially unsupported. A bounded start-label plus server-selector "
             "check was added and regression-tested before holdout collection. A development run then stalled when OCR "
             "omitted one character from the second entry prompt; it was stopped, and an exact alias requiring the release "
             "footer was added and tested. No core framework timings were tuned.",
             "- Airtest's default Minicap setup failed. Its working Javacap fallback was selected explicitly before holdout. "
             "The first Airtest development run timed out because the second entry prompt was missing; this was corrected "
             "before holdout. Development runs are retained separately and excluded from the measured table.", "",
             "## Reproduce and inspect", "",
             "See [protocol and commands](../benchmarks/android/README.md). "
             "The scripts are [Airtest](../examples/genshin/airtest_teleport.py) and "
             "[TestKit adapter](../benchmarks/android/testkit_teleport.py).", "",
             f"Raw local evidence: `{root.relative_to(Path.cwd()).as_posix()}/`. "
             "`metadata.json` records code/assets hashes, environment packages and settings; per-trial directories "
             "contain temperature samples, logs, state captures and independent ADB arrival screenshots. "
             "These generated files are ignored by Git.", "",
             "Primary references: [Airtest API](https://airtest.readthedocs.io/en/latest/all_module/airtest.core.api.html) "
             "and [Android battery telemetry](https://developer.android.com/reference/android/os/BatteryManager)."]
    return "\n".join(text) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = report(args.directory.resolve())
    if args.output:
        args.output.write_text(value, encoding="utf-8")
    else:
        print(value)
