"""Summarize every measured case without discarding failures or unmet rest gates."""
import argparse
import json
from pathlib import Path
import statistics

from steady_latency import summary


def aggregate(rows):
    groups={}
    engines=list(dict.fromkeys(r.get("variant",r["implementation"]) for r in rows))
    for mode in ("cold","warm"):
        for engine in engines:
            selected=[r for r in rows if r["mode"]==mode and r.get("variant",r["implementation"])==engine]
            if not selected:
                continue
            key=mode+"-"+engine
            group=dict(scheduled=len(selected),successes=sum(r["success"] for r in selected),
                       rest_gates=[r["rest_gate_met"] for r in selected])
            # A failed case remains visible; latency summaries describe successes only.
            good=[r for r in selected if r["success"]]
            measurements={"process":[r["process_seconds"] for r in good],
                "task":[r["task"]["task_seconds"] for r in good],
                "map_to_arrival":[r["task"]["map_to_arrival_seconds"] for r in good],
                "capture":[r["task"]["capture_seconds"] for r in good],
                "recognition":[r["task"]["recognition_seconds"] for r in good],
                "setup":[r["task"]["setup_seconds"] for r in good],
                "input":[i["capture_to_input_seconds"] for r in good for i in r["task"]["inputs"][-3:]],
                "dispatch":[i["dispatch_seconds"] for r in good for i in r["task"]["inputs"][-3:]]}
            group.update({name:summary([v*1000 for v in values]) for name,values in measurements.items() if values})
            if good:
                group["frames"]=[len(r["task"]["frames"]) for r in good]
                group["exact_repeats"]=[sum(f["repeated"] is True for f in r["task"]["frames"]) for r in good]
            groups[key]=group
    return groups


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    rows=json.loads((args.directory/"results.json").read_text())
    groups=aggregate(rows)
    (args.directory/"summary.json").write_text(json.dumps(groups,indent=2))
    engines=list(dict.fromkeys(r.get("variant",r["implementation"]) for r in rows))
    text=["| Condition / metric | "+" | ".join(engines)+" |",
          "|---|"+"---:|"*len(engines)]
    for mode in ("cold","warm"):
        for metric in ("process","task","map_to_arrival","capture","recognition","input","dispatch"):
            values=[]
            for engine in engines:
                value=groups.get(mode+"-"+engine,{}).get(metric)
                values.append(f"{value['median_ms']/1000:.3f}" if value else "unavailable")
            text.append(f"| {mode} {metric}, s | {' | '.join(values)} |")
    print("\n".join(text))


if __name__=="__main__":
    main()
