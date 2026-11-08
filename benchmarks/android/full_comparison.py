"""Balanced cold-process plus warm-workflow comparison against pinned Airtest."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import sys

from compare_genshin import ROOT, PACKAGE, command, run_trial, write_json


def source_hashes(assets):
    paths = list((ROOT/"swiftest").rglob("*.py"))
    paths += list((ROOT/"swiftest/assets").glob("*.java"))
    paths += [ROOT/"examples/genshin/profile.py", ROOT/"examples/genshin/airtest_teleport.py",
              Path(__file__), Path(__file__).with_name("full_comparison_trial.py"),
              Path(__file__).with_name("compare_genshin.py")]
    paths += list(assets.glob("*"))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--airtest-python",type=Path,default=ROOT/".venv/airtest-benchmark/Scripts/python.exe")
    parser.add_argument("--testkit-python",type=Path,default=ROOT/".venv/androidtestkit/Scripts/python.exe")
    parser.add_argument("--assets",type=Path,default=ROOT/"artifacts/airtest-benchmark/assets")
    parser.add_argument("--adb",default="adb")
    parser.add_argument("--rest",type=float,default=60)
    parser.add_argument("--max-rest",type=float,default=300)
    parser.add_argument("--order",default="airtest,testkit,testkit,airtest,testkit,airtest,airtest,testkit")
    args=parser.parse_args()
    order=args.order.split(",")
    if any(tool not in ("airtest","testkit") for tool in order):
        raise ValueError("Unknown implementation")
    for key in ("assets","output","airtest_python","testkit_python"):
        setattr(args,key,getattr(args,key).resolve())
    root=args.output
    root.mkdir(parents=True,exist_ok=False)
    for mode in ("cold","warm"):
        (root/mode).mkdir()
    adb=[args.adb,"-s",args.serial]
    focus=command(adb+["shell","dumpsys","window"]).decode("utf-8","replace")
    prior=re.search(r"mCurrentFocus=.*?\s([\w.]+/[\w.$]+)",focus)
    prior=prior[1] if prior else None
    hashes=source_hashes(args.assets)
    metadata=dict(order=order,hashes=hashes,rest=args.rest,max_rest=args.max_rest,
        serial=args.serial,host=platform.platform(),prior_activity=prior,
        testkit="workspace source; not installed wheel",airtest="installed pinned 1.4.3",
        cold="fresh process + force-stopped app",warm="new process; setup and verified-world warmup excluded")
    for label,argv in {"model":["getprop","ro.product.model"],
        "android":["getprop","ro.build.version.release"],
        "brightness":["settings","get","system","screen_brightness"],
        "brightness_mode":["settings","get","system","screen_brightness_mode"],
        "size":["wm","size"]}.items():
        metadata[label]=command(adb+["shell",*argv]).decode().strip()
    for name,python in (("airtest",args.airtest_python),("testkit",args.testkit_python)):
        metadata[name+"_packages"]=command([python,"-m","pip","freeze"]).decode().splitlines()
    write_json(root/"metadata.json",metadata)
    args.airtest_script=args.testkit_script=Path(__file__).with_name("full_comparison_trial.py")
    results=[]
    try:
        for index,engine in enumerate(order,1):
            for mode in ("cold","warm"):
                args.output=root/mode
                args.skip_cooldown=mode=="warm"
                args.airtest_options=["--engine","airtest","--mode",mode]
                args.testkit_options=["--engine","testkit","--mode",mode]
                row=run_trial(args,index,engine)
                row["mode"]=mode
                results.append(row)
                write_json(root/"results.json",results)
                assert hashes==source_hashes(args.assets), "Source changed during cohort"
                if not row["success"]:
                    raise RuntimeError("Cohort stopped on failed trial; evidence retained")
        write_json(root/"complete.json",dict(tasks=len(results),source_unchanged=True))
    finally:
        command(adb+["shell","am","force-stop",PACKAGE])
        if prior:
            command(adb+["shell","input","keyevent","224"])
            command(adb+["shell","am","start","-W","-n",prior])


if __name__=="__main__":
    main()
