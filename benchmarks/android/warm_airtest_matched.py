"""Compare fully initialized TestKit and Airtest with original/100-ms polling."""
import argparse
import hashlib
from pathlib import Path
import re
import sys

from compare_genshin import ROOT, PACKAGE, command, run_trial, write_json
from full_comparison import source_hashes


def worker():
    sys.argv.remove("--worker")
    sys.path.insert(0,str(ROOT))
    engine=sys.argv[sys.argv.index("--engine")+1]
    if engine=="airtest":
        from examples.genshin.airtest_teleport import Teleport
        initialize=Teleport.__init__
        def ready(self,*args,**kwargs):
            initialize(self,*args,**kwargs)
            assert self.device.touch_proxy.method_name=="MAXTOUCH"
        Teleport.__init__=ready
        if "--poll100" in sys.argv:
            sys.argv.remove("--poll100")
            original=Teleport.wait_for
            def wait_for(self,predicate,name,interval=.5):
                return original(self,predicate,name,interval=.1)
            Teleport.wait_for=wait_for
    else:
        from examples.genshin.profile import Phone
        initialize=Phone.__init__
        def ready(self,session):
            initialize(self,session)
            assert session.device.foreground()==PACKAGE
        Phone.__init__=ready
    from full_comparison_trial import main as trial
    trial()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--adb",default="adb")
    parser.add_argument("--airtest-python",type=Path,default=ROOT/".venv/airtest-benchmark/Scripts/python.exe")
    parser.add_argument("--testkit-python",type=Path,default=ROOT/".venv/androidtestkit/Scripts/python.exe")
    parser.add_argument("--assets",type=Path,default=ROOT/"artifacts/airtest-benchmark/assets")
    parser.add_argument("--rest",type=float,default=60)
    parser.add_argument("--max-rest",type=float,default=300)
    args=parser.parse_args()
    for key in ("output","assets","airtest_python","testkit_python"):
        setattr(args,key,getattr(args,key).resolve())
    root=args.output
    root.mkdir(parents=True,exist_ok=False)
    (root/"warmup").mkdir()
    (root/"warm").mkdir()
    adb=[args.adb,"-s",args.serial]
    focus=command(adb+["shell","dumpsys","window"]).decode("utf-8","replace")
    prior=re.search(r"mCurrentFocus=.*?\s([\w.]+/[\w.$]+)",focus)
    prior=prior[1] if prior else None
    def hashes():
        return dict(source_hashes(args.assets),**{str(Path(__file__).relative_to(ROOT)):
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    frozen=hashes()
    order="airtest,testkit,airtest100,airtest100,testkit,airtest,testkit,airtest,airtest100,airtest100,airtest,testkit".split(",")
    write_json(root/"metadata.json",dict(order=order,hashes=frozen,airtest_poll=.1,
        prior_activity=prior,serial=args.serial,one_variable="Airtest wait interval only; every case has initialized input transport",
        rest=args.rest,max_rest=args.max_rest))
    base=Path(__file__).with_name("full_comparison_trial.py")
    args.airtest_script=args.testkit_script=base
    results=[]
    try:
        args.output=root/"warmup"
        args.airtest_options=["--engine","airtest","--mode","cold"]
        preflight=run_trial(args,0,"airtest")
        write_json(root/"excluded-warmup.json",preflight)
        if not preflight["success"]:
            raise RuntimeError("Excluded warmup failed")
        args.output=root/"warm"
        args.skip_cooldown=True
        args.airtest_script=args.testkit_script=Path(__file__)
        args.testkit_options=["--worker","--engine","testkit","--mode","warm"]
        for index,case in enumerate(order,1):
            engine="airtest" if case.startswith("airtest") else "testkit"
            args.airtest_options=["--worker","--engine","airtest","--mode","warm"]
            if case=="airtest100":
                args.airtest_options.append("--poll100")
            row=run_trial(args,index,engine)
            row.update(mode="warm",variant=case)
            results.append(row)
            write_json(root/"results.json",results)
            assert hashes()==frozen,"Source changed during cohort"
            if not row["success"]:
                raise RuntimeError("Cohort stopped on failure; evidence retained")
        write_json(root/"complete.json",dict(tasks=len(results),source_unchanged=True))
    finally:
        command(adb+["shell","am","force-stop",PACKAGE])
        if prior:
            command(adb+["shell","input","keyevent","224"])
            command(adb+["shell","am","start","-W","-n",prior])


if __name__=="__main__":
    worker() if "--worker" in sys.argv else main()
