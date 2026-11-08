"""Check saved independent evidence and every action/telemetry record."""
import argparse
import json
from pathlib import Path

import cv2


def audit(root):
    rows=json.loads((root/"results.json").read_text())
    findings=[]
    for row in rows:
        engine,mode=row["implementation"],row["mode"]
        directory=root/mode/f"{row['index']:02d}-{engine}"
        task=row["task"]
        assert row["success"] and row["exit_code"]==0
        inputs=task["inputs"]
        assert len(inputs)==(5 if mode=="cold" else 3)
        assert all(i["sent"] for i in inputs)
        assert [i["reason"] for i in inputs[-3:]]==["minimap","waypoint","teleport"]
        if mode=="cold":
            expected=(["entry","door"] if engine=="airtest" else
                      ["recognized game entry text"]*2)
            assert [i["reason"] for i in inputs[:2]]==expected
        if engine=="airtest":
            events=task["events"]
            teleport=next(e["seconds"] for e in events if e.get("reason")=="teleport")
            loading=next(e["seconds"] for e in events if e["event"]=="loading")
            arrival=next(e["seconds"] for e in events if e["event"]=="arrival")
            assert teleport<loading<arrival
        else:
            observations=task["observations"]
            loading=next(o["ended"] for o in observations if o["state"]=="loading")
            assert inputs[-1]["ended"]<loading<observations[-1]["ended"]
            assert [o["state"] for o in observations[-2:]]==["world","world"]
            assert task["metrics"]["capture_acquisition"]=="direct-bitmap"
            assert task["metrics"]["ocr"]["backend"]=="openvino"
        independent=cv2.imread(str(directory/"independent-arrival.png"))
        assert independent is not None and independent.shape==(1080,2400,3)
        evidence=Path(task["evidence_directory"])
        for name in ("before","map","selected","loading","arrival"):
            image=cv2.imread(str(evidence/(name+".png")))
            assert image is not None and image.shape==(1080,2400,3), str(evidence/name)
        samples=json.loads((directory/"temperatures.json").read_text())
        findings.append(dict(index=row["index"],engine=engine,mode=mode,
            inputs=len(inputs),evidence_images=6,rest_gate=row["rest_gate_met"],
            samples=len(samples),sample_errors=sum("error" in s for s in samples),
            missing_sensors=sum(not all(k in s for k in ("battery_c","AP_c","SKIN_c")) for s in samples),
            charging_states=sorted(set(s.get("status","missing") for s in samples)),
            usb_powered=sorted(set(s.get("USB powered","missing") for s in samples))))
    return findings


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    result=audit(args.directory)
    (args.directory/"audit.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
