"""Validate persistent-channel framing and no retransmission on a real device."""

import argparse
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from swiftest.device import Android
from swiftest.shell import Shell, ShellError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    phone = Android(args.serial, persistent_shell=False)
    remote = "/data/local/tmp/testkit-once-" + uuid.uuid4().hex
    checks = []
    channel = Shell(phone.adb, phone.serial)
    try:
        assert channel.run("printf '中文\\n\\nlast'") == "中文\n\nlast".encode()
        assert channel.run("true") == b""
        assert channel.run("printf 'no newline'") == b"no newline"
        checks.append("exact framing, Unicode, blank lines and empty response")
        try:
            channel.run("exit 7")
        except ShellError as error:
            assert "exited 7" in str(error)
        else:
            raise AssertionError("Nonzero exit accepted")
        assert channel.closed
        checks.append("nonzero exit propagated")
        channel = Shell(phone.adb, phone.serial)
        channel.run("true")
        try:
            channel.run("printf 'once\\n' >> " + remote + "; sleep 2; printf late", timeout=.15)
        except ShellError:
            pass
        else:
            raise AssertionError("Timeout not detected")
        try:
            channel.run("printf 'twice\\n' >> " + remote)
        except ShellError:
            pass
        else:
            raise AssertionError("Closed channel sent another command")
        time.sleep(.3)
        assert phone.shell("cat", remote).splitlines() == ["once"]
        checks.append("timed-out command executed once; closed channel rejected subsequent command")
        args.output.write_text(json.dumps(dict(serial=args.serial, checks=checks), indent=2))
        print(json.dumps(checks), flush=True)
    finally:
        channel.close()
        phone.shell("rm", "-f", remote)
        phone.close()


if __name__ == "__main__":
    main()
