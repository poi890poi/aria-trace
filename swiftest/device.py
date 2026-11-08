"""One explicit Android device. Host programs are always invoked by argument list."""

from pathlib import Path
import re
import shlex
import shutil
import subprocess
import uuid
import xml.etree.ElementTree as ET

from .shell import Shell, ShellError


class DeviceError(RuntimeError):
    pass


def run(args, timeout=25):
    try:
        proc = subprocess.run(args, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DeviceError(str(exc)) from exc
    if proc.returncode:
        raise DeviceError(
            proc.stderr.decode("utf-8", "replace").strip()
            or f"{args[0]} exited {proc.returncode}"
        )
    return proc.stdout


def devices(adb="adb"):
    lines = run([adb, "devices"]).decode().splitlines()
    return [
        line.split()[0]
        for line in lines[1:]
        if len(line.split()) >= 2 and line.split()[1] == "device"
    ]


class Android:
    def __init__(self, serial=None, adb="adb", *, persistent_shell=True):
        self.adb = shutil.which(adb) or str(adb)
        available = devices(self.adb)
        if serial is None:
            if len(available) != 1:
                raise DeviceError(
                    f"Specify a serial; {len(available)} authorized devices are connected."
                )
            serial = available[0]
        if serial not in available:
            raise DeviceError(f"Device {serial!r} is unavailable.")
        self.serial = serial
        self.package = None
        self.xml_path = "/data/local/tmp/testkit-" + uuid.uuid4().hex + ".xml"
        self.used_tree = False
        self._unicode_input = None
        self._persistent_shell = persistent_shell
        self._shell = None

    def run(self, *args, timeout=25):
        if self._persistent_shell and len(args) == 2 and args[0] == "shell":
            try:
                if self._shell is None:
                    self._shell = Shell(self.adb, self.serial)
                return self._shell.run(args[1], timeout)
            except (OSError, ShellError) as exc:
                raise DeviceError(str(exc)) from exc
        return run([self.adb, "-s", self.serial, *args], timeout)

    def shell(self, *args):
        command = " ".join(shlex.quote(str(arg)) for arg in args)
        return self.run("shell", command).decode("utf-8", "replace")

    def foreground(self):
        # Filter on the device so a focus check transfers one line.
        line = self.run("shell", "dumpsys window | grep mCurrentFocus").decode(
            "utf-8", "replace"
        )
        match = re.search(r"\s([\w.]+)/[^\s}]+", line)
        return match.group(1) if match else None

    def launch(self, package, activity=None):
        if not re.fullmatch(r"[\w.]+", package):
            raise ValueError("Invalid Android package.")
        self.package = package
        self.shell("input", "keyevent", "KEYCODE_WAKEUP")
        if activity is None and self.foreground() == package:
            return
        if activity is None:
            resolved = self.shell(
                "cmd",
                "package",
                "resolve-activity",
                "--brief",
                "-a",
                "android.intent.action.MAIN",
                "-c",
                "android.intent.category.LAUNCHER",
                package,
            )
            activity = next(
                (
                    s.strip()
                    for s in resolved.splitlines()
                    if s.startswith(package + "/")
                ),
                None,
            )
            if not activity:
                raise DeviceError(
                    "No launcher activity for "
                    + package
                    + "; specify activity explicitly."
                )
        component = activity if "/" in activity else package + "/" + activity
        if not component.startswith(package + "/"):
            raise ValueError("Activity must belong to the requested package.")
        result = self.shell("am", "start", "-W", "-n", component)
        if "Error" in result or "Exception" in result:
            raise DeviceError(result.strip())

    def check_focus(self):
        if self.package and self.foreground() != self.package:
            raise DeviceError("Focus changed before input to " + self.package)

    def display_size(self):
        line = self.run(
            "shell", 'dumpsys input | grep "Viewport INTERNAL" | head -1'
        ).decode()
        match = re.search(r"logicalFrame=\[0, 0, (\d+), (\d+)\]", line)
        return tuple(map(int, match.groups())) if match else None

    def tap(self, x, y, duration=0):
        self.check_focus()
        x, y = str(round(x)), str(round(y))
        if duration:
            self.shell("input", "swipe", x, y, x, y, str(round(duration * 1000)))
        else:
            self.shell("input", "tap", x, y)

    def swipe(self, a, b, duration):
        self.check_focus()
        self.shell(
            "input",
            "swipe",
            *(str(round(n)) for n in (*a, *b)),
            str(round(duration * 1000)),
        )

    def key(self, key):
        self.check_focus()
        self.shell("input", "keyevent", str(key))

    def install(self, apk, replace=False):
        path = Path(apk).resolve(strict=True)
        return (
            self.run("install", *(["-r"] if replace else []), str(path), timeout=180)
            .decode()
            .strip()
        )

    def uninstall(self, package):
        return self.run("uninstall", package, timeout=60).decode().strip()

    def clear(self, package):
        result = self.shell("pm", "clear", package).strip()
        if result != "Success":
            raise DeviceError(result)

    def push(self, local, remote):
        return self.run(
            "push", str(Path(local).resolve(strict=True)), str(remote), timeout=180
        )

    def pull(self, remote, local):
        return self.run("pull", str(remote), str(local), timeout=180)

    def text(self, value):
        if not isinstance(value, str):
            raise TypeError("Text input requires a string.")
        if not value:
            return
        self.check_focus()
        if value.isascii() and value.isprintable() and "%s" not in value:
            self.shell("input", "text", value.replace(" ", "%s"))
            return
        if self._unicode_input is None:
            try:
                import uiautomator2
            except ImportError as exc:
                raise DeviceError(
                    "Install swiftest[unicode] for Unicode text input."
                ) from exc
            self._unicode_input = uiautomator2.connect(self.serial)
        self._unicode_input.send_keys(value)

    def hierarchy(self):
        self.used_tree = True
        self.shell("rm", "-f", self.xml_path)
        message = self.shell("uiautomator", "dump", "--compressed", self.xml_path)
        if "ERROR" in message.upper():
            raise DeviceError(message.strip())
        try:
            return ET.fromstring(self.run("exec-out", "cat", self.xml_path))
        except ET.ParseError as exc:
            raise DeviceError("Invalid Android UI hierarchy.") from exc

    def close(self):
        try:
            if self.used_tree:
                if self._shell and self._shell.closed:
                    # Cleanup is a separate idempotent command, never a replay
                    # of an input whose acknowledgement was lost.
                    run([self.adb, "-s", self.serial, "shell",
                         "rm -f " + shlex.quote(self.xml_path)])
                else:
                    self.shell("rm", "-f", self.xml_path)
        finally:
            if self._shell:
                self._shell.close()
