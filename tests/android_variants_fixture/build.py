"""Build this small Java fixture with a local Android SDK; all output is ignored."""
import argparse
from pathlib import Path
import shutil
import subprocess
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/androidtestkit/variants-fixture"))
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent
    android = sorted(args.sdk.glob("platforms/android-*/android.jar"))[-1]
    build = sorted(args.sdk.glob("build-tools/*/lib/d8.jar"))[-1].parent.parent
    suffix = ".exe" if (build/"aapt2.exe").exists() else ""

    def run(*argv):
        subprocess.run([str(v) for v in argv], check=True, timeout=120,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    classes = out/"classes"
    classes.mkdir(exist_ok=True)
    run("javac", "-encoding", "UTF-8", "-source", "8", "-target", "8", "-classpath", android,
        "-d", classes, source/"MainActivity.java")
    run("java", "-cp", build/"lib/d8.jar", "com.android.tools.r8.D8", "--min-api", "26",
        "--lib", android, "--output", out, *classes.rglob("*.class"))
    run(build/("aapt2"+suffix), "link", "-I", android, "--manifest", source/"AndroidManifest.xml",
        "-o", out/"unsigned.apk")
    with zipfile.ZipFile(out/"unsigned.apk", "a", zipfile.ZIP_DEFLATED) as apk:
        apk.write(out/"classes.dex", "classes.dex")
    run(build/("zipalign"+suffix), "-f", "4", out/"unsigned.apk", out/"aligned.apk")
    if not (out/"fixture.keystore").exists():
        run("keytool", "-genkeypair", "-keystore", out/"fixture.keystore", "-alias", "fixture",
            "-storepass", "android", "-keypass", "android", "-keyalg", "RSA",
            "-dname", "CN=Local TestKit Fixture", "-validity", "30")
    run("java", "-jar", build/"lib/apksigner.jar", "sign", "--ks", out/"fixture.keystore",
        "--ks-pass", "pass:android", "--out", out/"fixture.apk", out/"aligned.apk")
    print(out/"fixture.apk")


if __name__ == "__main__":
    main()
