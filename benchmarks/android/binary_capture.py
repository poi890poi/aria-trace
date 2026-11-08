"""Experimental binary transport, preserving production capture/JPEG/repeat policy."""
import hashlib
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from swiftest.capture import Capture


def build():
    source = (ROOT/"swiftest/assets/IntentScreenshot.java").read_text(encoding="utf-8")
    source = source.replace("DataOutputStream output = new DataOutputStream(System.out);", """
        android.net.LocalServerSocket server = new android.net.LocalServerSocket(args[0]);
        Thread lifetime = new Thread(() -> {
            try { while (System.in.read() != -1) {} } catch (Exception ignored) {}
            System.exit(0);
        });
        lifetime.setDaemon(true);
        lifetime.start();
        System.out.println("READY");
        System.out.flush();
        android.net.LocalSocket client = server.accept();
        java.io.InputStream input = client.getInputStream();
        DataOutputStream output = new DataOutputStream(client.getOutputStream());
    """)
    source = source.replace("(request = System.in.read())", "(request = input.read())")
    old = '''            if (request != 'C') output.writeBytes(camera.usedDirect ? "D:" : "S:");
            // adb shell on Windows translates line endings. Keep the wire alphabet text-safe.
            output.write(android.util.Base64.encode(image, android.util.Base64.NO_WRAP));
            output.writeByte(10);'''
    assert old in source
    source = source.replace(old, '''            output.writeInt(0x41544B31);
            output.writeByte(camera.usedDirect ? 'D' : 'S');
            output.writeInt(image.length);
            output.write(image);''')
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    cache = ROOT/"artifacts/android-binary-capture/build"/digest
    cache.mkdir(parents=True, exist_ok=True)
    jar = cache/"snapshot.jar"
    if jar.exists():
        return jar
    java = cache/"IntentScreenshot.java"
    java.write_text(source, encoding="utf-8")
    homes = [os.environ.get("ANDROID_HOME"), os.environ.get("ANDROID_SDK_ROOT"),
             Path.home()/"AppData/Local/Android/Sdk", Path(os.environ["LOCALAPPDATA"])/"Android/Sdk"]
    sdk = next(Path(p) for p in homes if p and (Path(p)/"platforms").is_dir())
    platform = sorted(sdk.glob("platforms/android-*/android.jar"), reverse=True)[0]
    compiler = sorted(sdk.glob("build-tools/*/lib/d8.jar"), reverse=True)[0]
    for argv in (["javac", "-source", "8", "-target", "8", "-classpath", str(platform),
                  "-d", str(cache), str(java)],
                 ["java", "-cp", str(compiler), "com.android.tools.r8.D8", "--min-api", "26",
                  "--lib", str(platform), "--output", str(jar), str(cache/"IntentScreenshot.class")]):
        subprocess.run(argv, check=True, capture_output=True, timeout=90)
    return jar


class BinaryCapture(Capture):
    def __init__(self, phone, mode="auto"):
        super().__init__(phone, mode="png")
        self.connection = self.forward = None
        self.prefer_direct = True
        self.prefer_bitmap = True
        try:
            jar = build()
            token = "atk-binary-" + uuid.uuid4().hex
            self.remote = "/data/local/tmp/" + token + ".jar"
            self.phone.run("push", str(jar), self.remote)
            self.setup_bytes = jar.stat().st_size
            self.process = subprocess.Popen([phone.adb,"-s",phone.serial,"shell","-T",
                "env","CLASSPATH="+self.remote,"app_process","/system/bin","IntentScreenshot",token],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
            # Bound readiness independently of the stream, including helper startup failures.
            import queue
            import threading
            ready = queue.Queue()
            threading.Thread(target=lambda: ready.put(self.process.stdout.readline()),daemon=True).start()
            if ready.get(timeout=15).strip() != b"READY":
                raise RuntimeError("Binary helper failed to start")
            port = int(phone.run("forward","tcp:0","localabstract:"+token).strip())
            self.forward = "tcp:"+str(port)
            self.connection = socket.create_connection(("127.0.0.1",port),timeout=15)
            self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            self.mode = "jpeg"
        except Exception as error:
            self.reason = repr(error)
            self.close()

    def _read_exact(self, size):
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self.connection.recv(size-len(chunks))
            if not chunk:
                raise RuntimeError("Truncated binary frame")
            chunks.extend(chunk)
        return bytes(chunks)

    def read(self):
        self.repeat = False
        if self.mode == "jpeg":
            try:
                self.connection.sendall(b"R")
                magic, acquisition, length = struct.unpack("!4scI",self._read_exact(9))
                if magic != b"ATK1" or acquisition not in (b"D",b"S") or length > 32*1024*1024:
                    raise RuntimeError("Invalid binary frame header")
                data = self._read_exact(length)
                self.bytes += 9+length
                self.acquisition = "direct-bitmap" if acquisition==b"D" else "screencap"
                if not data and self.previous is not None:
                    self.repeat = True
                    return self.previous
                if not (data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")):
                    raise RuntimeError("Invalid binary JPEG")
                self.previous = data
                return data
            except (OSError,RuntimeError,ValueError) as error:
                self.reason = repr(error)
                self.mode = "png"
        return super().read()

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None
        if self.process:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process.stdout.close()
            self.process = None
        if self.forward:
            self.phone.run("forward","--remove",self.forward)
            self.forward = None
        if self.remote:
            self.phone.shell("rm","-f",self.remote)
            self.remote = None
