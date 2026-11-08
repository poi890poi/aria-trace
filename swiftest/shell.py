"""A serialized text-command channel; a lost acknowledgement is never retried."""

import queue
import subprocess
import threading
import uuid


class ShellError(RuntimeError):
    pass


class Shell:
    def __init__(self, adb, serial):
        self.lock = threading.Lock()
        self.closed = False
        self.responses = queue.Queue()
        self.process = subprocess.Popen(
            [adb, "-s", serial, "shell", "-T", "sh"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.reader = threading.Thread(target=self._receive, daemon=True)
        self.reader.start()

    def _receive(self):
        try:
            while True:
                line = self.process.stdout.readline(32 * 1024 * 1024 + 1)
                if not line:
                    break
                self.responses.put(line)
        except OSError:
            pass
        finally:
            self.responses.put(None)

    def run(self, command, timeout=25):
        import time

        with self.lock:
            if self.closed:
                raise ShellError("Shell channel is closed; command was not sent.")
            marker = ("testkit_" + uuid.uuid4().hex + ":").encode()
            # A subshell isolates cwd, environment, exit and syntax errors. The
            # extra newline also delimits commands that do not print a newline.
            request = "(\n" + command + "\n)\nprintf '\\n" + marker.decode() + "%s\\n' \"$?\"\n"
            deadline = time.monotonic() + timeout
            output = bytearray()
            try:
                self.process.stdin.write(request.encode("utf-8"))
                self.process.stdin.flush()
                while True:
                    line = self.responses.get(timeout=max(0, deadline - time.monotonic()))
                    if line is None:
                        raise ShellError("Shell disconnected; command outcome is unknown.")
                    if line.startswith(marker):
                        status = int(line[len(marker):].strip())
                        result = bytes(output)
                        # Some Windows ADB builds expand every LF on stdout.
                        # The marker contains one known LF, so it detects this
                        # without guessing from the command's own text.
                        if line.endswith(b"\r\n"):
                            result = result.replace(b"\r\n", b"\n")
                        result = result[:-1]  # remove framing newline only
                        if status:
                            raise ShellError(result.decode("utf-8", "replace").strip()
                                             or f"Device command exited {status}")
                        return result
                    output.extend(line)
                    if len(output) > 32 * 1024 * 1024:
                        raise ShellError("Device command exceeded the text output limit.")
            except (OSError, ValueError, queue.Empty) as error:
                self.close()
                raise ShellError("Shell acknowledgement failed; command was not retried.") from error
            except ShellError:
                # Exit failures have a known outcome; nevertheless close the
                # channel so unread/late data can never belong to a later call.
                self.close()
                raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdin.close()
        self.reader.join(timeout=3)
        if not self.reader.is_alive():
            self.process.stdout.close()
