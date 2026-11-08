# Steady-state Android latency: remove repeated work

**The source checkout now reuses an ADB shell connection and calculates the
Genshin brightness percentile with a histogram. In the final fixture cohort,
guarded input fell from 547 to 348 ms median (36.3%). Fresh-image preprocessing
fell from 17.1 to 4.4 ms (74.0%). A separate direct-capture prototype reached
85 ms versus 219 ms, but remains experimental.**

Measured November 7, 2026, Windows/Python 3.12.10, Samsung SM-G781B, Android 13,
USB ADB, 2400×1080/1080×2400. No inference model, OCR threshold, polling policy,
focus check, coordinate check or input retry policy was relaxed. The existing
installed wheel remains 0.4.0; these are uncommitted source changes, not a new
installed release. Generated evidence remains ignored under
`artifacts/android-latency/`.

## First-principles analysis

A decision needs a current observation, enough computation to identify the
control, current input guards, and one dispatched action. Reopening a host/device
connection does not contribute new information. Likewise, an 8-bit percentile
needs counts of 256 values, not a partition of millions of image samples.

The loop consists of acquisition, preprocessing, recognition, input guards and
dispatch, plus any deliberate wait. Improving one stage cannot remove another.
The previous ~1.30-second game capture-start-to-tap figure is not remeasured here;
adding medians from different experiments would not establish a new loop time.

The initial read-only paired probe isolated connection cost: an ADB no-op took
67.8 ms median with a new process/connection and 29.1 ms through the persistent
channel. Keeping the same commands reduced focus and display checks by roughly
40 ms each. The fixture input path retains both focus checks, the overlay query,
display-size validation and the same `input tap` command.

## Final measurements

Warm samples exclude setup, first use, app launch, native UI grading and cleanup.
Input uses process/persistent/persistent/process blocks within each layout;
capture and preprocessing reverse method order on alternating passes.

| Component | Baseline median / p95 | Candidate median / p95 | Samples per method | Decision |
|---|---:|---:|---:|---|
| Guarded input in native fixture | 547.3 / 584.5 ms | 348.4 / 393.8 ms | 36 | Applied in source |
| Brightness preparation, fresh images | 17.10 / 18.64 ms | 4.44 / 6.06 ms | 72 | Applied in Genshin profile |
| Full capture, JPEG transfer and decode | 218.9 / 230.3 ms | 85.0 / 95.4 ms | 18 | Retain Android 13 prototype |

Input mean/max was 549.5/598.5 ms versus 348.4/447.0 ms. All three language/layout
variants passed. Native counters accounted for 84 taps including warmups; 12
additional focus-mismatch attempts were rejected without changing those counters.
The final cohort used landscape layouts; the preceding validation cohort also
exercised portrait layouts. It reproduced the input gain (560 to 328 ms median),
but was not the final frozen-source cohort.

Preprocessing development covered ten saved game/fixture images plus dimmed
versions (240 samples per method, 18.4 to 4.4 ms median). Nine existing adversarial
images reproduced the gain (14.3 to 4.5 ms). The final three native screenshots
were acquired after selecting the method; their original and dimmed versions
produced identical output pixels in every comparison. The implementation retains
the original percentile for other dtypes and images larger than 2^24 samples.

The capture experiment changes acquisition only: persistent Android 13
`SurfaceControl.captureDisplay` replaces the per-request `screencap` subprocess
and raw pipe. Both paths retain full size, JPEG quality 80, Base64 transport and
host decoding. Final requests force encoding, so repeated-frame suppression
cannot explain the gain. All three saved method pairs had identical decoded
pixels and dimensions. Earlier portrait and landscape pairs also matched.
This does not establish capture behavior on moving game scenes, secure surfaces,
multiple displays, other Android releases or other phones.

## Implementation and failure evidence

`androidtestkit/shell.py` serializes text commands, delimits responses and checks
exit status. Each command runs in a subshell, preserving the old isolation of
working directory and environment. A disconnect, missing acknowledgement or
command failure closes the channel; no input is automatically resent. Binary
capture and file transfer still use their original ADB paths. The original
transport remains selectable with `Android(serial, persistent_shell=False)`.

A real-device check caught Windows ADB expanding LF to CRLF. The known delimiter
now detects that conversion and removes it without losing literal CRLF in the
command's original output. Regression tests cover both transport formats. Device
checks passed for Unicode, empty/multiline output, nonzero exit and a forced
timeout: the disposable counter recorded exactly one execution, and the closed
channel rejected the following command. Successful text-command diagnostics
share the response stream; callers requiring the original stream behavior can
select the original transport.

Other rejected/development paths: `dumpsys window windows` did not expose the
required focus field on this phone. `dumpsys window displays` was faster, but is
not adopted without wider format/compatibility evidence. The phone's `input`
script already delegates to `cmd input`; changing that spelling is not elimination
of a Java startup. Fixture harness preflight needed an absent-package check and
the existing case-normalized text matcher for Android's uppercase Close label.
These failures occurred before the final cohort. No latency threshold or OCR
alias was tuned on the final data.

82 focused Android, Genshin and OCR-policy tests pass. The final input harness
checked runtime/fixture source hashes before and after its cohort. The disposable
fixture and remote helpers were removed, and the previous app was restored.
No new full-game completion, energy, thermal, human-speed or sub-500-ms loop claim
follows from these component experiments.

## Sources and reproduction

Primary sources checked November 7, 2026:
[scrcpy developer architecture](https://github.com/Genymobile/scrcpy/blob/master/doc/develop.md)
demonstrates persistent control/video connections and warns that hidden Android
methods differ across versions. The
[Android 13 SurfaceControl implementation](https://raw.githubusercontent.com/aosp-mirror/platform_frameworks_base/android13-release/core/java/android/view/SurfaceControl.java)
defines capture buffers and rotation behavior; copying a hardware buffer to a
software bitmap remains real work. These mechanisms informed the experiments;
their upstream performance is not a local result. Recent OCR comparisons in
[the solution evaluation](android-ocr-solutions-2026-11-07.md) remain applicable;
no new model or approximate cross-frame reuse was selected.

Raw provenance: `baseline/`, `commands-development-2/`, `fixture-development-3/`,
`fixture-holdout/`, `final-holdout/`, `preprocess-development/`,
`preprocess-holdout/`, `preprocess-fresh/`, and `shell-contract.json`, all below
`artifacts/android-latency/`. Each completed benchmark saves per-sample values and
source/input hashes. `final-holdout/capture-summary.json` contains the pooled
capture statistics and pixel comparisons. Earlier preflights are not final data.

```text
python benchmarks/android/steady_latency.py --serial SERIAL --output artifacts/android-latency/new-commands
python tests/android_variants_fixture/build.py --sdk ANDROID_SDK_PATH
python benchmarks/android/latency_fixture.py --serial SERIAL --output artifacts/android-latency/new-fixture --rounds 6 --capture
python benchmarks/android/preprocess_latency.py --manifest IMAGE_MANIFEST --output artifacts/android-latency/new-preprocessing
python benchmarks/android/shell_contract.py --serial SERIAL --output artifacts/android-latency/new-contract.json
```

At the end of this component study, direct capture remained an isolated
experiment. The subsequent [warm E2E study](android-e2e-latency-2026-11-07.md)
integrates capability reporting and fresh fallback, checks rotation and UI
changes, and measures the actual source runtime against its frozen control.
