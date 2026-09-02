# Third-party notices

IRIS distributes the following unmodified command-line components as separate
programs. IRIS communicates with them through process, pipe, socket, and ADB
interfaces; it does not link their code into IRIS.

## scrcpy server 4.1

- Project: https://github.com/Genymobile/scrcpy
- Binary source: the official `scrcpy-win64-v4.1.zip` release
- License: Apache License 2.0
- Copyright: Genymobile and Romain Vimont
- Installed file: `third_party/scrcpy/scrcpy-server`
- License text: `third_party/scrcpy/LICENSE.txt`

The corresponding scrcpy source is included in the companion
`IRIS-Third-Party-Source.zip` release asset.

## FFmpeg n9.0.1-6-g9d4ca21220-20260823

- Project: https://ffmpeg.org/
- Binary provider: https://github.com/BtbN/FFmpeg-Builds
- Build: Windows x64 LGPL static, without `--enable-gpl` or `--enable-nonfree`
- License: GNU Lesser General Public License 2.1 or later
- Installed file: `third_party/ffmpeg/bin/ffmpeg.exe`
- License text: `third_party/ffmpeg/LICENSE.txt`

This software uses code of FFmpeg licensed under the LGPLv2.1 or later. The
exact FFmpeg source and the pinned BtbN build recipes are included in the
companion `IRIS-Third-Party-Source.zip` release asset. The complete configure
line remains inspectable with `ffmpeg.exe -version`. Nothing in IRIS restricts
reverse engineering or replacement of the separately bundled FFmpeg program.

The notices above are informational and are not legal advice. The license texts
shipped beside each binary control redistribution and use.
