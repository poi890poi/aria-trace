# User-verified route-tracking milestone — September 6, 2026

The user reports that the latest route-tracking test succeeded and requests a
tag and evidence archive. Tag: `route-tracking-known-good-20260906`.

The latest retained live run is `20260906T063250629421Z-5b0a8ea5`, recorded from
14:32:50 to 14:34:00 Asia/Taipei. Its manifest is complete with no runtime error.
It used route-assisted, real-time tracking with a demonstrated start and
current-frame map correlation. It stopped with `route-finish-confirmed`: all
three required current-image confirmations were accepted, the endpoint layer
matched, and reported endpoint distance was 12.830 px within the 14 px gate.

Inputs: atlas `08b6f2d6-820a-4bfd-875a-6a55d1986a4e`, route package
`362f9d53-e80f-4eeb-9cdb-f89959835cd2-72904af0`, mini-map calibration
`segments-df624035-833-bd07601f-708`, and scene-yaw calibration
`01dbaa74-8e00-4763-a215-9ea37e18b1b2`.

The run retains 1,582 telemetry rows: 1,496 fresh XY measurements, 50 held poses,
and 36 unavailable outputs. The longest sampled nonfresh interval, including
startup, is 2.132 s; capture-to-publication P95 is 108.025 ms. Route similarity
P95 is 8.392 canonical px and demonstrated-route coverage is 100% using the
saved 35 px coverage radius. Similarity measures distance to the demonstration,
not independent localization error. No external position truth is available for
this live run, so longest incorrect-position loss cannot be established from
its freshness flags alone.

The overlay video is complete: 1,604 source frames, 2,042 encoded frames,
815 repeated frames, and 319 dropped frames reported by its video sink. Video
playback timestamps are therefore presentation times, not independent capture
timestamps. The original event-frame filenames retain capture timestamps.

This tag records user-verified route completion, not a claim of error-free
localization, sustained 30 Hz control, or autonomous cruising. The capture does
not embed the running process's Git revision. The tagged repository includes
the existing production route tracker plus subsequent session-transfer and
benchmark work; experimental transition policies and candidate atlases remain
inactive.

Evidence package:
`artifacts/exports/route-tracking-known-good-20260906-evidence.zip`.
It includes an offline image gallery, readable summary, original live-run files
(including event images, telemetry and video), atlas and route-package inputs,
calibration JSON, and separately labeled narrow-lap benchmark context. A member
manifest provides SHA-256 identities; the ZIP also has a SHA-256 sidecar.

Type: release documentation and evidence packaging. Verification: user-reported
pass, retained completion/finish metadata, original image inspection, archive
member checksum verification and image decoding. No tracker behavior changes.
