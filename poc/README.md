# Pose estimation POC

This is a supporting experiment for AriaTrace, not the project objective. Its outputs may help align a live run to a human demonstration, but replay success must be measured by route completion and recovery rather than pose accuracy alone.

Portal-start camera initialization is evaluated separately in [PORTAL_INITIALIZATION_RESULTS.md](PORTAL_INITIALIZATION_RESULTS.md). It uses repeated TartanAir visits to one synthetic portal, compares one-direction and bidirectional reference maps, and applies a three-pose confirmation rule.

The relative experiment estimates camera yaw with GFTT features, forward/backward KLT tracking, optional essential-matrix outlier rejection, and robust angular-flow aggregation. The absolute experiment builds a SIFT/COLMAP map and registers held-out frames against it.

The backend contract is intentionally only `reset()` and `update(frame)`. It can later be replaced by full visual odometry, SLAM, or a learned estimator.

## Run

```powershell
python -m pip install -r requirements-poc.txt
python poc\run_yaw_poc.py synthetic --image data\gid\Seq-046\Frames-Sparse\000000000000.jpg
python poc\run_yaw_poc.py video --video data\gid\Seq-046\Seq-046.mp4
python poc\compare_colmap_yaw.py --images artifacts\colmap_seq046\reference_text\images.txt --online-csv artifacts\yaw_poc_colmap\video.csv --output artifacts\yaw_poc_colmap
python poc\replay_pose_fusion.py --trials 100
```

Results are written under `artifacts/` as CSV, JSON, and PNG plots. The tested-state handoff and relocalization commands are in `../PROJECT_STATUS.md`.

The fusion replay uses real TartanAir metric poses and actual PnP outputs. Control-derived motion and the coarse prior are simulated because no aligned game controls or minimap measurements exist yet. See `artifacts/fusion_replay/summary.json` for machine-readable results.

See `RESULTS.md` for the Genshin Seq-046 comparisons.

## Dataset note

Genshin Seq-046 is retained as a realistic game-video test. Its supplied orientation columns are not unit quaternions (norm range observed: 0.000124 to 0.369), so they are not used as quantitative yaw truth. The controlled homography sequence provides exact yaw labels for this first check.

The Genshin run currently uses an approximate focal length (`0.9 * image_width`). Its curve is therefore a robustness result, not an absolute-accuracy result. A calibrated focal length will replace this value for the USB camera.
