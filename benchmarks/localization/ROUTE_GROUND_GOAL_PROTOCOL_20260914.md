# Prepared ground-path point: first screen

Type: offline design experiment, no production guidance or game commands.
The route-recognition assay supplies context but its image offsets and retrieved
mouse input are not steering errors. This assay asks whether a physical point
on the future demonstrated path can instead be transferred into the live view.

Use Run21 teaching frames before 124 s and their next one-second frame. A fixed
provisional avatar-foot calibration (320, 290) at 640x360 supplies that future
path point. This calibration is approximate; also transfer its +/-15 px vertical
interval. It is not a precise labelled position or a certified travel axis.
Estimate a ground-plane homography from learned features in a lower trapezoid,
excluding the avatar. Map the future foot point back into the teaching image;
then map that same point into a later traversal using current-only features.
No query future image or input may participate. Reject points outside the
inlier convex hull, weak spatial support, or invalid projective geometry.

The trapezoid is a provisional sensor region, not semantic floor segmentation.
Walls, stairs, shadows, repeating paving and avatar/camera animation can
invalidate it. Review native correspondence pairs, retain rejected preparations,
and report support and availability separately from path accuracy. Run this
small screen before any integrated controller. Do not promote merely because
the output resembles an aiming reticle. A new-view point is not itself a valid
mouse correction until travel/camera response and physical correspondence are
established.

Follow-up after the direct one-second match failed: change preparation only to
short backward LK steps on native frames. Detect up to 400 masked Shi-Tomasi
corners per step; require forward/backward discrepancy below 1.5 px. Apply the
same ground-plane projection and convex-hull gate, accumulating the point back
one second. Keep later-lap transfer and its learned features unchanged. This
tests preparation continuity separately from cross-lap ground correspondence.
