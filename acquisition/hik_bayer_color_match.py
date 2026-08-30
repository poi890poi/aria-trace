"""Offline color matching for MVS Bayer-to-BGR conversion.

The fitted gamma and CCM are applied by MVS inside the Bayer conversion that
the HIK adapter already performs.  This module is calibration-only: it never
adds a per-frame operation to the production stream.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import cv2
import numpy as np


def synchronized_frame_pairs(
    android_times_ns: np.ndarray,
    hik_times_ns: np.ndarray,
    maximum_pairs: int = 16,
) -> list[tuple[int, int, float]]:
    """Return the closest unique frame pairs, ordered by HIK time."""

    candidates = []
    for hik_index, hik_time in enumerate(hik_times_ns.tolist()):
        android_index = int(np.argmin(np.abs(android_times_ns - int(hik_time))))
        delta_ms = abs(int(android_times_ns[android_index]) - int(hik_time)) / 1.0e6
        candidates.append((delta_ms, android_index, hik_index))
    candidates.sort()
    selected = []
    used_android = set()
    used_hik = set()
    for delta_ms, android_index, hik_index in candidates:
        if android_index in used_android or hik_index in used_hik:
            continue
        used_android.add(android_index)
        used_hik.add(hik_index)
        selected.append((android_index, hik_index, float(delta_ms)))
        if len(selected) >= int(maximum_pairs):
            break
    return sorted(selected, key=lambda item: item[1])


def apply_mvs_bayer_model(
    bgr: np.ndarray,
    gamma: float,
    ccm_rgb_3x3: Sequence[Sequence[float]],
) -> np.ndarray:
    """Emulate the measured MVS order: RGB CCM first, scalar gamma second."""

    matrix = np.asarray(ccm_rgb_3x3, np.float64)
    rgb = bgr[..., ::-1].astype(np.float64) / 255.0
    corrected = np.matmul(rgb, matrix.T)
    corrected = np.power(np.clip(corrected, 0.0, 1.0), float(gamma))
    return np.clip(np.round(corrected[..., ::-1] * 255.0), 0, 255).astype(
        np.uint8
    )


def _predict_rgb(
    source_rgb: np.ndarray, gamma: float, matrix: np.ndarray
) -> np.ndarray:
    return np.power(
        np.clip(np.matmul(source_rgb, matrix.T), 0.0, 1.0), float(gamma)
    )


def _fit_matrix(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    gamma: float,
) -> np.ndarray:
    # MVS applies CCM before gamma, so linearize the ADB target for each gamma.
    target_before_gamma = np.power(
        np.clip(target_rgb, 0.0, 1.0), 1.0 / float(gamma)
    )
    weights = np.ones(source_rgb.shape[0], np.float64)
    identity = np.eye(3, dtype=np.float64)
    regularization = max(1.0, source_rgb.shape[0] * 0.002)
    matrix = identity.copy()
    for _ in range(3):
        root = np.sqrt(weights)[:, None]
        design = source_rgb * root
        response = target_before_gamma * root
        augmented_design = np.vstack(
            (design, np.sqrt(regularization) * identity)
        )
        augmented_response = np.vstack(
            (response, np.sqrt(regularization) * identity)
        )
        coefficients = np.linalg.lstsq(
            augmented_design, augmented_response, rcond=None
        )[0]
        matrix = np.clip(coefficients.T, -4.0, 4.0)
        residual = np.linalg.norm(
            _predict_rgb(source_rgb, gamma, matrix) - target_rgb, axis=1
        )
        scale = max(float(np.percentile(residual, 60)), 1.0 / 255.0)
        weights = np.minimum(1.0, (1.5 * scale) / np.maximum(residual, 1.0e-9))
    # Use the exact matrix the runtime can represent.
    return np.round(matrix * 1024.0) / 1024.0


def _metrics(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    gamma: float,
    matrix: np.ndarray,
) -> dict:
    before_gamma = np.matmul(source_rgb, matrix.T)
    predicted = np.power(
        np.clip(before_gamma, 0.0, 1.0), float(gamma)
    )
    absolute = np.abs(predicted - target_rgb) * 255.0
    source_gray = np.matmul(predicted, [0.2126, 0.7152, 0.0722])
    target_gray = np.matmul(target_rgb, [0.2126, 0.7152, 0.0722])
    return {
        "rgb_mae_dn": float(np.mean(absolute)),
        "rgb_p95_absolute_error_dn": float(np.percentile(absolute, 95)),
        "gray_mae_dn": float(np.mean(np.abs(source_gray - target_gray)) * 255.0),
        "ccm_channel_clipping_fraction": float(
            np.mean((before_gamma < 0.0) | (before_gamma > 1.0))
        ),
        "mean_signed_error_rgb_dn": [
            float(value)
            for value in np.mean(predicted - target_rgb, axis=0) * 255.0
        ],
    }


def _labeled_tile(image: np.ndarray, label: str) -> np.ndarray:
    header = np.full((30, image.shape[1], 3), 24, np.uint8)
    cv2.putText(
        header,
        label,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((header, image))


def _sample_pair(
    android: np.ndarray,
    hik: np.ndarray,
    matrix: np.ndarray,
    hik_mask: np.ndarray,
    maximum_pixels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = hik.shape[:2]
    warped = cv2.warpPerspective(
        android,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    android_valid = cv2.warpPerspective(
        np.full(android.shape[:2], 255, np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = cv2.bitwise_and(hik_mask, android_valid)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)

    # Flat regions are robust to the residual sub-pixel registration error and
    # capture delay; edges and moving cursor pixels are poor color references.
    android_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    hik_gray = cv2.cvtColor(hik, cv2.COLOR_BGR2GRAY)
    android_gradient = cv2.magnitude(
        cv2.Sobel(android_gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(android_gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    hik_gradient = cv2.magnitude(
        cv2.Sobel(hik_gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(hik_gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    stable = (
        (mask > 0)
        & (android_gradient < 36.0)
        & (hik_gradient < 36.0)
        & (warped.max(axis=2) < 252)
        & (hik.max(axis=2) < 252)
        & (warped.max(axis=2) > 6)
        & (hik.max(axis=2) > 2)
    )
    indices = np.flatnonzero(stable)
    if indices.size < 256:
        indices = np.flatnonzero(mask > 0)
    if indices.size > maximum_pixels:
        positions = np.linspace(0, indices.size - 1, maximum_pixels).astype(int)
        indices = indices[positions]
    target = warped.reshape((-1, 3))[indices, ::-1].astype(np.float64) / 255.0
    source = hik.reshape((-1, 3))[indices, ::-1].astype(np.float64) / 255.0
    return source, target, warped, mask


def optimize_mvs_bayer_conversion(
    android_frames: np.ndarray,
    android_times_ns: np.ndarray,
    hik_frames: np.ndarray,
    hik_times_ns: np.ndarray,
    android_to_hik_3x3: Sequence[Sequence[float]],
    hik_mask: np.ndarray,
    *,
    maximum_pairs: int = 16,
    maximum_pixels_per_pair: int = 2500,
) -> tuple[dict, Mapping[str, np.ndarray]]:
    """Fit MVS gamma+CCM from synchronized ADB/HIK mini-map pixels."""

    matrix = np.asarray(android_to_hik_3x3, np.float64)
    pairs = synchronized_frame_pairs(
        android_times_ns, hik_times_ns, maximum_pairs=maximum_pairs
    )
    if len(pairs) < 4:
        raise ValueError("Fewer than four unique synchronized frame pairs")

    training_source, training_target = [], []
    validation_source, validation_target = [], []
    evidence_candidates = []
    for pair_index, (android_index, hik_index, delta_ms) in enumerate(pairs):
        source, target, warped, mask = _sample_pair(
            android_frames[android_index],
            hik_frames[hik_index],
            matrix,
            hik_mask,
            maximum_pixels_per_pair,
        )
        if source.shape[0] < 256:
            continue
        destination = (
            (validation_source, validation_target)
            if pair_index % 4 == 0
            else (training_source, training_target)
        )
        destination[0].append(source)
        destination[1].append(target)
        evidence_candidates.append(
            (delta_ms, android_index, hik_index, warped, mask)
        )
    if not training_source or not validation_source:
        raise ValueError("Synchronized frames contain insufficient stable color samples")
    source_train = np.vstack(training_source)
    target_train = np.vstack(training_target)
    source_validation = np.vstack(validation_source)
    target_validation = np.vstack(validation_target)

    identity = np.eye(3, dtype=np.float64)
    baseline = _metrics(
        source_validation, target_validation, 1.0, identity
    )
    candidates = []
    for gamma in np.linspace(0.25, 2.50, 91):
        ccm = _fit_matrix(source_train, target_train, float(gamma))
        metrics = _metrics(
            source_validation, target_validation, float(gamma), ccm
        )
        candidates.append((metrics["rgb_mae_dn"], float(gamma), ccm, metrics))
    candidates.sort(key=lambda item: item[0])
    _, gamma, ccm, selected = candidates[0]
    improvement = baseline["rgb_mae_dn"] - selected["rgb_mae_dn"]
    relative = improvement / max(baseline["rgb_mae_dn"], 1.0e-9)

    delta_ms, android_index, hik_index, warped, mask = min(
        evidence_candidates, key=lambda item: item[0]
    )
    hik = hik_frames[hik_index]
    adjusted = apply_mvs_bayer_model(hik, gamma, ccm)
    x, y, width, height = cv2.boundingRect(mask)
    crop = (slice(y, y + height), slice(x, x + width))
    target_crop = warped[crop]
    baseline_crop = hik[crop]
    adjusted_crop = adjusted[crop]
    before_difference = cv2.absdiff(target_crop, baseline_crop)
    after_difference = cv2.absdiff(target_crop, adjusted_crop)
    evidence = {
        "bayer_color_match_target_adb_warped.png": target_crop,
        "bayer_color_match_hik_identity.png": baseline_crop,
        "bayer_color_match_hik_adjusted.png": adjusted_crop,
        "bayer_color_match_review.png": np.hstack(
            (
                _labeled_tile(target_crop, "ADB target"),
                _labeled_tile(baseline_crop, "HIK identity"),
                _labeled_tile(adjusted_crop, "HIK gamma + CCM"),
                _labeled_tile(before_difference, "absolute error before"),
                _labeled_tile(after_difference, "absolute error after"),
            )
        ),
    }
    summary = {
        "schema_version": "1.0",
        "status": "selected" if improvement > 0.0 else "identity_preferred",
        "backend": "hik_mvs",
        "operation": "MVS Bayer-to-BGR CCM followed by gamma",
        "runtime_application": {
            "gamma_api": "MV_CC_SetGammaValue",
            "ccm_api": "MV_CC_SetBayerCCMParam",
            "timing": "set once after camera open and before the first frame",
            "additional_frame_passes": 0,
            "additional_frame_copies": 0,
        },
        "gamma": gamma if improvement > 0.0 else 1.0,
        "ccm_rgb_3x3": ccm.tolist() if improvement > 0.0 else identity.tolist(),
        "ccm_quantization_scale": 1024,
        "fit": {
            "model_order": "clip(CCM * RGB) then scalar gamma",
            "training_sample_count": int(source_train.shape[0]),
            "validation_sample_count": int(source_validation.shape[0]),
            "synchronized_pair_count": len(evidence_candidates),
            "baseline_validation": baseline,
            "selected_validation": selected,
            "rgb_mae_improvement_dn": float(max(0.0, improvement)),
            "relative_rgb_mae_improvement": float(max(0.0, relative)),
            "representative_pair": {
                "android_frame_array_index": int(android_index),
                "hik_frame_array_index": int(hik_index),
                "absolute_capture_time_delta_ms": float(delta_ms),
            },
        },
        "evidence": list(evidence),
        "non_gating": True,
    }
    return summary, evidence


__all__ = [
    "apply_mvs_bayer_model",
    "optimize_mvs_bayer_conversion",
    "synchronized_frame_pairs",
]
