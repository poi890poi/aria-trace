"""PySide6 Windows desktop UI for camera-to-phone rig calibration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import qrcode
from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from aria_trace.services.calibration.rig.contracts import ControlEvent, SignalObservation
from aria_trace.services.calibration.rig.feature_matching import (
    aggregate_feature_matching,
    evaluate_feature_matching,
    generate_feature_target,
)
from aria_trace.services.calibration.rig.geometry import CharucoLayout, generate_charuco_target
from aria_trace.services.calibration.rig.image_quality import (
    aggregate_esfr_measurements,
    generate_slanted_edge_target,
    measure_slanted_edge_esfr,
)
from aria_trace.services.calibration.rig.inspection import (
    render_esfr_curve,
    render_feature_matching_curve,
    render_feature_matching_overlay,
    render_latency_timeline,
)
from aria_trace.services.calibration.rig.latency import estimate_latency
from aria_trace.adapters.rig.devices import (
    AdbAdapter,
    CameraAdapter,
    CameraConfiguration,
    create_adb_adapter,
    create_camera_adapter,
    load_adapter_factory,
)
from aria_trace.adapters.android.display import LocalPhoneTargetServer, PhoneTargetAdapter
from aria_trace.workflows.rig_calibration import (
    CalibrationAnalysis,
    CalibrationInputs,
    analyze_frame,
    save_analysis_bundle,
)


def _bgr_qimage(image: np.ndarray) -> QImage:
    if image.ndim == 2:
        contiguous = np.ascontiguousarray(image)
        value = QImage(
            contiguous.data,
            contiguous.shape[1],
            contiguous.shape[0],
            contiguous.strides[0],
            QImage.Format_Grayscale8,
        )
    else:
        rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        value = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format_RGB888,
        )
    return value.copy()


class ImagePane(QScrollArea):
    """Image view that can preserve exact source pixels or fit the viewport."""

    def __init__(self, exact_pixels: bool = False) -> None:
        super().__init__()
        self.exact_pixels = bool(exact_pixels)
        self._image: Optional[QImage] = None
        self.label = QLabel("No image yet")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background:#16191d;color:#9ca3af;")
        self.setWidget(self.label)
        self.setWidgetResizable(not self.exact_pixels)

    def set_bgr(self, image: np.ndarray) -> None:
        self._image = _bgr_qimage(image)
        self._refresh()

    def _refresh(self) -> None:
        if self._image is None:
            return
        pixmap = QPixmap.fromImage(self._image)
        if not self.exact_pixels:
            target = self.viewport().size()
            pixmap = pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(pixmap)
        self.label.resize(pixmap.size())

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        if not self.exact_pixels:
            self._refresh()


class CameraThread(QThread):
    frame_ready = Signal(object)
    opened = Signal(object)
    failed = Signal(str)

    def __init__(self, adapter: CameraAdapter, configuration: CameraConfiguration) -> None:
        super().__init__()
        self.adapter = adapter
        self.configuration = configuration

    def run(self) -> None:
        try:
            effective = self.adapter.open(self.configuration)
            self.opened.emit(dict(effective))
            while not self.isInterruptionRequested():
                self.frame_ready.emit(self.adapter.read())
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit("{}: {}".format(type(exc).__name__, exc))
        finally:
            self.adapter.close()


class AnalysisThread(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        frame: np.ndarray,
        inputs: CalibrationInputs,
        layout: CharucoLayout,
        target: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        super().__init__()
        self.frame = frame.copy()
        self.inputs = inputs
        self.layout = layout
        self.target = target.copy()
        self.metadata = dict(metadata)

    def run(self) -> None:
        try:
            self.completed.emit(
                analyze_frame(
                    self.frame,
                    self.inputs,
                    self.layout,
                    self.target,
                    self.metadata,
                )
            )
        except Exception as exc:
            self.failed.emit("{}: {}".format(type(exc).__name__, exc))


class RigCalibrationWindow(QMainWindow):
    """Guided desktop workflow; all device access is operator initiated."""

    def __init__(
        self,
        camera: CameraAdapter,
        adb: AdbAdapter,
        phone_target: PhoneTargetAdapter,
        output_root: Path,
    ) -> None:
        super().__init__()
        self.camera = camera
        self.adb = adb
        self.phone_target = phone_target
        self.output_root = Path(output_root)
        self.camera_thread: Optional[CameraThread] = None
        self.analysis_thread: Optional[AnalysisThread] = None
        self.latest_sample: Any = None
        self.camera_metadata: dict[str, Any] = {}
        self.layout: Optional[CharucoLayout] = None
        self.target_image: Optional[np.ndarray] = None
        self.analysis: Optional[CalibrationAnalysis] = None
        self.image_quality: Optional[dict[str, Any]] = None
        self.feature_matching: Optional[dict[str, Any]] = None
        self.quality_evidence: dict[str, np.ndarray] = {}
        self.timing: Optional[dict[str, Any]] = None
        self.adb_reference: Optional[np.ndarray] = None
        self._quality_plan: list[dict[str, Any]] = []
        self._esfr_measurements: list[dict[str, Any]] = []
        self._feature_trials: list[dict[str, Any]] = []
        self._quality_failures: list[dict[str, str]] = []
        self._quality_active = False
        self._latency_active = False
        self._latency_events: list[ControlEvent] = []
        self._latency_observations: list[SignalObservation] = []
        self._latency_index = 0
        self._latency_total = 0
        self._latency_timer = QTimer(self)
        self._latency_timer.setSingleShot(True)
        self._latency_timer.timeout.connect(self._latency_tick)
        self._build_ui()
        self._set_status(
            "Idle — camera, ADB, and phone target are not accessed until you request them."
        )

    def _build_ui(self) -> None:
        self.setWindowTitle("AriaTrace Rig Calibration")
        self.resize(1440, 920)
        central = QWidget()
        outer = QVBoxLayout(central)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("padding:9px;background:#20252b;color:#e5e7eb;")
        outer.addWidget(self.status)
        body = QHBoxLayout()
        outer.addLayout(body, 1)
        self.images = QTabWidget()
        self.image_panes: dict[str, ImagePane] = {}
        for title, exact in (
            ("Live camera", False),
            ("Geometry evidence", False),
            ("Normalized phone", False),
            ("Exact 1:1 pixels", True),
            ("4× nearest-neighbour", True),
            ("e-SFR / MTF", False),
            ("Feature matching", False),
            ("Match examples", False),
            ("Edge evidence", True),
            ("Latency", False),
            ("ADB reference", False),
        ):
            pane = ImagePane(exact_pixels=exact)
            self.image_panes[title] = pane
            self.images.addTab(pane, title)
        body.addWidget(self.images, 3)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls = QWidget()
        self.controls_layout = QVBoxLayout(controls)
        self.controls_layout.addWidget(self._phone_group())
        self.controls_layout.addWidget(self._camera_group())
        self.controls_layout.addWidget(self._geometry_group())
        self.controls_layout.addWidget(self._quality_group())
        self.controls_layout.addWidget(self._latency_group())
        self.controls_layout.addWidget(self._adb_group())
        self.controls_layout.addWidget(self._save_group())
        self.controls_layout.addStretch(1)
        controls_scroll.setWidget(controls)
        body.addWidget(controls_scroll, 2)
        self.setCentralWidget(central)

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        return widget

    @staticmethod
    def _double(minimum: float, maximum: float, value: float, suffix: str = "") -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(2)
        widget.setValue(value)
        widget.setSuffix(suffix)
        return widget

    def _phone_group(self) -> QGroupBox:
        group = QGroupBox("1 · Phone target")
        form = QFormLayout(group)
        dimensions = QHBoxLayout()
        self.screen_width = self._spin(240, 10000, 1080)
        self.screen_height = self._spin(240, 10000, 2400)
        dimensions.addWidget(self.screen_width)
        dimensions.addWidget(QLabel("×"))
        dimensions.addWidget(self.screen_height)
        form.addRow("Canonical pixels", dimensions)
        self.phone_diagonal = self._double(2.0, 30.0, 6.5, " in")
        form.addRow("Display diagonal", self.phone_diagonal)
        self.squares_x = self._spin(3, 30, 7)
        self.squares_y = self._spin(3, 40, 11)
        board = QHBoxLayout()
        board.addWidget(self.squares_x)
        board.addWidget(QLabel("×"))
        board.addWidget(self.squares_y)
        form.addRow("ChArUco squares", board)
        self.phone_url = QLineEdit()
        self.phone_url.setReadOnly(True)
        self.phone_url.setPlaceholderText("Start the target service to obtain a URL")
        form.addRow("Phone URL", self.phone_url)
        self.phone_qr = QLabel("QR appears after the service starts")
        self.phone_qr.setAlignment(Qt.AlignCenter)
        self.phone_qr.setMinimumHeight(150)
        form.addRow(self.phone_qr)
        buttons = QHBoxLayout()
        start = QPushButton("Start target")
        stop = QPushButton("Stop target")
        start.clicked.connect(self._start_target)
        stop.clicked.connect(self._stop_target)
        buttons.addWidget(start)
        buttons.addWidget(stop)
        form.addRow(buttons)
        note = QLabel("Open the URL on the phone and tap its fullscreen button. No port is bound before Start target.")
        note.setWordWrap(True)
        form.addRow(note)
        return group

    def _camera_group(self) -> QGroupBox:
        group = QGroupBox("2 · USB camera")
        form = QFormLayout(group)
        self.camera_device = QLineEdit("0")
        form.addRow("Device ID / URL", self.camera_device)
        self.camera_backend = QComboBox()
        self.camera_backend.addItems(["dshow", "msmf", "auto"])
        form.addRow("Backend", self.camera_backend)
        mode = QHBoxLayout()
        self.camera_width = self._spin(160, 16384, 1920)
        self.camera_height = self._spin(120, 16384, 1080)
        self.camera_fps = self._double(1.0, 240.0, 30.0, " fps")
        mode.addWidget(self.camera_width)
        mode.addWidget(QLabel("×"))
        mode.addWidget(self.camera_height)
        mode.addWidget(self.camera_fps)
        form.addRow("Requested mode", mode)
        buttons = QHBoxLayout()
        probe = QPushButton("Probe indices")
        start = QPushButton("Start camera")
        stop = QPushButton("Stop camera")
        probe.clicked.connect(self._probe_cameras)
        start.clicked.connect(self._start_camera)
        stop.clicked.connect(self._stop_camera)
        buttons.addWidget(probe)
        buttons.addWidget(start)
        buttons.addWidget(stop)
        form.addRow(buttons)
        note = QLabel("Probing briefly opens candidate devices. It happens only when you click Probe indices.")
        note.setWordWrap(True)
        form.addRow(note)
        return group

    def _geometry_group(self) -> QGroupBox:
        group = QGroupBox("3 · Position and geometry")
        form = QFormLayout(group)
        self.roi_x = self._spin(0, 10000, 0)
        self.roi_y = self._spin(0, 10000, 0)
        self.roi_width = self._spin(1, 10000, 1080)
        self.roi_height = self._spin(1, 10000, 2400)
        roi = QGridLayout()
        roi.addWidget(QLabel("X"), 0, 0)
        roi.addWidget(self.roi_x, 0, 1)
        roi.addWidget(QLabel("Y"), 0, 2)
        roi.addWidget(self.roi_y, 0, 3)
        roi.addWidget(QLabel("W"), 1, 0)
        roi.addWidget(self.roi_width, 1, 1)
        roi.addWidget(QLabel("H"), 1, 2)
        roi.addWidget(self.roi_height, 1, 3)
        form.addRow("Required ROI", roi)
        self.camera_hfov = self._double(5.0, 175.0, 70.0, "°")
        form.addRow("Estimated camera HFOV", self.camera_hfov)
        fit = QPushButton("Fit reviewed current frame")
        fit.clicked.connect(self._fit_geometry)
        form.addRow(fit)
        self.geometry_results = QTextEdit()
        self.geometry_results.setReadOnly(True)
        self.geometry_results.setMinimumHeight(190)
        form.addRow(self.geometry_results)
        return group

    def _quality_group(self) -> QGroupBox:
        group = QGroupBox("4 · Display detail and feature matching")
        form = QFormLayout(group)
        atlas_note = QLabel(
            "The ChArUco atlas must be fitted first. Trials use only the camera-visible "
            "part of the required ROI; whole-display visibility is not assumed."
        )
        atlas_note.setWordWrap(True)
        form.addRow(atlas_note)
        self.quality_use_adb = QCheckBox(
            "Use the configured ADB adapter for feature-target references"
        )
        self.quality_use_adb.setToolTip(
            "Opt-in: invokes ADB only for the three feature-matching targets. "
            "Slanted-edge geometry always comes from its exact generated specification."
        )
        form.addRow(self.quality_use_adb)
        sweep = QPushButton("Run ISO e-SFR and feature-matching trials")
        sweep.clicked.connect(self._start_quality_sweep)
        form.addRow(sweep)
        self.quality_result = QLabel("Fit the ChArUco atlas and review IoU first.")
        self.quality_result.setWordWrap(True)
        form.addRow(self.quality_result)
        self.matching_result = QLabel("Feature matching not measured.")
        self.matching_result.setWordWrap(True)
        form.addRow(self.matching_result)
        return group

    def _latency_group(self) -> QGroupBox:
        group = QGroupBox("5 · Control-to-perception latency")
        form = QFormLayout(group)
        self.latency_transitions = self._spin(4, 128, 16)
        self.latency_interval = self._spin(300, 5000, 700)
        self.latency_interval.setSuffix(" ms")
        form.addRow("Alternations", self.latency_transitions)
        form.addRow("Signal interval", self.latency_interval)
        run = QPushButton("Run camera endpoint")
        run.clicked.connect(self._start_latency)
        form.addRow(run)
        self.latency_result = QLabel("Not measured")
        self.latency_result.setWordWrap(True)
        form.addRow(self.latency_result)
        return group

    def _adb_group(self) -> QGroupBox:
        group = QGroupBox("Optional · ADB reference")
        form = QFormLayout(group)
        self.adb_serial = QLineEdit()
        self.adb_serial.setPlaceholderText("blank = ADB default device")
        form.addRow("Serial", self.adb_serial)
        buttons = QHBoxLayout()
        discover = QPushButton("List devices")
        capture = QPushButton("Capture reference")
        discover.clicked.connect(self._list_adb)
        capture.clicked.connect(self._capture_adb)
        buttons.addWidget(discover)
        buttons.addWidget(capture)
        form.addRow(buttons)
        self.adb_status = QLabel("ADB is never called automatically.")
        self.adb_status.setWordWrap(True)
        form.addRow(self.adb_status)
        return group

    def _save_group(self) -> QGroupBox:
        group = QGroupBox("6 · Review and save")
        form = QFormLayout(group)
        self.output_path = QLineEdit(str(self.output_root))
        browse = QPushButton("Choose…")
        browse.clicked.connect(self._choose_output)
        row = QHBoxLayout()
        row.addWidget(self.output_path, 1)
        row.addWidget(browse)
        form.addRow("Artifact root", row)
        save = QPushButton("Save reviewed calibration bundle")
        save.clicked.connect(self._save)
        form.addRow(save)
        self.save_result = QLabel("Nothing saved yet")
        self.save_result.setWordWrap(True)
        form.addRow(self.save_result)
        return group

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status.setText(text)
        color = "#5b2026" if error else "#20252b"
        self.status.setStyleSheet("padding:9px;background:{};color:#f3f4f6;".format(color))

    def _current_layout(self) -> CharucoLayout:
        width, height = self.screen_width.value(), self.screen_height.value()
        margin = max(8, int(round(min(width, height) * 0.025)))
        return CharucoLayout(
            screen_size_px=(width, height),
            squares_x=self.squares_x.value(),
            squares_y=self.squares_y.value(),
            margin_px=(margin, margin),
        )

    def _current_inputs(self) -> CalibrationInputs:
        return CalibrationInputs(
            screen_size_px=(self.screen_width.value(), self.screen_height.value()),
            required_roi_xywh=(
                self.roi_x.value(),
                self.roi_y.value(),
                self.roi_width.value(),
                self.roi_height.value(),
            ),
            phone_diagonal_in=self.phone_diagonal.value(),
            camera_horizontal_fov_deg=self.camera_hfov.value(),
            patch_size_mm=20.0,
        )

    def _start_target(self) -> None:
        try:
            self.layout = self._current_layout()
            self.target_image = generate_charuco_target(self.layout)
            url = self.phone_target.start(self.layout)
            self.phone_url.setText(url)
            QApplication.clipboard().setText(url)
            qr_rgb = np.asarray(qrcode.make(url).convert("RGB"), dtype=np.uint8)
            qr_bgr = cv2.cvtColor(qr_rgb, cv2.COLOR_RGB2BGR)
            self.phone_qr.setPixmap(
                QPixmap.fromImage(_bgr_qimage(qr_bgr)).scaled(
                    180, 180, Qt.KeepAspectRatio, Qt.FastTransformation
                )
            )
            self._set_status("Phone target ready. URL copied to clipboard; open it on the phone and enter fullscreen.")
        except Exception as exc:
            self._set_status("Cannot start phone target: {}".format(exc), True)

    def _stop_target(self) -> None:
        try:
            self.phone_target.stop()
            self.phone_url.clear()
            self.phone_qr.setText("QR appears after the service starts")
            self._set_status("Phone target stopped; its listening port is closed.")
        except Exception as exc:
            self._set_status("Cannot stop phone target: {}".format(exc), True)

    def _probe_cameras(self) -> None:
        try:
            devices = self.camera.devices(probe=True)
            if devices:
                self.camera_device.setText(devices[0].device_id)
                labels = ", ".join("{} ({})".format(item.label, item.device_id) for item in devices)
                self._set_status("Camera probe found: {}".format(labels))
            else:
                self._set_status("Camera probe found no devices. You can still enter an adapter-specific ID.", True)
        except Exception as exc:
            self._set_status("Camera probe failed: {}".format(exc), True)

    def _start_camera(self) -> None:
        if self.camera_thread is not None and self.camera_thread.isRunning():
            self._set_status("Camera is already running.")
            return
        try:
            configuration = CameraConfiguration(
                device_id=self.camera_device.text().strip(),
                width_px=self.camera_width.value(),
                height_px=self.camera_height.value(),
                fps=self.camera_fps.value(),
                backend=self.camera_backend.currentText(),
            )
            thread = CameraThread(self.camera, configuration)
            thread.frame_ready.connect(self._on_frame)
            thread.opened.connect(self._on_camera_opened)
            thread.failed.connect(lambda text: self._set_status(text, True))
            thread.finished.connect(self._on_camera_stopped)
            self.camera_thread = thread
            thread.start()
            self._set_status("Opening the selected camera…")
        except Exception as exc:
            self._set_status("Cannot start camera: {}".format(exc), True)

    def _stop_camera(self) -> bool:
        thread = self.camera_thread
        if thread is None:
            return True
        thread.requestInterruption()
        if not thread.wait(3000):
            self._set_status("Camera did not stop promptly; it will be released when its read returns.", True)
            return False
        else:
            self._set_status("Camera released.")
            return True

    def _on_camera_opened(self, metadata: object) -> None:
        self.camera_metadata = dict(metadata)
        self._set_status("Camera active: {}×{} at requested {:.1f} fps.".format(
            self.camera_metadata.get("width_px", "?"),
            self.camera_metadata.get("height_px", "?"),
            self.camera_fps.value(),
        ))

    def _on_camera_stopped(self) -> None:
        self.camera_thread = None

    def _on_frame(self, sample: object) -> None:
        self.latest_sample = sample
        self.image_panes["Live camera"].set_bgr(sample.image)
        if self._latency_active and self.analysis is not None:
            polygon = np.asarray(self.analysis.geometry.screen_polygon_input_xy, dtype=np.float64)
            center = np.mean(polygon, axis=0)
            edge = min(
                np.linalg.norm(polygon[1] - polygon[0]),
                np.linalg.norm(polygon[3] - polygon[0]),
            )
            radius = max(3, int(round(edge * 0.10)))
            x0 = max(0, int(round(center[0])) - radius)
            y0 = max(0, int(round(center[1])) - radius)
            x1 = min(sample.image.shape[1], x0 + radius * 2 + 1)
            y1 = min(sample.image.shape[0], y0 + radius * 2 + 1)
            patch = sample.image[y0:y1, x0:x1]
            if patch.size:
                probability_white = float(np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)) / 255.0)
                self._latency_observations.append(
                    SignalObservation(
                        time_ns=sample.time_ns,
                        probabilities={"black": 1.0 - probability_white, "white": probability_white},
                        source_id=sample.source_id,
                        metadata={"mean_luma": probability_white * 255.0},
                    )
                )

    def _fit_geometry(self) -> None:
        if self.latest_sample is None:
            self._set_status("Start the camera and obtain a frame before fitting geometry.", True)
            return
        if self.layout is None or self.target_image is None:
            self._set_status("Start the phone target before fitting geometry.", True)
            return
        if self.analysis_thread is not None and self.analysis_thread.isRunning():
            return
        try:
            browser = dict(self.phone_target.telemetry().get("browser", {}))
            canvas_width = browser.get("canvas_width")
            canvas_height = browser.get("canvas_height")
            declared_width, declared_height = self._current_inputs().screen_size_px
            if canvas_width is not None and canvas_height is not None and (
                int(canvas_width) != declared_width
                or int(canvas_height) != declared_height
            ):
                raise ValueError(
                    "Phone canvas is {}x{} physical pixels, but the declared display "
                    "raster is {}x{}. Enter fullscreen and make these values agree "
                    "before fitting the atlas.".format(
                        canvas_width,
                        canvas_height,
                        declared_width,
                        declared_height,
                    )
                )
            thread = AnalysisThread(
                self.latest_sample.image,
                self._current_inputs(),
                self.layout,
                self.target_image,
                self.camera_metadata,
            )
            thread.completed.connect(self._analysis_complete)
            thread.failed.connect(self._analysis_failed)
            thread.finished.connect(self._analysis_finished)
            self.analysis_thread = thread
            thread.start()
            self._set_status("Fitting ChArUco geometry from the reviewed frame…")
        except Exception as exc:
            self._set_status("Cannot start geometry fit: {}".format(exc), True)

    def _analysis_complete(self, value: object) -> None:
        self.analysis = value
        self.image_quality = None
        self.feature_matching = None
        self.quality_evidence = {}
        self.timing = None
        self.image_panes["Geometry evidence"].set_bgr(value.overlay)
        self.image_panes["Normalized phone"].set_bgr(value.normalized)
        self.image_panes["Exact 1:1 pixels"].set_bgr(value.one_to_one_patch)
        self.image_panes["4× nearest-neighbour"].set_bgr(value.magnified_patch)
        metrics = value.geometry.metrics
        pose = value.guidance["pose"]
        focus = value.guidance["focus"]
        quality_region = value.guidance["quality_region"]
        lines = [
            "ChArUco atlas: {} corners / {} markers".format(value.detection["corner_count"], value.detection["marker_count"]),
            "Screen coverage: {:.1%}".format(metrics["screen_coverage"]),
            "Camera utilization: {:.1%}".format(metrics["camera_utilization"]),
            "Camera/screen IoU: {:.1%}".format(metrics["screen_view_iou"]),
            "Required ROI coverage: {:.1%}".format(metrics["required_region_coverage"]),
            "Required ROI supported by corners: {:.1%}".format(metrics["required_region_detected_hull_coverage"]),
            "Reprojection P95: {:.2f} px".format(metrics["reprojection_p95_px"]),
            "Geometry confidence: {:.1%}".format(value.geometry.confidence),
            (
                "Visible quality patch: x={} y={} w={} h={} display px".format(
                    *quality_region["xywh"]
                )
                if quality_region["xywh"] is not None
                else "Visible quality patch: unavailable ({})".format(
                    quality_region.get("error", "insufficient visible task area")
                )
            ),
            "Approx. distance: {:.0f} mm".format(pose["distance_mm"]),
            "Approx. off-axis: {:.1f}°; roll: {:.1f}°".format(pose["off_axis_deg"], pose["roll_deg"]),
            "Relative focus: Laplacian {:.1f}; Tenengrad {:.1f}".format(
                focus["laplacian_variance_relative"], focus["tenengrad_relative"]
            ),
            "",
            "Guidance:",
        ] + ["• " + message for message in value.guidance["messages"]]
        lines.append("")
        lines.append(
            "Atlas geometry and IoU are established first. Display-referred e-SFR/MTF and "
            "feature matching use only the visible quality patch."
        )
        lines.append(
            "Distance and tilt use the entered diagonal and HFOV assumptions; they do not affect cy/dpx."
        )
        self.geometry_results.setPlainText("\n".join(lines))
        self.images.setCurrentWidget(self.image_panes["Geometry evidence"])
        self._set_status("Geometry evidence is ready for review. Inspect the overlay and exact-pixel focus tabs.")

    def _analysis_failed(self, text: str) -> None:
        self._set_status("Geometry fit failed: {}".format(text), True)

    def _analysis_finished(self) -> None:
        self.analysis_thread = None

    def _start_quality_sweep(self) -> None:
        if self.analysis is None:
            self._set_status("Fit and review the ChArUco atlas and IoU before quality measurement.", True)
            return
        if not self.phone_url.text():
            self._set_status("Start the phone target service before quality measurement.", True)
            return
        if self._quality_active or self._latency_active:
            self._set_status("Another controlled presentation is already active.", True)
            return
        quality_rect = self.analysis.guidance["quality_region"]["xywh"]
        if quality_rect is None:
            self._set_status(
                "Atlas IoU is available, but the camera-visible required ROI is too small "
                "for a safe quality patch. Reposition the camera or adjust the required ROI.",
                True,
            )
            return
        self._quality_plan = []
        for channel in ("luminance", "red", "green", "blue"):
            for angle in (5.0, 85.0):
                for phase in (0.0, 0.5):
                    self._quality_plan.append(
                        {
                            "kind": "esfr",
                            "channel": channel,
                            "angle": angle,
                            "phase": phase,
                            "label": "e-SFR {} {:.0f}deg phase {:.1f}".format(
                                channel, angle, phase
                            ),
                        }
                    )
        for repeat, seed in enumerate((7319, 11587, 19001), 1):
            self._quality_plan.append(
                {
                    "kind": "feature_matching",
                    "seed": seed,
                    "label": "feature matching target {}".format(repeat),
                }
            )
        self._quality_total = len(self._quality_plan)
        self._esfr_measurements = []
        self._feature_trials = []
        self._quality_failures = []
        self.quality_evidence = {}
        self.image_quality = None
        self.feature_matching = None
        self._quality_active = True
        self.quality_result.setText(
            "Running {} controlled observations in the visible display patch…".format(
                self._quality_total
            )
        )
        self.matching_result.setText("Waiting for feature targets.")
        self._set_status(
            "e-SFR and matching trials are active. Keep the atlas-fitted rig fixed."
        )
        self._quality_next()

    def _quality_next(self) -> None:
        if not self._quality_plan:
            self._finish_quality()
            return
        item = self._quality_plan.pop(0)
        try:
            quality_rect = self.analysis.guidance["quality_region"]["xywh"]
            if item["kind"] == "esfr":
                item["target"] = generate_slanted_edge_target(
                    self.analysis.inputs.screen_size_px,
                    quality_rect,
                    edge_angle_deg=item["angle"],
                    phase_display_px=item["phase"],
                    channel=item["channel"],
                )
            else:
                item["target"] = generate_feature_target(
                    self.analysis.inputs.screen_size_px,
                    quality_rect,
                    item["seed"],
                )
            presentation = self.phone_target.present_image(item["target"], item["label"])
            item["issued_time_ns"] = int(presentation.issued_time_ns)
            item["revision"] = int(presentation.revision)
            item["ack_wait_attempts"] = 0
            item["frame_wait_attempts"] = 0
        except Exception as exc:
            self._quality_active = False
            self._set_status("Cannot present quality target: {}".format(exc), True)
            return
        QTimer.singleShot(750, lambda: self._quality_capture(item))

    def _quality_capture(self, item: dict[str, Any]) -> None:
        if not self._quality_active or self.latest_sample is None or self.analysis is None:
            self._quality_active = False
            return
        acknowledgements = self.phone_target.telemetry().get(
            "acknowledgements", []
        )
        matching_acknowledgements = [
            value
            for value in acknowledgements
            if int(value.get("revision", -1)) == int(item.get("revision", -2))
            and bool(value.get("painted", False))
        ]
        if not matching_acknowledgements:
            if int(item.get("ack_wait_attempts", 0)) < 20:
                item["ack_wait_attempts"] = int(
                    item.get("ack_wait_attempts", 0)
                ) + 1
                QTimer.singleShot(100, lambda: self._quality_capture(item))
                return
            self._quality_failures.append(
                {
                    "kind": str(item.get("kind")),
                    "label": str(item.get("label")),
                    "error": "phone did not acknowledge painting this target revision",
                }
            )
            self._quality_next()
            return
        paint_ack_time = max(
            int(value.get("server_receive_time_ns", 0))
            for value in matching_acknowledgements
        )
        # Phone presentation and acknowledgement timestamps use this process's
        # monotonic clock.
        # Custom camera adapters may expose a device clock in ``time_ns``;
        # ``receive_time_ns`` is therefore the comparable freshness clock.
        receive_time = getattr(self.latest_sample, "receive_time_ns", None)
        frame_clock = str(
            getattr(self.latest_sample, "clock_id", "host_monotonic_ns")
        )
        if receive_time is None and frame_clock != "host_monotonic_ns":
            self._quality_failures.append(
                {
                    "kind": str(item.get("kind")),
                    "label": str(item.get("label")),
                    "error": (
                        "camera adapter uses clock {!r} but did not supply a "
                        "host-monotonic receive_time_ns"
                    ).format(frame_clock),
                }
            )
            self._quality_next()
            return
        frame_time = int(
            receive_time
            if receive_time is not None
            else getattr(self.latest_sample, "time_ns", 0)
        )
        freshness_boundary = max(
            int(item.get("issued_time_ns", 0)), paint_ack_time
        )
        if frame_time <= freshness_boundary:
            if int(item.get("frame_wait_attempts", 0)) < 20:
                item["frame_wait_attempts"] = int(
                    item.get("frame_wait_attempts", 0)
                ) + 1
                QTimer.singleShot(100, lambda: self._quality_capture(item))
                return
            self._quality_failures.append(
                {
                    "kind": str(item.get("kind")),
                    "label": str(item.get("label")),
                    "error": "no post-presentation camera frame arrived",
                }
            )
            self._quality_next()
            return
        try:
            camera_frame = self.latest_sample.image.copy()
            quality_rect = self.analysis.guidance["quality_region"]["xywh"]
            transform = self.analysis.geometry.matrix_3x3
            if item["kind"] == "esfr":
                result, evidence = measure_slanted_edge_esfr(
                    camera_frame,
                    transform,
                    quality_rect,
                    edge_angle_deg=item["angle"],
                    phase_display_px=item["phase"],
                    channel=item["channel"],
                    geometry_confidence=self.analysis.geometry.confidence,
                )
                result["target_label"] = item["label"]
                result["presentation_issued_time_ns"] = int(item["issued_time_ns"])
                result["phone_paint_ack_time_ns"] = paint_ack_time
                result["camera_frame_time_ns"] = frame_time
                result["camera_frame_time_basis"] = (
                    "host_receive_monotonic"
                    if receive_time is not None
                    else "host_monotonic_adapter_time"
                )
                self._esfr_measurements.append(result)
                evidence_name = "edge-{}-{:02d}-phase-{}".format(
                    item["channel"], int(item["angle"]), str(item["phase"]).replace(".", "p")
                )
                self.quality_evidence[evidence_name] = evidence
                self.image_panes["Edge evidence"].set_bgr(evidence)
            else:
                reference = item["target"]
                reference_mode = "generated_to_camera"
                if self.quality_use_adb.isChecked():
                    adb_sample = self.adb.capture_screen(
                        self.adb_serial.text().strip() or None
                    )
                    expected_size = self.analysis.inputs.screen_size_px
                    if adb_sample.image.shape[1::-1] != expected_size:
                        raise ValueError(
                            "ADB reference is {}x{}, expected canonical {}x{}".format(
                                adb_sample.image.shape[1],
                                adb_sample.image.shape[0],
                                expected_size[0],
                                expected_size[1],
                            )
                        )
                    reference = adb_sample.image
                    reference_mode = "adb_to_camera"
                    self.adb_reference = reference.copy()
                    self.image_panes["ADB reference"].set_bgr(self.adb_reference)
                trial = evaluate_feature_matching(
                    reference,
                    camera_frame,
                    transform,
                    quality_rect,
                    reference_mode=reference_mode,
                    geometry_confidence=self.analysis.geometry.confidence,
                )
                trial["target_label"] = item["label"]
                trial["target_seed"] = int(item["seed"])
                trial["presentation_issued_time_ns"] = int(item["issued_time_ns"])
                trial["phone_paint_ack_time_ns"] = paint_ack_time
                trial["camera_frame_time_ns"] = frame_time
                trial["camera_frame_time_basis"] = (
                    "host_receive_monotonic"
                    if receive_time is not None
                    else "host_monotonic_adapter_time"
                )
                self._feature_trials.append(trial)
                match_evidence = render_feature_matching_overlay(
                    reference, camera_frame, trial
                )
                evidence_name = "feature-matches-seed-{}".format(item["seed"])
                self.quality_evidence[evidence_name] = match_evidence
                self.image_panes["Match examples"].set_bgr(match_evidence)
        except Exception as exc:
            self._quality_failures.append(
                {"kind": str(item.get("kind")), "label": str(item.get("label")), "error": str(exc)}
            )
        completed = self._quality_total - len(self._quality_plan)
        self.quality_result.setText(
            "Captured {} of {} observations; {} failed conditions.".format(
                completed, self._quality_total, len(self._quality_failures)
            )
        )
        self._quality_next()

    def _finish_quality(self) -> None:
        try:
            if not self._esfr_measurements:
                raise ValueError("Every slanted-edge condition failed")
            self.image_quality = aggregate_esfr_measurements(self._esfr_measurements)
            self.image_quality["quality_region"] = dict(
                self.analysis.guidance["quality_region"]
            )
            self.image_quality["failed_conditions"] = [
                item for item in self._quality_failures if item["kind"] == "esfr"
            ]
            display = self.image_quality["display_referred"]
            edge_failures = len(
                [
                    item
                    for item in self._quality_failures
                    if item["kind"] == "esfr"
                ]
            )
            self.quality_result.setText(
                "Display-referred MTF50 {}; MTF10 {} cy/dpx; {} measured edge conditions, "
                "{} failures; confidence {:.1%}.".format(
                    "{:.3f}".format(display["mtf50_conservative"])
                    if display["mtf50_conservative"] is not None
                    else ">0.500",
                    "{:.3f}".format(display["mtf10_conservative"])
                    if display["mtf10_conservative"] is not None
                    else ">0.500",
                    len(self._esfr_measurements),
                    edge_failures,
                    self.image_quality["confidence"],
                )
            )
            self.image_panes["e-SFR / MTF"].set_bgr(render_esfr_curve(self.image_quality))
            if self._feature_trials:
                self.feature_matching = aggregate_feature_matching(self._feature_trials)
                self.feature_matching["failed_trials"] = [
                    item
                    for item in self._quality_failures
                    if item["kind"] == "feature_matching"
                ]
                primary = self.feature_matching["primary_threshold_display_px"]
                self.matching_result.setText(
                    "At {} display px: repeatability {:.1%}, matching score {:.1%}, "
                    "MMA {:.1%}; {} evaluated matches; coverage {:.1%}; catastrophic {:.1%}.".format(
                        primary,
                        self.feature_matching["repeatability_by_threshold_px"][primary],
                        self.feature_matching["matching_score_by_threshold_px"][primary],
                        self.feature_matching["mma_by_threshold_px"][primary],
                        self.feature_matching["evaluated_match_count"],
                        self.feature_matching["spatial_coverage_min"],
                        self.feature_matching["catastrophic_mismatch_rate"],
                    )
                )
                self.image_panes["Feature matching"].set_bgr(
                    render_feature_matching_curve(self.feature_matching)
                )
            else:
                self.feature_matching = None
                self.matching_result.setText("Every feature-matching target failed.")
            self.images.setCurrentWidget(self.image_panes["e-SFR / MTF"])
            self._set_status(
                "Display-referred e-SFR and ground-truth matching evidence are ready for review."
            )
        except Exception as exc:
            self._set_status("Quality evaluation failed: {}".format(exc), True)
        finally:
            self._quality_active = False
            try:
                self.phone_target.present_charuco()
            except Exception:
                pass

    def _start_latency(self) -> None:
        if self.analysis is None or not self.phone_url.text():
            self._set_status("Geometry and a running phone target are required for latency measurement.", True)
            return
        if self._quality_active or self._latency_active:
            self._set_status("Another controlled presentation is already active.", True)
            return
        self._latency_active = True
        self._latency_events = []
        self._latency_observations = []
        self._latency_index = 0
        self._latency_total = self.latency_transitions.value()
        self.latency_result.setText("Measuring {} alternations…".format(self._latency_total))
        self._set_status("Latency measurement active. Keep the phone and camera unobstructed.")
        self._latency_tick()

    def _latency_tick(self) -> None:
        if not self._latency_active:
            return
        if self._latency_index < self._latency_total:
            state = "black" if self._latency_index % 2 == 0 else "white"
            token = "transition-{:03d}".format(self._latency_index)
            try:
                presentation = self.phone_target.present_signal(state, token)
                self._latency_events.append(
                    ControlEvent(
                        token=token,
                        state=state,
                        time_ns=presentation.issued_time_ns,
                        metadata={"presentation_revision": presentation.revision},
                    )
                )
                self._latency_index += 1
                self.latency_result.setText("Issued {} of {} alternations…".format(self._latency_index, self._latency_total))
                self._latency_timer.start(self.latency_interval.value())
            except Exception as exc:
                self._latency_active = False
                self._set_status("Latency presentation failed: {}".format(exc), True)
        else:
            QTimer.singleShot(self.latency_interval.value(), self._finish_latency)

    def _finish_latency(self) -> None:
        if not self._latency_active:
            return
        self._latency_active = False
        try:
            result = estimate_latency(
                self._latency_events,
                self._latency_observations,
                probability_threshold=0.75,
                ambiguity_margin=0.30,
                stable_observations=2,
                maximum_latency_ns=max(500_000_000, self.latency_interval.value() * 900_000),
            )
            self.timing = {
                "camera": result,
                "phone_target": dict(self.phone_target.telemetry()),
            }
            self.latency_result.setText(
                "Median {:.1f} ms; P95 {:.1f} ms; jitter {:.1f} ms; accepted {}/{}; confidence {:.1%}.".format(
                    result["median_ns"] / 1.0e6,
                    result["p95_ns"] / 1.0e6,
                    result["robust_jitter_ns"] / 1.0e6,
                    result["accepted_transitions"],
                    result["issued_transitions"],
                    result["confidence"],
                )
            )
            self.image_panes["Latency"].set_bgr(render_latency_timeline(result))
            self.images.setCurrentWidget(self.image_panes["Latency"])
            self._set_status("Latency evidence is ready for review.")
        except Exception as exc:
            self._set_status("Latency evaluation failed: {}".format(exc), True)
        finally:
            try:
                self.phone_target.present_charuco()
            except Exception:
                pass

    def _list_adb(self) -> None:
        try:
            devices = self.adb.devices()
            self.adb_status.setText(
                "Devices: {}".format(", ".join(devices)) if devices else "No authorized ADB devices found."
            )
            if devices and not self.adb_serial.text().strip():
                self.adb_serial.setText(devices[0])
        except Exception as exc:
            self.adb_status.setText("ADB discovery failed: {}".format(exc))

    def _capture_adb(self) -> None:
        try:
            sample = self.adb.capture_screen(self.adb_serial.text().strip() or None)
            self.adb_reference = sample.image.copy()
            self.image_panes["ADB reference"].set_bgr(self.adb_reference)
            self.images.setCurrentWidget(self.image_panes["ADB reference"])
            self.adb_status.setText("Captured {}×{} reference from {}.".format(
                sample.image.shape[1], sample.image.shape[0], sample.source_id
            ))
        except Exception as exc:
            self.adb_status.setText("ADB capture failed: {}".format(exc))

    def _choose_output(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Choose calibration artifact root", self.output_path.text())
        if selected:
            self.output_path.setText(selected)

    def _save(self) -> None:
        if self.analysis is None:
            self._set_status("Fit and review geometry before saving.", True)
            return
        try:
            output = Path(self.output_path.text()) / self.analysis.calibration["calibration_id"]
            self.analysis.calibration["rig"]["phone"]["target_telemetry"] = dict(
                self.phone_target.telemetry()
            )
            yaml_path = save_analysis_bundle(
                output,
                self.analysis,
                image_quality=self.image_quality,
                feature_matching=self.feature_matching,
                timing=self.timing,
                adb_reference=self.adb_reference,
                quality_evidence=self.quality_evidence,
            )
            self.save_result.setText("Saved {}".format(yaml_path))
            self._set_status("Calibration bundle saved. Status remains warning until required evidence gates pass.")
        except Exception as exc:
            self._set_status("Cannot save calibration: {}".format(exc), True)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._quality_active = False
        self._latency_active = False
        self._latency_timer.stop()
        if not self._stop_camera():
            QMessageBox.warning(
                self,
                "Camera still stopping",
                "The camera read has not returned yet. The app will remain open so the adapter is not destroyed while active. Try closing again after it stops.",
            )
            event.ignore()
            return
        try:
            self.phone_target.stop()
        finally:
            self.adb.close()
        event.accept()


def _target_adapter(
    specification: Optional[str], host: str, port: int, advertised_host: Optional[str]
) -> PhoneTargetAdapter:
    adapter = (
        load_adapter_factory(specification)()
        if specification
        else LocalPhoneTargetServer(
            bind_host=host, port=port, advertised_host=advertised_host
        )
    )
    if not isinstance(adapter, PhoneTargetAdapter):
        raise TypeError("Target factory must return PhoneTargetAdapter")
    return adapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-adapter", help="zero-argument module:function factory")
    parser.add_argument("--adb-adapter", help="zero-argument module:function factory")
    parser.add_argument("--target-adapter", help="zero-argument module:function factory")
    parser.add_argument("--adb", default="adb", help="built-in adapter executable")
    parser.add_argument("--target-host", default="0.0.0.0")
    parser.add_argument("--target-port", type=int, default=0)
    parser.add_argument(
        "--target-advertised-host",
        help="LAN host/IP shown to the phone when automatic selection is unsuitable",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts") / "rig_calibrations",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    application = QApplication(sys.argv[:1] + list(argv or []))
    application.setApplicationName("AriaTrace Rig Calibration")
    application.setOrganizationName("AriaTrace")
    application.setStyle("Fusion")
    window = RigCalibrationWindow(
        camera=create_camera_adapter(args.camera_adapter),
        adb=create_adb_adapter(args.adb_adapter, args.adb),
        phone_target=_target_adapter(
            args.target_adapter,
            args.target_host,
            args.target_port,
            args.target_advertised_host,
        ),
        output_root=args.output_root,
    )
    window.show()
    return int(application.exec())


if __name__ == "__main__":
    raise SystemExit(main())
