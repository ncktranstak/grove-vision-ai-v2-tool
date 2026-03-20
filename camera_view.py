"""
Grove Vision AI V2 – Camera View
=================================
Standalone viewer that streams live inference frames from the device
over its USB-CDC serial port using the AT command protocol.

Protocol:
  →  AT+INVOKE=-1,0,0\r        start continuous inference
  ←  \r{"type":1,...,"data":{"image":"<b64>","boxes":[…],"perf":[…]}} \n
  →  AT+BREAK\r                stop
  →  AT+TSCORE=<v>\r           set confidence threshold
  →  AT+TIOU=<v>\r             set IOU threshold
"""

import sys
import os
import json
import base64
import time

import serial
import serial.tools.list_ports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QGroupBox, QDoubleSpinBox,
    QSizePolicy, QFileDialog, QSplitter, QStatusBar, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, QRect, QPoint
from PyQt5.QtGui import (
    QFont, QPalette, QColor, QImage, QPixmap, QPainter, QPen,
    QBrush, QFontMetrics
)

# ---------------------------------------------------------------------------
# Constants – ported directly from Himax web toolkit JS
# ---------------------------------------------------------------------------

BOX_COLORS = [
    "#FF0000","#FFA500","#FFFF00","#32CD32","#006400","#4169E1","#0000FF",
    "#FF1493","#FFC0CB","#800080","#FFD700","#9ACD32","#ADFF2F","#00FFFF",
    "#1E90FF","#FF4500","#CD853F","#FF8C00","#FF6347","#8B4513","#FF69B4",
    "#FF00FF","#BA55D3","#9400D3","#8A2BE2","#4682B4","#87CEEB","#00CED1",
    "#20B2AA","#FFB6C1","#696969","#808080","#A9A9A9","#C0C0C0","#D3D3D3",
    "#FFFAFA","#F0FFF0","#F5F5DC","#FFE4C4","#FFDAB9","#EEE8AA","#F0E68C",
    "#BDB76B","#FFD700","#F5DEB3","#D2B48C","#DEB887","#BC8F8F","#F4A460",
    "#DAA520","#CD853F","#A52A2A","#8B4513","#D2691E","#B22222","#FF6347",
    "#FF4500","#FF8C00","#FFA07A","#FA8072","#E9967A","#FF69B4","#FF1493",
    "#DB7093","#C71585",
]

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]

PEOPLENET_CLASSES = ["person", "bag", "face"]
GENDER_CLASSES    = ["Female", "Male"]


def _color(class_id: int) -> QColor:
    return QColor(BOX_COLORS[class_id % len(BOX_COLORS)])


def _class_name(boxes_key: str, class_id: int, custom: list) -> str:
    if custom and class_id < len(custom):
        return custom[class_id]
    if boxes_key == "boxes":
        return COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id)
    if boxes_key == "peoplenet_boxes":
        return PEOPLENET_CLASSES[class_id] if class_id < len(PEOPLENET_CLASSES) else str(class_id)
    if boxes_key == "gender_cls_boxes":
        return GENDER_CLASSES[class_id] if class_id < len(GENDER_CLASSES) else str(class_id)
    return str(class_id)


# ---------------------------------------------------------------------------
# Serial / camera worker thread
# ---------------------------------------------------------------------------

class CameraWorker(QThread):
    frame_ready = pyqtSignal(QImage, dict)   # (decoded image, raw data dict)
    fps_update  = pyqtSignal(float)
    status      = pyqtSignal(str)
    error       = pyqtSignal(str)

    _CMD_INVOKE = b"AT+INVOKE=-1,0,0\r"
    _CMD_BREAK  = b"AT+BREAK\r"

    def __init__(self, port: str, baudrate: int):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self._running = False
        self._ser: serial.Serial | None = None

    # ── send helpers (called from main thread via queued connection) ────────

    def send_raw(self, cmd: bytes):
        if self._ser and self._ser.is_open:
            try:
                self._ser.write(cmd)
                self._ser.flush()
            except Exception:
                pass

    # ── thread body ─────────────────────────────────────────────────────────

    def run(self):
        try:
            self._ser = serial.Serial(self.port, self.baudrate, timeout=0.05)
        except serial.SerialException as exc:
            self.error.emit(str(exc))
            return

        self._running = True

        # On Windows, opening the serial port asserts DTR which resets the device.
        # Ping AT+ID? until the device responds before sending AT+INVOKE.
        self.status.emit("Waiting for device…")
        ready = False
        for _ in range(15):
            if not self._running:
                break
            self._ser.reset_input_buffer()
            self._ser.write(b"AT+ID?\r")
            self._ser.flush()
            time.sleep(0.5)
            resp = self._ser.read(self._ser.in_waiting or 0)
            if b'"ID?"' in resp or b'"name"' in resp:
                ready = True
                break

        if not ready or not self._running:
            if self._ser and self._ser.is_open:
                try:
                    self._ser.close()
                except Exception:
                    pass
            if self._running:
                self.error.emit("Device did not respond to AT+ID? (check connection)")
            return

        self.status.emit("Device ready — starting inference…")
        self._ser.reset_input_buffer()
        self._ser.write(self._CMD_INVOKE)
        self._ser.flush()

        buf = bytearray()
        in_json = False
        t_prev = time.perf_counter()
        fps_acc = fps_n = 0

        while self._running:
            chunk = self._ser.read(self._ser.in_waiting or 1)
            if chunk:
                buf.extend(chunk)

            while True:
                if not in_json:
                    idx = buf.find(b"\r{")
                    if idx == -1:
                        buf = buf[-1:] if buf else buf
                        break
                    del buf[:idx + 1]   # keep '{'
                    in_json = True

                end = buf.find(b"}\n")
                if end == -1:
                    break

                json_bytes = bytes(buf[:end + 1])
                del buf[:end + 2]
                in_json = False

                try:
                    obj = json.loads(json_bytes)
                except json.JSONDecodeError:
                    continue

                data = obj.get("data", {})
                b64 = data.get("image")
                if not b64:
                    continue

                try:
                    jpg = base64.b64decode(b64)
                except Exception:
                    continue

                img = QImage()
                if img.loadFromData(jpg, "JPEG") and not img.isNull():
                    self.frame_ready.emit(img, data)
                    t_now = time.perf_counter()
                    fps_acc += 1.0 / max(t_now - t_prev, 1e-6)
                    fps_n += 1
                    t_prev = t_now
                    if fps_n >= 10:
                        self.fps_update.emit(fps_acc / fps_n)
                        fps_acc = fps_n = 0

        if self._ser and self._ser.is_open:
            try:
                self._ser.write(self._CMD_BREAK)
                self._ser.flush()
            except Exception:
                pass
            self._ser.close()

    def stop(self):
        self._running = False
        self.wait(3000)


# ---------------------------------------------------------------------------
# Video canvas – draws the frame + detection overlays via QPainter
# ---------------------------------------------------------------------------

class VideoCanvas(QWidget):
    """
    Renders decoded inference frames with bounding-box overlays.
    Coordinate scaling matches the Himax web toolkit:
      - scale = 3  when image width < 640 or height < 480 (small sensor output)
      - scale = 1  otherwise
    """

    LABEL_H = 20
    FONT    = QFont("Arial", 10, QFont.Bold)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #0d0d0d;")
        self._placeholder = True
        self.score_thresh: float = 0.50
        self.custom_classes: list = []

    # ── public API ──────────────────────────────────────────────────────────

    def update_frame(self, img: QImage, data: dict):
        self._placeholder = False
        # Determine coordinate scale factor (matches web toolkit logic)
        sw = img.width()
        sh = img.height()
        scale = 3 if (sw < 640 or sh < 480) else 1

        # Create a working pixmap at the DRAWN size (not the canvas-element size)
        draw_w = sw * scale
        draw_h = sh * scale
        pix = QPixmap(draw_w, draw_h)
        pix.fill(Qt.black)

        p = QPainter(pix)
        # Draw base image scaled up
        p.drawImage(QRect(0, 0, draw_w, draw_h), img)

        # Draw all box types
        for key in ("boxes", "peoplenet_boxes", "fm_face_boxes",
                    "gender_cls_boxes", "keypoints"):
            raw = data.get(key)
            if not raw:
                continue
            for box in raw:
                coords = box if key != "keypoints" else box[0]
                if len(coords) < 6:
                    continue
                x = coords[0] * scale
                y = coords[1] * scale
                w = coords[2] * scale
                h = coords[3] * scale
                score    = coords[4]
                class_id = int(coords[5])
                if score < self.score_thresh:
                    continue
                name  = _class_name(key, class_id, self.custom_classes)
                color = _color(class_id)
                self._draw_box(p, x, y, w, h, score, name, color)

        # Draw classifications
        classes = data.get("classes")
        if classes:
            self._draw_classes(p, classes, draw_w, draw_h)

        p.end()
        self._pixmap = pix
        self.update()

    def clear(self):
        self._pixmap = None
        self._placeholder = True
        self.update()

    # ── Qt overrides ────────────────────────────────────────────────────────

    def paintEvent(self, _):
        p = QPainter(self)
        if self._placeholder or self._pixmap is None:
            p.fillRect(self.rect(), QColor("#0d0d0d"))
            p.setPen(QColor("#555"))
            p.setFont(QFont("Consolas", 11))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "Select a COM port and press  ▶  Start")
            return

        # Scale pixmap to fit widget, maintaining aspect ratio
        scaled = self._pixmap.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        x_off = (self.width()  - scaled.width())  // 2
        y_off = (self.height() - scaled.height()) // 2
        p.drawPixmap(x_off, y_off, scaled)

    # ── private helpers ─────────────────────────────────────────────────────

    def _draw_box(self, p: QPainter, x, y, w, h, score, name, color: QColor):
        p.setPen(QPen(color, 2))
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRect(int(x), int(y), int(w), int(h)))

        # Label background
        label = f"{name}: {score:.2f}"
        fm = QFontMetrics(self.FONT)
        lw = fm.horizontalAdvance(label) + 8
        p.fillRect(QRect(int(x), int(y), lw, self.LABEL_H), color)

        # Label text
        p.setFont(self.FONT)
        p.setPen(QColor("#000000") if color.lightness() > 128 else QColor("#ffffff"))
        p.drawText(QPoint(int(x) + 4, int(y) + 14), label)

    def _draw_classes(self, p: QPainter, classes, img_w, img_h):
        n = len(classes)
        if n == 0:
            return
        slot_w = img_w // n
        for i, (value, class_id) in enumerate(classes):
            name = _class_name("classes", class_id, self.custom_classes)
            color = _color(class_id)
            bar_h = img_h // 10
            p.setOpacity(0.3)
            p.fillRect(QRect(slot_w * i, 0, slot_w, bar_h), color)
            p.setOpacity(1.0)
            font = QFont("Arial", max(8, img_h // 16), QFont.Bold)
            p.setFont(font)
            p.setPen(QColor("#ffffff"))
            p.drawText(QPoint(slot_w * i + 4, img_h // 16), f"{name}: {value}")


# ---------------------------------------------------------------------------
# Control panel (left sidebar)
# ---------------------------------------------------------------------------

class ControlPanel(QWidget):
    connect_requested    = pyqtSignal(str, int)
    disconnect_requested = pyqtSignal()
    score_changed        = pyqtSignal(float)
    iou_changed          = pyqtSignal(float)
    snapshot_requested   = pyqtSignal()

    BAUDRATES = ["115200", "230400", "460800", "921600"]

    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._connected = False
        self.setFixedWidth(220)
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(8, 8, 8, 8)

        # Logo / title
        title = QLabel("Grove Vision AI V2")
        title.setFont(QFont("Arial", 11, QFont.Bold))
        title.setStyleSheet("color: #6ec6ff;")
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #444;")
        root.addWidget(sep)

        # ── Connection ──────────────────────────────────────────────────
        conn_grp = QGroupBox("Connection")
        conn_lay = QVBoxLayout(conn_grp)
        conn_lay.setSpacing(6)

        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("Port:"))
        self.port_combo = QComboBox()
        port_row.addWidget(self.port_combo)
        refresh_btn = QPushButton("⟳")
        refresh_btn.setFixedWidth(28)
        refresh_btn.setToolTip("Rescan")
        refresh_btn.clicked.connect(self._refresh_ports)
        port_row.addWidget(refresh_btn)
        conn_lay.addLayout(port_row)

        baud_row = QHBoxLayout()
        baud_row.addWidget(QLabel("Baud:"))
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(self.BAUDRATES)
        self.baud_combo.setCurrentText("921600")
        baud_row.addWidget(self.baud_combo)
        conn_lay.addLayout(baud_row)

        self.connect_btn = QPushButton("▶  Start")
        self.connect_btn.setFixedHeight(34)
        self.connect_btn.setStyleSheet(
            "background-color: #2d6a4f; color: white; font-weight: bold;"
        )
        self.connect_btn.clicked.connect(self._toggle_connect)
        conn_lay.addWidget(self.connect_btn)

        root.addWidget(conn_grp)
        self._refresh_ports()

        # ── Inference settings ──────────────────────────────────────────
        inf_grp = QGroupBox("Inference")
        inf_lay = QVBoxLayout(inf_grp)
        inf_lay.setSpacing(6)

        score_row = QHBoxLayout()
        score_row.addWidget(QLabel("Confidence:"))
        self.score_spin = QDoubleSpinBox()
        self.score_spin.setRange(0.01, 1.0)
        self.score_spin.setSingleStep(0.05)
        self.score_spin.setValue(0.50)
        self.score_spin.setDecimals(2)
        self.score_spin.valueChanged.connect(
            lambda v: self.score_changed.emit(v)
        )
        score_row.addWidget(self.score_spin)
        inf_lay.addLayout(score_row)

        iou_row = QHBoxLayout()
        iou_row.addWidget(QLabel("IOU:"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setDecimals(2)
        self.iou_spin.valueChanged.connect(
            lambda v: self.iou_changed.emit(v)
        )
        iou_row.addWidget(self.iou_spin)
        inf_lay.addLayout(iou_row)

        root.addWidget(inf_grp)

        # ── Actions ─────────────────────────────────────────────────────
        act_grp = QGroupBox("Actions")
        act_lay = QVBoxLayout(act_grp)

        self.snap_btn = QPushButton("📷  Snapshot")
        self.snap_btn.setFixedHeight(30)
        self.snap_btn.setEnabled(False)
        self.snap_btn.clicked.connect(self.snapshot_requested)
        act_lay.addWidget(self.snap_btn)

        root.addWidget(act_grp)

        # ── Stats ────────────────────────────────────────────────────────
        self.stats_grp = QGroupBox("Stats")
        stats_lay = QVBoxLayout(self.stats_grp)
        self.fps_lbl   = QLabel("FPS: —")
        self.pre_lbl   = QLabel("Pre:  — ms")
        self.inf_lbl   = QLabel("Infer: — ms")
        self.post_lbl  = QLabel("Post: — ms")
        self.det_lbl   = QLabel("Detections: —")
        for lbl in (self.fps_lbl, self.pre_lbl, self.inf_lbl,
                    self.post_lbl, self.det_lbl):
            lbl.setFont(QFont("Consolas", 9))
            stats_lay.addWidget(lbl)

        root.addWidget(self.stats_grp)
        root.addStretch()

    def _load_settings(self):
        saved_port = self.settings.value("camview/port", "")
        idx = self.port_combo.findText(saved_port)
        if idx >= 0:
            self.port_combo.setCurrentIndex(idx)
        saved_baud = self.settings.value("camview/baud", "921600")
        bi = self.baud_combo.findText(saved_baud)
        if bi >= 0:
            self.baud_combo.setCurrentIndex(bi)
        self.score_spin.setValue(float(self.settings.value("camview/score", 0.50)))
        self.iou_spin.setValue(float(self.settings.value("camview/iou", 0.45)))

    def save_settings(self):
        self.settings.setValue("camview/port",  self.port_combo.currentText())
        self.settings.setValue("camview/baud",  self.baud_combo.currentText())
        self.settings.setValue("camview/score", self.score_spin.value())
        self.settings.setValue("camview/iou",   self.iou_spin.value())

    def _refresh_ports(self):
        current = self.port_combo.currentText()
        self.port_combo.clear()
        self.port_combo.addItems(
            [p.device for p in serial.tools.list_ports.comports()]
        )
        idx = self.port_combo.findText(current)
        if idx >= 0:
            self.port_combo.setCurrentIndex(idx)

    def _toggle_connect(self):
        if self._connected:
            self.disconnect_requested.emit()
        else:
            port = self.port_combo.currentText()
            baud = int(self.baud_combo.currentText())
            if port:
                self.save_settings()
                self.connect_requested.emit(port, baud)

    def set_connected(self, connected: bool):
        self._connected = connected
        if connected:
            self.connect_btn.setText("■  Stop")
            self.connect_btn.setStyleSheet(
                "background-color: #d62828; color: white; font-weight: bold;"
            )
            self.port_combo.setEnabled(False)
            self.baud_combo.setEnabled(False)
            self.snap_btn.setEnabled(True)
        else:
            self.connect_btn.setText("▶  Start")
            self.connect_btn.setStyleSheet(
                "background-color: #2d6a4f; color: white; font-weight: bold;"
            )
            self.port_combo.setEnabled(True)
            self.baud_combo.setEnabled(True)
            self.snap_btn.setEnabled(False)
            self._reset_stats()

    def update_fps(self, fps: float):
        self.fps_lbl.setText(f"FPS: {fps:.1f}")

    def update_stats(self, data: dict):
        perf = data.get("perf")
        if perf and len(perf) >= 3:
            self.pre_lbl.setText(f"Pre:  {perf[0]} ms")
            self.inf_lbl.setText(f"Infer: {perf[1]} ms")
            self.post_lbl.setText(f"Post: {perf[2]} ms")
        boxes = (data.get("boxes") or data.get("peoplenet_boxes") or
                 data.get("fm_face_boxes") or data.get("gender_cls_boxes") or [])
        self.det_lbl.setText(f"Detections: {len(boxes)}")

    def _reset_stats(self):
        for lbl, text in (
            (self.fps_lbl,  "FPS: —"),
            (self.pre_lbl,  "Pre:  — ms"),
            (self.inf_lbl,  "Infer: — ms"),
            (self.post_lbl, "Post: — ms"),
            (self.det_lbl,  "Detections: —"),
        ):
            lbl.setText(text)

    @property
    def score_value(self) -> float:
        return self.score_spin.value()

    @property
    def iou_value(self) -> float:
        return self.iou_spin.value()


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("GroveTool", "CameraView")
        self._worker: CameraWorker | None = None
        self._last_frame: QImage | None = None
        self._setup_ui()
        self._restore_geometry()

    def _setup_ui(self):
        self.setWindowTitle("Grove Vision AI V2 – Camera View")
        self.resize(1100, 680)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        self.setCentralWidget(central)

        # Left: control panel
        self.ctrl = ControlPanel(self.settings)
        self.ctrl.connect_requested.connect(self._start)
        self.ctrl.disconnect_requested.connect(self._stop)
        self.ctrl.score_changed.connect(self._on_score_changed)
        self.ctrl.iou_changed.connect(self._on_iou_changed)
        self.ctrl.snapshot_requested.connect(self._snapshot)
        layout.addWidget(self.ctrl)

        # Right: video canvas
        self.canvas = VideoCanvas()
        self.canvas.score_thresh = self.ctrl.score_value
        layout.addWidget(self.canvas, stretch=1)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — select a COM port and press Start")

    # ── connection lifecycle ─────────────────────────────────────────────

    def _start(self, port: str, baud: int):
        self._worker = CameraWorker(port, baud)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.fps_update.connect(self.ctrl.update_fps)
        self._worker.status.connect(self.status.showMessage)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        self.ctrl.set_connected(True)
        self.status.showMessage(f"Connecting — {port} @ {baud}…")

    def _stop(self):
        if self._worker:
            self._worker.stop()
            self._worker = None
        self.ctrl.set_connected(False)
        self.canvas.clear()
        self.status.showMessage("Disconnected")

    def _on_error(self, msg: str):
        self._stop()
        self.status.showMessage(f"Error: {msg}")

    # ── frame / data updates ─────────────────────────────────────────────

    def _on_frame(self, img: QImage, data: dict):
        self._last_frame = img
        self.canvas.update_frame(img, data)
        self.ctrl.update_stats(data)

    # ── settings commands ────────────────────────────────────────────────

    def _on_score_changed(self, v: float):
        self.canvas.score_thresh = v
        if self._worker:
            cmd = f"AT+TSCORE={v:.2f}\r".encode()
            self._worker.send_raw(cmd)

    def _on_iou_changed(self, v: float):
        if self._worker:
            cmd = f"AT+TIOU={v:.2f}\r".encode()
            self._worker.send_raw(cmd)

    # ── snapshot ────────────────────────────────────────────────────────

    def _snapshot(self):
        if not self._last_frame:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save snapshot", "snapshot.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if path:
            self._last_frame.save(path)
            self.status.showMessage(f"Snapshot saved → {path}")

    # ── window lifecycle ─────────────────────────────────────────────────

    def _restore_geometry(self):
        geo = self.settings.value("camview/geometry")
        if geo:
            self.restoreGeometry(geo)

    def closeEvent(self, event):
        self.settings.setValue("camview/geometry", self.saveGeometry())
        self.ctrl.save_settings()
        if self._worker and self._worker.isRunning():
            self._worker.stop()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Dark palette
# ---------------------------------------------------------------------------

def apply_dark_palette(app: QApplication):
    palette = QPalette()
    roles = {
        QPalette.Window:          "#1e1e1e",
        QPalette.WindowText:      "#f0f0f0",
        QPalette.Base:            "#141414",
        QPalette.AlternateBase:   "#1e1e1e",
        QPalette.ToolTipBase:     "#141414",
        QPalette.ToolTipText:     "#f0f0f0",
        QPalette.Text:            "#f0f0f0",
        QPalette.Button:          "#2d2d2d",
        QPalette.ButtonText:      "#f0f0f0",
        QPalette.BrightText:      "#ff5555",
        QPalette.Link:            "#6ec6ff",
        QPalette.Highlight:       "#264f78",
        QPalette.HighlightedText: "#ffffff",
    }
    for role, hex_color in roles.items():
        color = QColor(hex_color)
        palette.setColor(role, color)
        palette.setColor(QPalette.Disabled, role, color.darker(160))
    app.setPalette(palette)
    app.setStyleSheet("""
        QGroupBox {
            border: 1px solid #3a3a3a;
            border-radius: 4px;
            margin-top: 6px;
            padding-top: 6px;
            font-weight: bold;
            color: #aaa;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
        }
        QComboBox, QDoubleSpinBox, QLineEdit {
            background: #141414;
            border: 1px solid #3a3a3a;
            border-radius: 3px;
            padding: 2px 4px;
            color: #f0f0f0;
        }
        QPushButton {
            background: #2d2d2d;
            border: 1px solid #3a3a3a;
            border-radius: 3px;
            padding: 4px 10px;
            color: #f0f0f0;
        }
        QPushButton:hover  { background: #3a3a3a; }
        QPushButton:pressed{ background: #1a1a1a; }
        QStatusBar { color: #888; font-size: 11px; }
    """)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    apply_dark_palette(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
