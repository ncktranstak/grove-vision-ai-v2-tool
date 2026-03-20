import sys
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import serial
import serial.tools.list_ports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QLineEdit, QPushButton, QComboBox, QFileDialog,
    QTextEdit, QGroupBox, QFormLayout, QProgressBar, QSizePolicy,
    QSplitter, QCheckBox, QStatusBar, QSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QTextCursor, QImage, QPixmap

# ---------------------------------------------------------------------------
# Shared worker thread base
# ---------------------------------------------------------------------------

class CommandWorker(QThread):
    output = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, cmd, cwd=None, env=None):
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd
        self.env = env

    def run(self):
        try:
            proc = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                shell=True,
            )
            for line in proc.stdout:
                self.output.emit(line.rstrip())
            proc.wait()
            if proc.returncode == 0:
                self.finished.emit(True, "Command completed successfully.")
            else:
                self.finished.emit(False, f"Command exited with code {proc.returncode}.")
        except Exception as exc:
            self.finished.emit(False, str(exc))


# ---------------------------------------------------------------------------
# Serial reader thread
# ---------------------------------------------------------------------------

class SerialReader(QThread):
    data_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, port, baudrate):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self._running = False
        self._serial = None

    def run(self):
        try:
            self._serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self._running = True
            while self._running:
                if self._serial.in_waiting:
                    raw = self._serial.read(self._serial.in_waiting)
                    try:
                        text = raw.decode("utf-8", errors="replace")
                    except Exception:
                        text = repr(raw)
                    self.data_received.emit(text)
                self.msleep(10)
        except serial.SerialException as exc:
            self.error_occurred.emit(str(exc))
        finally:
            if self._serial and self._serial.is_open:
                self._serial.close()

    def send(self, data: str):
        if self._serial and self._serial.is_open:
            self._serial.write(data.encode("utf-8", errors="replace"))

    def stop(self):
        self._running = False
        self.wait(2000)


# ---------------------------------------------------------------------------
# Reusable path row widget
# ---------------------------------------------------------------------------

class PathRow(QWidget):
    def __init__(self, label, mode="file", filter_str="", placeholder="", parent=None):
        super().__init__(parent)
        self._mode = mode
        self._filter = filter_str
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.lbl = QLabel(label)
        self.lbl.setFixedWidth(140)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.btn = QPushButton("Browse…")
        self.btn.setFixedWidth(80)
        self.btn.clicked.connect(self._browse)
        layout.addWidget(self.lbl)
        layout.addWidget(self.edit)
        layout.addWidget(self.btn)

    def _browse(self):
        if self._mode == "dir":
            path = QFileDialog.getExistingDirectory(self, "Select folder")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select file", filter=self._filter)
        if path:
            self.edit.setText(path)

    @property
    def text(self):
        return self.edit.text().strip()

    @text.setter
    def text(self, val):
        self.edit.setText(val)


# ---------------------------------------------------------------------------
# Tab 1 – Build Firmware
# ---------------------------------------------------------------------------

class BuildTab(QWidget):
    log_message = pyqtSignal(str)

    SCENARIOS = [
        "tflm_fd_fm",
        "tflm_yolov8_od",
        "tflm_yolov8_pose",
        "tflm_yolov8_gender_cls",
        "pdm_record",
        "kws_pdm_record",
        "imu_read",
        "tflm_peoplenet",
        "tflm_yolo11_od",
        "tflm_mb_cls",
        "torch_mb_cls",
    ]

    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._worker = None
        self._build_env = None
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)

        # Paths group
        paths_grp = QGroupBox("Paths")
        paths_lay = QVBoxLayout(paths_grp)

        self.repo_row = PathRow("Repo directory:", mode="dir",
                                placeholder="/path/to/Seeed_Grove_Vision_AI_Module_V2")
        self.toolchain_row = PathRow("ARM toolchain:", mode="dir",
                                     placeholder="/path/to/arm-gnu-toolchain/bin")
        paths_lay.addWidget(self.repo_row)
        paths_lay.addWidget(self.toolchain_row)
        root.addWidget(paths_grp)

        # Scenario group
        scen_grp = QGroupBox("Scenario App")
        scen_lay = QHBoxLayout(scen_grp)
        scen_lay.addWidget(QLabel("App:"))
        self.scenario_combo = QComboBox()
        self.scenario_combo.addItems(self.SCENARIOS)
        scen_lay.addWidget(self.scenario_combo)
        scen_lay.addStretch()
        root.addWidget(scen_grp)

        # Actions
        btn_row = QHBoxLayout()
        self.clean_btn = QPushButton("Clean")
        self.clean_btn.setFixedWidth(100)
        self.build_btn = QPushButton("Build")
        self.build_btn.setFixedWidth(100)
        self.build_btn.setStyleSheet("background-color: #2d6a4f; color: white;")
        self.clean_btn.clicked.connect(self._run_clean)
        self.build_btn.clicked.connect(self._run_build)
        btn_row.addWidget(self.clean_btn)
        btn_row.addWidget(self.build_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        # Output ELF display
        elf_grp = QGroupBox("Output ELF")
        elf_lay = QHBoxLayout(elf_grp)
        self.elf_label = QLabel("—")
        self.elf_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.elf_label.setStyleSheet("color: #6ec6ff;")
        elf_lay.addWidget(self.elf_label)
        root.addWidget(elf_grp)

        root.addStretch()

    def _load_settings(self):
        self.repo_row.text = self.settings.value("build/repo_path", "")
        self.toolchain_row.text = self.settings.value("build/toolchain_path", "")
        idx = self.scenario_combo.findText(self.settings.value("build/scenario", "tflm_yolov8_od"))
        if idx >= 0:
            self.scenario_combo.setCurrentIndex(idx)

    def _save_settings(self):
        self.settings.setValue("build/repo_path", self.repo_row.text)
        self.settings.setValue("build/toolchain_path", self.toolchain_row.text)
        self.settings.setValue("build/scenario", self.scenario_combo.currentText())

    def _make_env(self):
        env = os.environ.copy()
        tc = self.toolchain_row.text
        if tc:
            env["PATH"] = tc + os.pathsep + env.get("PATH", "")
        return env

    def _set_busy(self, busy: bool):
        self.clean_btn.setEnabled(not busy)
        self.build_btn.setEnabled(not busy)

    def _run_clean(self):
        repo = self.repo_row.text
        if not repo:
            self.log_message.emit("[Build] ERROR: Repo path is empty.")
            return
        self._save_settings()
        cwd = os.path.join(repo, "EPII_CM55M_APP_S")
        scenario = self.scenario_combo.currentText()
        cmd = f"make clean APP={scenario}"
        self.log_message.emit(f"[Build] Running: {cmd}  (cwd: {cwd})")
        self._launch(cmd, cwd)

    def _run_build(self):
        repo = self.repo_row.text
        if not repo:
            self.log_message.emit("[Build] ERROR: Repo path is empty.")
            return
        self._save_settings()
        cwd = os.path.join(repo, "EPII_CM55M_APP_S")
        scenario = self.scenario_combo.currentText()
        cmd = f"make APP={scenario}"
        self.log_message.emit(f"[Build] Running: {cmd}  (cwd: {cwd})")
        self._launch(cmd, cwd)

    def _launch(self, cmd, cwd):
        self._set_busy(True)
        self._worker = CommandWorker(cmd, cwd=cwd, env=self._make_env())
        self._worker.output.connect(lambda line: self.log_message.emit(line))
        self._worker.finished.connect(self._on_done)
        self._worker.start()

    def _on_done(self, success: bool, msg: str):
        self._set_busy(False)
        self.log_message.emit(f"[Build] {'OK' if success else 'FAIL'}: {msg}")
        if success:
            repo = self.repo_row.text
            scenario = self.scenario_combo.currentText()
            elf = (
                f"{repo}/EPII_CM55M_APP_S/obj_epii_evb_icv30_bdv10/"
                f"gnu_epii_evb_WLCSP65/EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf"
            )
            self.elf_label.setText(os.path.normpath(elf))
            self.settings.setValue("last_elf", os.path.normpath(elf))


# ---------------------------------------------------------------------------
# Tab 2 – Generate Image
# ---------------------------------------------------------------------------

class GenerateTab(QWidget):
    log_message = pyqtSignal(str)

    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._worker = None
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)

        grp = QGroupBox("Image Generation")
        lay = QVBoxLayout(grp)

        self.elf_row = PathRow("ELF file:", mode="file",
                               filter_str="ELF files (*.elf);;All files (*)",
                               placeholder="EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf")
        self.repo_row = PathRow("Repo directory:", mode="dir",
                                placeholder="/path/to/Seeed_Grove_Vision_AI_Module_V2")

        lay.addWidget(self.elf_row)
        lay.addWidget(self.repo_row)
        root.addWidget(grp)

        btn_row = QHBoxLayout()
        self.gen_btn = QPushButton("Generate Image")
        self.gen_btn.setFixedWidth(140)
        self.gen_btn.setStyleSheet("background-color: #2d6a4f; color: white;")
        self.gen_btn.clicked.connect(self._run_generate)
        btn_row.addWidget(self.gen_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        out_grp = QGroupBox("Output Image")
        out_lay = QHBoxLayout(out_grp)
        self.img_label = QLabel("—")
        self.img_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.img_label.setStyleSheet("color: #6ec6ff;")
        out_lay.addWidget(self.img_label)
        root.addWidget(out_grp)

        root.addStretch()

    def _load_settings(self):
        self.elf_row.text = self.settings.value("last_elf", "")
        self.repo_row.text = self.settings.value("build/repo_path", "")

    def _save_settings(self):
        self.settings.setValue("last_elf", self.elf_row.text)
        self.settings.setValue("build/repo_path", self.repo_row.text)

    def _run_generate(self):
        elf_path = self.elf_row.text
        repo = self.repo_row.text
        if not elf_path or not repo:
            self.log_message.emit("[Gen] ERROR: ELF path and repo path are required.")
            return
        self._save_settings()

        gen_dir = os.path.join(repo, "we2_image_gen_local")
        dest_dir = os.path.join(gen_dir, "input_case1_secboot")
        os.makedirs(dest_dir, exist_ok=True)

        dest_elf = os.path.join(dest_dir, os.path.basename(elf_path))
        try:
            shutil.copy2(elf_path, dest_elf)
            self.log_message.emit(f"[Gen] Copied ELF → {dest_elf}")
        except Exception as exc:
            self.log_message.emit(f"[Gen] ERROR copying ELF: {exc}")
            return

        gen_exe = os.path.join(gen_dir, "we2_local_image_gen.exe")
        if not os.path.isfile(gen_exe):
            self.log_message.emit(f"[Gen] ERROR: we2_local_image_gen.exe not found at {gen_exe}")
            return

        cmd = f'"{gen_exe}" project_case1_blp_wlcsp.json'
        self.log_message.emit(f"[Gen] Running: {cmd}")
        self.gen_btn.setEnabled(False)
        self._worker = CommandWorker(cmd, cwd=gen_dir)
        self._worker.output.connect(lambda l: self.log_message.emit(l))
        self._worker.finished.connect(self._on_done)
        self._worker.start()

    def _on_done(self, success, msg):
        self.gen_btn.setEnabled(True)
        self.log_message.emit(f"[Gen] {'OK' if success else 'FAIL'}: {msg}")
        if success:
            repo = self.repo_row.text
            img = os.path.normpath(
                os.path.join(repo, "we2_image_gen_local",
                             "output_case1_sec_wlcsp", "output.img"))
            self.img_label.setText(img)
            self.settings.setValue("last_img", img)


# ---------------------------------------------------------------------------
# Tab 3 – Flash Firmware
# ---------------------------------------------------------------------------

class FlashTab(QWidget):
    log_message = pyqtSignal(str)

    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._worker = None
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)

        # Connection group
        conn_grp = QGroupBox("Serial Port")
        conn_lay = QHBoxLayout(conn_grp)
        conn_lay.addWidget(QLabel("Port:"))
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(120)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedWidth(70)
        refresh_btn.clicked.connect(self._refresh_ports)
        conn_lay.addWidget(self.port_combo)
        conn_lay.addWidget(refresh_btn)
        conn_lay.addWidget(QLabel("Baud: 921600 (fixed)"))
        conn_lay.addStretch()
        root.addWidget(conn_grp)
        self._refresh_ports()

        # Firmware group
        fw_grp = QGroupBox("Firmware File (.img)")
        fw_lay = QVBoxLayout(fw_grp)
        self.fw_row = PathRow("Firmware:", mode="file",
                              filter_str="Image files (*.img);;All files (*)",
                              placeholder="output.img")
        fw_lay.addWidget(self.fw_row)
        root.addWidget(fw_grp)

        # Optional model group
        model_grp = QGroupBox("Optional: Flash TFLite Model")
        model_lay = QVBoxLayout(model_grp)
        self.model_check = QCheckBox("Include model")
        self.model_check.toggled.connect(self._toggle_model)
        model_lay.addWidget(self.model_check)

        self.model_row = PathRow("TFLite model:", mode="file",
                                 filter_str="TFLite (*.tflite);;All files (*)")
        self.model_row.setEnabled(False)
        model_lay.addWidget(self.model_row)

        pos_row = QHBoxLayout()
        pos_row.addWidget(QLabel("Flash position:"))
        self.flash_pos = QLineEdit("0x000A0000")
        self.flash_pos.setFixedWidth(120)
        self.flash_pos.setEnabled(False)
        pos_row.addWidget(self.flash_pos)
        pos_row.addWidget(QLabel("Offset:"))
        self.flash_offset = QLineEdit("0x00000000")
        self.flash_offset.setFixedWidth(120)
        self.flash_offset.setEnabled(False)
        pos_row.addWidget(self.flash_offset)
        pos_row.addStretch()
        model_lay.addLayout(pos_row)
        root.addWidget(model_grp)

        # Repo path for xmodem script
        xmodem_grp = QGroupBox("xmodem Script Location")
        xmodem_lay = QVBoxLayout(xmodem_grp)
        self.xmodem_row = PathRow("Repo directory:", mode="dir",
                                  placeholder="/path/to/Seeed_Grove_Vision_AI_Module_V2")
        xmodem_lay.addWidget(self.xmodem_row)
        root.addWidget(xmodem_grp)

        # Progress + button
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        btn_row = QHBoxLayout()
        self.flash_btn = QPushButton("Flash Firmware")
        self.flash_btn.setFixedWidth(140)
        self.flash_btn.setStyleSheet("background-color: #d62828; color: white;")
        self.flash_btn.clicked.connect(self._run_flash)
        self.status_lbl = QLabel("Ready.")
        btn_row.addWidget(self.flash_btn)
        btn_row.addWidget(self.status_lbl)
        btn_row.addStretch()
        root.addLayout(btn_row)

        root.addStretch()

    def _load_settings(self):
        self.fw_row.text = self.settings.value("last_img", "")
        self.xmodem_row.text = self.settings.value("build/repo_path", "")
        saved_port = self.settings.value("flash/port", "")
        idx = self.port_combo.findText(saved_port)
        if idx >= 0:
            self.port_combo.setCurrentIndex(idx)

    def _save_settings(self):
        self.settings.setValue("last_img", self.fw_row.text)
        self.settings.setValue("build/repo_path", self.xmodem_row.text)
        self.settings.setValue("flash/port", self.port_combo.currentText())

    def _refresh_ports(self):
        current = self.port_combo.currentText()
        self.port_combo.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo.addItems(ports)
        idx = self.port_combo.findText(current)
        if idx >= 0:
            self.port_combo.setCurrentIndex(idx)

    def _toggle_model(self, checked):
        self.model_row.setEnabled(checked)
        self.flash_pos.setEnabled(checked)
        self.flash_offset.setEnabled(checked)

    def _run_flash(self):
        port = self.port_combo.currentText()
        fw = self.fw_row.text
        repo = self.xmodem_row.text
        if not port or not fw or not repo:
            self.log_message.emit("[Flash] ERROR: Port, firmware file and repo path are required.")
            return
        self._save_settings()

        xmodem_script = os.path.join(repo, "xmodem", "xmodem_send.py")
        cmd = (
            f'python "{xmodem_script}" --port={port} --baudrate=921600 '
            f'--protocol=xmodem --file="{fw}"'
        )

        if self.model_check.isChecked():
            model_path = self.model_row.text
            pos = self.flash_pos.text().strip()
            offset = self.flash_offset.text().strip()
            if model_path:
                cmd += f' --model="{model_path} {pos} {offset}"'

        self.log_message.emit(f"[Flash] Running: {cmd}")
        self.flash_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.status_lbl.setText("Flashing…")
        self._worker = CommandWorker(cmd)
        self._worker.output.connect(lambda l: self.log_message.emit(l))
        self._worker.finished.connect(self._on_done)
        self._worker.start()

    def _on_done(self, success, msg):
        self.flash_btn.setEnabled(True)
        self.progress.setVisible(False)
        status = "Done — press RESET on device." if success else f"Failed: {msg}"
        self.status_lbl.setText(status)
        self.log_message.emit(f"[Flash] {'OK' if success else 'FAIL'}: {msg}")


# ---------------------------------------------------------------------------
# Tab 4 – Serial Monitor
# ---------------------------------------------------------------------------

class SerialMonitorTab(QWidget):
    log_message = pyqtSignal(str)

    BAUDRATES = ["9600", "19200", "38400", "57600", "115200", "230400", "460800", "921600"]

    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._reader = None
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # Control bar
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Port:"))
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(110)
        ctrl.addWidget(self.port_combo)

        refresh_btn = QPushButton("⟳")
        refresh_btn.setFixedWidth(30)
        refresh_btn.setToolTip("Refresh ports")
        refresh_btn.clicked.connect(self._refresh_ports)
        ctrl.addWidget(refresh_btn)

        ctrl.addWidget(QLabel("Baud:"))
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(self.BAUDRATES)
        self.baud_combo.setCurrentText("921600")
        ctrl.addWidget(self.baud_combo)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setFixedWidth(100)
        self.connect_btn.setStyleSheet("background-color: #2d6a4f; color: white;")
        self.connect_btn.clicked.connect(self._toggle_connect)
        ctrl.addWidget(self.connect_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setFixedWidth(70)
        self.clear_btn.clicked.connect(self._clear_output)
        ctrl.addWidget(self.clear_btn)
        ctrl.addStretch()
        root.addLayout(ctrl)

        # Output area
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Consolas", 9))
        self.output.setStyleSheet(
            "background-color: #0d0d0d; color: #d4d4d4; border: 1px solid #444;"
        )
        root.addWidget(self.output, stretch=1)

        # Input bar
        inp = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Type command and press Enter or Send…")
        self.input_edit.returnPressed.connect(self._send_data)
        self.send_btn = QPushButton("Send")
        self.send_btn.setFixedWidth(70)
        self.send_btn.clicked.connect(self._send_data)
        inp.addWidget(self.input_edit)
        inp.addWidget(self.send_btn)
        root.addLayout(inp)

        self._refresh_ports()

    def _load_settings(self):
        saved_port = self.settings.value("serial/port", "")
        saved_baud = self.settings.value("serial/baud", "921600")
        idx = self.port_combo.findText(saved_port)
        if idx >= 0:
            self.port_combo.setCurrentIndex(idx)
        idx2 = self.baud_combo.findText(saved_baud)
        if idx2 >= 0:
            self.baud_combo.setCurrentIndex(idx2)

    def _save_settings(self):
        self.settings.setValue("serial/port", self.port_combo.currentText())
        self.settings.setValue("serial/baud", self.baud_combo.currentText())

    def _refresh_ports(self):
        current = self.port_combo.currentText()
        self.port_combo.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo.addItems(ports)
        idx = self.port_combo.findText(current)
        if idx >= 0:
            self.port_combo.setCurrentIndex(idx)

    def _toggle_connect(self):
        if self._reader and self._reader.isRunning():
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        port = self.port_combo.currentText()
        baud = int(self.baud_combo.currentText())
        if not port:
            self.log_message.emit("[Serial] ERROR: No port selected.")
            return
        self._save_settings()
        self._reader = SerialReader(port, baud)
        self._reader.data_received.connect(self._append_output)
        self._reader.error_occurred.connect(self._on_serial_error)
        self._reader.start()
        self.connect_btn.setText("Disconnect")
        self.connect_btn.setStyleSheet("background-color: #d62828; color: white;")
        self.port_combo.setEnabled(False)
        self.baud_combo.setEnabled(False)
        self.log_message.emit(f"[Serial] Connected to {port} @ {baud} baud.")

    def _disconnect(self):
        if self._reader:
            self._reader.stop()
            self._reader = None
        self.connect_btn.setText("Connect")
        self.connect_btn.setStyleSheet("background-color: #2d6a4f; color: white;")
        self.port_combo.setEnabled(True)
        self.baud_combo.setEnabled(True)
        self.log_message.emit("[Serial] Disconnected.")

    def _on_serial_error(self, msg: str):
        self._disconnect()
        self.log_message.emit(f"[Serial] ERROR: {msg}")

    def _append_output(self, text: str):
        cursor = self.output.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.output.setTextCursor(cursor)
        self.output.ensureCursorVisible()

    def _clear_output(self):
        self.output.clear()

    def _send_data(self):
        text = self.input_edit.text()
        if self._reader and self._reader.isRunning():
            self._reader.send(text + "\r\n")
            self.input_edit.clear()
        else:
            self.log_message.emit("[Serial] Not connected.")

    def cleanup(self):
        if self._reader and self._reader.isRunning():
            self._reader.stop()


# ---------------------------------------------------------------------------
# Tab 5 – Camera View  (serial JPEG stream from Grove Vision AI V2)
# ---------------------------------------------------------------------------

class CameraWorker(QThread):
    """
    Reads raw bytes from the device's serial port, locates JPEG frames by
    their SOI (0xFF 0xD8) / EOI (0xFF 0xD9) markers, and emits each frame
    as a QImage decoded natively by Qt — no OpenCV required.
    """
    frame_ready = pyqtSignal(QImage)
    fps_update  = pyqtSignal(float)
    error       = pyqtSignal(str)

    _SOI = b'\xff\xd8'
    _EOI = b'\xff\xd9'
    _BUF_LIMIT = 2 * 1024 * 1024  # 2 MB — discard if no frame found

    def __init__(self, port: str, baudrate: int):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self._running = False

    def run(self):
        try:
            ser = serial.Serial(self.port, self.baudrate, timeout=0.05)
        except serial.SerialException as exc:
            self.error.emit(str(exc))
            return

        self._running = True
        buf = bytearray()
        t_prev = time.perf_counter()
        fps_acc, fps_n = 0.0, 0

        while self._running:
            chunk = ser.read(ser.in_waiting or 1)
            if chunk:
                buf.extend(chunk)

            # Extract every complete JPEG frame sitting in the buffer
            while True:
                soi = buf.find(self._SOI)
                if soi == -1:
                    buf.clear()
                    break
                # Discard garbage before SOI
                if soi > 0:
                    del buf[:soi]
                eoi = buf.find(self._EOI, 2)
                if eoi == -1:
                    break  # frame not complete yet
                jpg_bytes = bytes(buf[:eoi + 2])
                del buf[:eoi + 2]

                img = QImage()
                if img.loadFromData(jpg_bytes, "JPEG") and not img.isNull():
                    self.frame_ready.emit(img)
                    t_now = time.perf_counter()
                    fps_acc += 1.0 / max(t_now - t_prev, 1e-6)
                    fps_n += 1
                    t_prev = t_now
                    if fps_n >= 10:
                        self.fps_update.emit(fps_acc / fps_n)
                        fps_acc, fps_n = 0.0, 0

            # Safety valve — drop buffer if it grows huge with no valid frame
            if len(buf) > self._BUF_LIMIT:
                buf.clear()

        ser.close()

    def stop(self):
        self._running = False
        self.wait(3000)


class CameraTab(QWidget):
    log_message = pyqtSignal(str)

    BAUDRATES = ["115200", "230400", "460800", "921600"]

    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._worker: CameraWorker | None = None
        self._last_frame: QImage | None = None
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # ── Control bar ────────────────────────────────────────────────────
        ctrl = QHBoxLayout()

        ctrl.addWidget(QLabel("Port:"))
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(110)
        self._refresh_ports()
        ctrl.addWidget(self.port_combo)

        refresh_btn = QPushButton("⟳")
        refresh_btn.setFixedWidth(30)
        refresh_btn.setToolTip("Rescan serial ports")
        refresh_btn.clicked.connect(self._refresh_ports)
        ctrl.addWidget(refresh_btn)

        ctrl.addWidget(QLabel("Baud:"))
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(self.BAUDRATES)
        self.baud_combo.setCurrentText("921600")
        ctrl.addWidget(self.baud_combo)

        self.start_btn = QPushButton("Start")
        self.start_btn.setFixedWidth(80)
        self.start_btn.setStyleSheet("background-color: #2d6a4f; color: white;")
        self.start_btn.clicked.connect(self._toggle_camera)
        ctrl.addWidget(self.start_btn)

        self.snap_btn = QPushButton("Snapshot")
        self.snap_btn.setFixedWidth(90)
        self.snap_btn.setEnabled(False)
        self.snap_btn.clicked.connect(self._save_snapshot)
        ctrl.addWidget(self.snap_btn)

        ctrl.addStretch()

        self.fps_lbl = QLabel("FPS: —")
        self.fps_lbl.setStyleSheet("color: #6ec6ff;")
        ctrl.addWidget(self.fps_lbl)

        self.res_lbl = QLabel("")
        self.res_lbl.setStyleSheet("color: #888;")
        ctrl.addWidget(self.res_lbl)

        root.addLayout(ctrl)

        # ── Video display ──────────────────────────────────────────────────
        self.view = QLabel()
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setStyleSheet("background-color: #111; border: 1px solid #444;")
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setText("Select the device COM port and press Start")
        self.view.setFont(QFont("Consolas", 10))
        root.addWidget(self.view, stretch=1)

    def _load_settings(self):
        saved_port = self.settings.value("camera/port", "")
        idx = self.port_combo.findText(saved_port)
        if idx >= 0:
            self.port_combo.setCurrentIndex(idx)
        saved_baud = self.settings.value("camera/baud", "921600")
        bi = self.baud_combo.findText(saved_baud)
        if bi >= 0:
            self.baud_combo.setCurrentIndex(bi)

    def _save_settings(self):
        self.settings.setValue("camera/port", self.port_combo.currentText())
        self.settings.setValue("camera/baud", self.baud_combo.currentText())

    def _refresh_ports(self):
        current = self.port_combo.currentText()
        self.port_combo.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo.addItems(ports)
        idx = self.port_combo.findText(current)
        if idx >= 0:
            self.port_combo.setCurrentIndex(idx)

    def _toggle_camera(self):
        if self._worker and self._worker.isRunning():
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        port = self.port_combo.currentText()
        baud = int(self.baud_combo.currentText())
        if not port:
            self.log_message.emit("[Camera] ERROR: No serial port selected.")
            return
        self._save_settings()
        self.log_message.emit(f"[Camera] Connecting to {port} @ {baud} — waiting for JPEG stream…")

        self._worker = CameraWorker(port, baud)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.fps_update.connect(self._on_fps)
        self._worker.error.connect(self._on_error)
        self._worker.start()

        self.start_btn.setText("Stop")
        self.start_btn.setStyleSheet("background-color: #d62828; color: white;")
        self.snap_btn.setEnabled(True)
        self.port_combo.setEnabled(False)
        self.baud_combo.setEnabled(False)

    def _stop_camera(self):
        if self._worker:
            self._worker.stop()
            self._worker = None
        self.start_btn.setText("Start")
        self.start_btn.setStyleSheet("background-color: #2d6a4f; color: white;")
        self.snap_btn.setEnabled(False)
        self.port_combo.setEnabled(True)
        self.baud_combo.setEnabled(True)
        self.fps_lbl.setText("FPS: —")
        self.res_lbl.setText("")
        self.view.setPixmap(QPixmap())
        self.view.setText("Select the device COM port and press Start")
        self.log_message.emit("[Camera] Disconnected.")

    def _on_frame(self, img: QImage):
        self._last_frame = img
        w = self.view.width()
        h = self.view.height()
        pix = QPixmap.fromImage(img).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.view.setPixmap(pix)
        self.res_lbl.setText(f"{img.width()}×{img.height()}")

    def _on_fps(self, fps: float):
        self.fps_lbl.setText(f"FPS: {fps:.1f}")

    def _on_error(self, msg: str):
        self._stop_camera()
        self.log_message.emit(f"[Camera] ERROR: {msg}")

    def _save_snapshot(self):
        if not self._last_frame:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save snapshot", "snapshot.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if path:
            self._last_frame.save(path)
            self.log_message.emit(f"[Camera] Snapshot saved → {path}")

    def cleanup(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

def apply_dark_palette(app: QApplication):
    palette = QPalette()
    c = {
        QPalette.Window:          QColor("#2b2b2b"),
        QPalette.WindowText:      QColor("#f0f0f0"),
        QPalette.Base:            QColor("#1e1e1e"),
        QPalette.AlternateBase:   QColor("#2b2b2b"),
        QPalette.ToolTipBase:     QColor("#1e1e1e"),
        QPalette.ToolTipText:     QColor("#f0f0f0"),
        QPalette.Text:            QColor("#f0f0f0"),
        QPalette.Button:          QColor("#3c3f41"),
        QPalette.ButtonText:      QColor("#f0f0f0"),
        QPalette.BrightText:      QColor("#ff5555"),
        QPalette.Link:            QColor("#6ec6ff"),
        QPalette.Highlight:       QColor("#214283"),
        QPalette.HighlightedText: QColor("#ffffff"),
    }
    for role, color in c.items():
        palette.setColor(role, color)
        palette.setColor(QPalette.Disabled, role, color.darker(150))
    app.setPalette(palette)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("GroveTool", "GroveVisionAI_V2")
        self._setup_ui()
        self._restore_geometry()

    def _setup_ui(self):
        self.setWindowTitle("Grove Vision AI V2 Tool")
        self.resize(1100, 750)

        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)
        self.setCentralWidget(central)

        splitter = QSplitter(Qt.Vertical)

        # Tabs
        self.tabs = QTabWidget()
        self.build_tab = BuildTab(self.settings)
        self.gen_tab = GenerateTab(self.settings)
        self.flash_tab = FlashTab(self.settings)
        self.serial_tab = SerialMonitorTab(self.settings)
        self.camera_tab = CameraTab(self.settings)

        self.tabs.addTab(self.build_tab,  "Build Firmware")
        self.tabs.addTab(self.gen_tab,    "Generate Image")
        self.tabs.addTab(self.flash_tab,  "Flash Firmware")
        self.tabs.addTab(self.serial_tab, "Serial Monitor")
        self.tabs.addTab(self.camera_tab, "Camera View")

        splitter.addWidget(self.tabs)

        # Shared log panel
        log_widget = QWidget()
        log_lay = QVBoxLayout(log_widget)
        log_lay.setContentsMargins(2, 2, 2, 2)
        log_lay.setSpacing(2)
        log_hdr = QHBoxLayout()
        log_hdr.addWidget(QLabel("Log"))
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.setFixedWidth(80)
        clear_log_btn.clicked.connect(self._clear_log)
        log_hdr.addStretch()
        log_hdr.addWidget(clear_log_btn)
        log_lay.addLayout(log_hdr)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFont(QFont("Consolas", 8))
        self.log_area.setStyleSheet(
            "background-color: #111; color: #a8c6a8; border: 1px solid #444;"
        )
        self.log_area.setMaximumHeight(180)
        log_lay.addWidget(self.log_area)
        splitter.addWidget(log_widget)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        # Wire all tabs' log signals
        for tab in (self.build_tab, self.gen_tab, self.flash_tab, self.serial_tab, self.camera_tab):
            tab.log_message.connect(self._append_log)

        # Status bar
        self.statusBar().showMessage("Ready")

    def _append_log(self, msg: str):
        self.log_area.append(msg)
        self.log_area.ensureCursorVisible()

    def _clear_log(self):
        self.log_area.clear()

    def _restore_geometry(self):
        geo = self.settings.value("window/geometry")
        if geo:
            self.restoreGeometry(geo)

    def closeEvent(self, event):
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.serial_tab.cleanup()
        self.camera_tab.cleanup()
        super().closeEvent(event)


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
