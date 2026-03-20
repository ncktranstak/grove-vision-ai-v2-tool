# Grove Vision AI V2 Tool

A PyQt5 desktop GUI for building, flashing, and monitoring the **Seeed Grove Vision AI Module V2** (Himax WiseEye2).

## Features

| Tab | Description |
|-----|-------------|
| **Build Firmware** | Select a scenario app, run `make clean` / `make` against the ARM toolchain |
| **Generate Image** | Copy the ELF output and run `we2_local_image_gen.exe` to produce `output.img` |
| **Flash Firmware** | Flash `output.img` (and optionally a `.tflite` model) over serial via `xmodem_send.py` |
| **Serial Monitor** | Connect to the device at 921600 baud, view live output, send commands |

All long-running operations stream live output to the shared **Log** panel at the bottom.
Settings (paths, COM port, baud rate) are persisted across sessions via `QSettings`.

## Requirements

- Python 3.8+
- [ARM GNU Toolchain 13.2](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads)
- [xpack windows-build-tools](https://github.com/xpack-dev-tools/windows-build-tools-xpack/releases) (`make`)
- [Seeed_Grove_Vision_AI_Module_V2](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) repo cloned locally

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python grove_vision_ai_v2_tool.py
```

## Usage

1. **Build Firmware**
   - Set *Repo directory* to your cloned repo root.
   - Set *ARM toolchain* to the `/bin` folder of the extracted ARM toolchain.
   - Pick a *Scenario App* from the dropdown.
   - Click **Clean** then **Build**. The output ELF path is shown on success.

2. **Generate Image**
   - The ELF path carries over automatically from the Build tab.
   - Click **Generate Image** — `output.img` will appear in `we2_image_gen_local/output_case1_sec_wlcsp/`.

3. **Flash Firmware**
   - Select the COM port for your device and browse to `output.img`.
   - Optionally enable model flashing (`.tflite` + flash position/offset).
   - Click **Flash Firmware**, then press the physical **RESET** button on the module when prompted.

4. **Serial Monitor**
   - Select the COM port and baud rate (default 921600).
   - Click **Connect** to stream device output.
   - Type commands in the input bar and press **Send** or Enter.

## Supported Scenario Apps

`tflm_fd_fm` · `tflm_yolov8_od` · `tflm_yolov8_pose` · `tflm_yolov8_gender_cls` · `pdm_record` · `kws_pdm_record` · `imu_read` · `tflm_peoplenet` · `tflm_yolo11_od` · `tflm_mb_cls` · `torch_mb_cls`
