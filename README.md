# LeRobot Demo (Linux)

## Quick start

1) Ensure the conda env is ready and cameras/robot are plugged in.
2) Run the demo script:

```bash
./scripts/demo.sh
```

- This records locally to a timestamped folder under `/home/aditya/lerobot/local_eval_demo_<timestamp>`.
- Uses `/dev/ttyACM0` for the follower arm and two OpenCV cameras:
  - gripper: `/dev/video4`
  - top: `/dev/video6`
- Cameras run at 640x480 @ 30 fps by default.

## Adjustments

- Different camera devices:
  - Edit `scripts/demo.sh` and change `index_or_path` for `gripper` and `top` to your `/dev/videoN` paths.
- Show camera windows during the demo:
  - `./scripts/demo.sh --display_data=true`

## Calibration prompt

On first run (or after hardware changes), you may see:

- "Mismatch between calibration values... Press ENTER to use provided calibration file ... or type 'c'".
  - Press ENTER to use the existing file.
  - Type `c` to run calibration again.

## Common failure cases and fixes

- Permission denied on serial port `/dev/ttyACM0`:
  - Create a udev rule or temporarily: `sudo chmod 0777 /dev/ttyACM0`.
- Permission denied on cameras `/dev/video*`:
  - `sudo usermod -aG video $USER && newgrp video` or adjust udev rules.
- Wrong camera resolution on Linux (stuck at 640x480):
  - Force MJPG and desired mode using `v4l2-ctl`:
    ```bash
    sudo v4l2-ctl -d /dev/video6 --set-fmt-video=width=640,height=480,pixelformat=MJPG --set-parm=30
    sudo v4l2-ctl -d /dev/video4 --set-fmt-video=width=640,height=480,pixelformat=MJPG --set-parm=30
    ```
- ModuleNotFoundError: No module named 'lerobot':
  - Ensure the env is active or let the script activate it (it sources conda and tries to `conda activate lerobot`).
- Hugging Face login prompts or 404s during demo:
  - The demo saves locally and does not push to the Hub; no login is required.

## Notes

- To change the policy or dataset settings, edit `scripts/demo.sh`.
