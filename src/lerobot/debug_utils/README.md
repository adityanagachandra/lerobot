# Debug Utils

A comprehensive collection of debugging and diagnostic tools for LeRobot development and troubleshooting.

## 🔧 Quick Diagnostics

### Robot Connection Testing
- **`test_robot_connection.py`** - Quick diagnostic to test robot connection before recording
- **`debug_robot_state.py`** - Print current servo positions and robot state information

### Camera Performance Testing  
- **`test_camera_performance.py`** - Test camera performance with specific settings and configurations

### Recording Performance Analysis
- **`debug_recording_performance.py`** - Measure timing for each component in the recording loop

## 📊 Data Logging & Analysis

### Inference Logging
- **`inference_logger.py`** - Comprehensive logging of robot state, policy outputs, and trajectories during inference
- **`analyze_inference_logs.py`** - Analyze inference logs for performance metrics and visualization
- **`INFERENCE_LOGGING_README.md`** - Complete guide for using the inference logging system

### Joint Value Debugging
- **`debug_joint_values.py`** - Log joint values from both leader and follower robots to CSV for analysis

## 📚 Documentation & Guides

### Control Loop Optimization
- **`optimize_control_loop.md`** - Strategies for achieving constant 30 Hz control loop frequency
- **`control_loop_timing_explanation.md`** - Detailed explanation of control loop timing mechanics
- **`control_loop_frequency_analysis.md`** - Analysis and troubleshooting guide for control loop frequency

### Recording Documentation
- **`LEROBOT_RECORD_DOCUMENTATION.md`** - Comprehensive documentation for the LeRobot recording system

## 🗂️ Data Management

### Episode Management
- **`delete_episode.py`** - Utility to safely delete episodes from LeRobot datasets
  - Removes episode data files, videos, and metadata
  - Supports both local and Hub datasets

## 🚀 Quick Start

### Test Your Setup
```bash
# Test robot connection
python src/lerobot/debug_utils/test_robot_connection.py

# Check robot state
python src/lerobot/debug_utils/debug_robot_state.py

# Test camera performance
python src/lerobot/debug_utils/test_camera_performance.py
```

### Debug Recording Issues
```bash
# Analyze recording performance
python src/lerobot/debug_utils/debug_recording_performance.py

# Log joint values for servo debugging
python src/lerobot/debug_utils/debug_joint_values.py
```

### Enable Inference Logging
```bash
# Record with logging enabled
python -m lerobot.record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760434091 \
    --log=true \
    --dataset.repo_id=your_dataset

# Analyze the logs
python src/lerobot/debug_utils/analyze_inference_logs.py
```

## 📁 File Structure

```
debug_utils/
├── README.md                              # This file
├── __init__.py                           # Module initialization
│
├── 🔧 Quick Diagnostics
├── test_robot_connection.py              # Robot connection testing
├── debug_robot_state.py                  # Robot state inspection
├── test_camera_performance.py            # Camera performance testing
├── debug_recording_performance.py        # Recording performance analysis
│
├── 📊 Data Logging & Analysis
├── inference_logger.py                   # Inference data logging
├── analyze_inference_logs.py             # Log analysis tools
├── debug_joint_values.py                 # Joint value logging
│
├── 📚 Documentation
├── INFERENCE_LOGGING_README.md           # Inference logging guide
├── optimize_control_loop.md              # Control loop optimization
├── control_loop_timing_explanation.md    # Timing mechanics explanation
├── control_loop_frequency_analysis.md    # Frequency analysis guide
├── LEROBOT_RECORD_DOCUMENTATION.md       # Recording system docs
│
└── 🗂️ Data Management
    └── delete_episode.py                 # Episode deletion utility
```

## 🎯 Common Use Cases

### Performance Troubleshooting
1. **Slow Control Loop**: Use `optimize_control_loop.md` and `debug_recording_performance.py`
2. **Camera Issues**: Use `test_camera_performance.py` 
3. **Servo Problems**: Use `debug_joint_values.py` and `debug_robot_state.py`

### Development & Analysis
1. **Policy Development**: Use `inference_logger.py` to capture detailed execution data
2. **Dataset Management**: Use `delete_episode.py` to clean up problematic episodes
3. **System Validation**: Use connection and state testing tools before recording

### Documentation & Learning
- Read the comprehensive guides in the markdown files
- Follow the inference logging README for detailed logging setup
- Use the control loop documentation to understand timing optimization

## 🛠️ Contributing

When adding new debug utilities:
1. Include clear docstrings explaining the tool's purpose
2. Add command-line interfaces where appropriate
3. Update this README with the new tool's description
4. Consider adding example usage in relevant documentation files

## 📝 Notes

- All tools are designed to be run independently for quick diagnostics
- CSV output format is used consistently for easy analysis in spreadsheet applications
- Timing measurements use high-precision `time.perf_counter()` for accuracy
- Tools gracefully handle connection failures and provide informative error messages