import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from video_analyzer.pose_detector import PoseDetector, calculate_angle

def test_angle_calculation():
    print("Testing angle calculation...")
    # 90 degrees
    angle = calculate_angle((0, 1), (0, 0), (1, 0))
    print(f"Angle (0,1)-(0,0)-(1,0): {angle}")
    assert 89.9 < angle < 90.1
    
    # 180 degrees
    angle = calculate_angle((1, 0), (0, 0), (-1, 0))
    print(f"Angle (1,0)-(0,0)-(-1,0): {angle}")
    assert 179.9 < angle <= 180.0
    
    # 45 degrees
    angle = calculate_angle((1, 1), (0, 0), (1, 0))
    print(f"Angle (1,1)-(0,0)-(1,0): {angle}")
    assert 44.9 < angle < 45.1
    print("✅ Angle calculation test passed!")

def test_model_loading():
    print("Testing model loading...")
    try:
        detector = PoseDetector()
        assert detector.model is not None
        print("✅ Model loading test passed!")
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        raise

def test_video_detection():
    print("Testing video detection...")
    video_path = "short_pickle.mp4"
    if not os.path.exists(video_path):
        print(f"⚠️ Skipping video detection test: {video_path} not found")
        return

    detector = PoseDetector()
    pose_data = detector.detect_video(video_path, sample_rate=10) # High sample rate for speed
    
    print(f"Processed {pose_data.processed_frames} / {pose_data.total_frames} frames")
    assert pose_data.processed_frames > 0
    assert len(pose_data.frames) > 0
    
    # Test summary generation
    summary = pose_data.to_summary_dict()
    assert "average_angles" in summary
    assert "angle_ranges" in summary
    assert len(summary["sample_frames"]) > 0
    print("✅ Video detection and summary test passed!")

if __name__ == "__main__":
    try:
        test_angle_calculation()
        test_model_loading()
        test_video_detection()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        sys.exit(1)
