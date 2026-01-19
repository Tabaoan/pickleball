import os
import sys
from typing import List

# Add parent directory to path to import video_analyzer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from video_analyzer.frame_extractor import extract_frames, get_video_info, ExtractedFrame

def test_basic_extraction():
    print("Running Test 1: Basic extraction...")
    video_path = "short_pickle.mp4"
    if not os.path.exists(video_path):
        print(f"Skipping Test 1: {video_path} not found")
        return

    frames = extract_frames(video_path, num_frames=5)
    assert len(frames) == 5
    assert all(f.base64_data.startswith("data:image") for f in frames)
    print("✅ Test 1 passed: Basic extraction successful")


def test_video_info():
    print("Running Test 2: Video info...")
    video_path = "short_pickle.mp4"
    if not os.path.exists(video_path):
        print(f"Skipping Test 2: {video_path} not found")
        return

    info = get_video_info(video_path)
    assert info.fps > 0
    assert info.duration > 0
    assert info.width > 0
    assert info.height > 0
    print(f"✅ Test 2 passed: Video info: {info.width}x{info.height}, {info.fps} FPS, {info.duration:.2f}s")


def test_invalid_video():
    print("Running Test 3: Invalid video...")
    try:
        extract_frames("nonexistent.mp4")
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError:
        print("✅ Test 3 passed: FileNotFoundError raised correctly")


def test_keyframes_mode():
    print("Running Test 4: Keyframes mode...")
    video_path = "short_pickle.mp4"
    if not os.path.exists(video_path):
        print(f"Skipping Test 4: {video_path} not found")
        return

    frames = extract_frames(video_path, num_frames=8, sampling_mode="keyframes")
    assert len(frames) == 8
    # Check if frames are sorted by number
    frame_numbers = [f.frame_number for f in frames]
    assert frame_numbers == sorted(frame_numbers)
    print(f"✅ Test 4 passed: Keyframes extraction successful. Frame numbers: {frame_numbers}")


if __name__ == "__main__":
    try:
        test_basic_extraction()
        test_video_info()
        test_invalid_video()
        test_keyframes_mode()
        print("\nAll tests passed successfully!")
    except AssertionError as e:
        print(f"\n❌ Test failed with AssertionError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
