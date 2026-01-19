"""
Test cases for the main analyzer module
"""

import os
import pytest
from video_analyzer import analyze_pickleball_video, AnalysisConfig, quick_analyze


def test_full_pipeline_mock(mocker):
    """Test full pipeline with mocked LLM and Pose Detector"""
    # Mocking frames
    mocker.patch("video_analyzer.analyzer.extract_frames", return_value=[mocker.Mock(base64_data="abc")])
    mocker.patch("video_analyzer.analyzer.get_video_info", return_value=mocker.Mock(total_frames=10, fps=30))
    
    # Mocking pose detector
    mock_pose_data = mocker.Mock()
    mock_pose_data.processed_frames = 5
    mock_pose_data.avg_angles = {"elbow": 90}
    mock_pose_data.to_summary_dict.return_value = {"summary": "pose"}
    mocker.patch("video_analyzer.analyzer.PoseDetector.detect_video", return_value=mock_pose_data)
    
    # Mocking LLM
    mock_analyzer_class = mocker.patch("video_analyzer.analyzer.PickleballAnalyzer")
    mock_analyzer_instance = mock_analyzer_class.return_value
    
    mock_result = mocker.Mock()
    mock_result.overall_score = 8
    mock_result.skill_level = "Intermediate"
    mock_result.summary = "Good play"
    mock_result.to_report.return_value = "Mock Report"
    
    mock_analyzer_instance.analyze.return_value = mock_result
    
    # Run analysis
    video_path = "short_pickle.mp4" # Use an existing file to pass os.path.exists
    result = analyze_pickleball_video(video_path)
    
    assert result.overall_score == 8
    assert result.skill_level == "Intermediate"
    assert "Good play" in result.summary


def test_invalid_video():
    """Test with non-existent video file"""
    with pytest.raises(FileNotFoundError):
        analyze_pickleball_video("nonexistent.mp4")


def test_quick_analyze_mock(mocker):
    """Test quick_analyze with mocks"""
    mocker.patch("video_analyzer.analyzer.analyze_pickleball_video")
    from video_analyzer.analyzer import analyze_pickleball_video as mock_analyze
    
    mock_result = mocker.Mock()
    mock_result.summary = "Quick summary"
    mock_analyze.return_value = mock_result
    
    # We need an existing file for os.path.exists check if we didn't mock it
    # But wait, analyze_pickleball_video is mocked, so let's see.
    # The quick_analyze calls analyze_pickleball_video(video_path, config)
    
    mocker.patch("os.path.exists", return_value=True)
    summary = quick_analyze("dummy.mp4")
    assert summary == "Quick summary"

if __name__ == "__main__":
    # If running directly, try a real run if short_pickle.mp4 exists and API key is set
    if os.path.exists("short_pickle.mp4") and os.getenv("OPENAI__API_KEY"):
        print("Running real analysis test...")
        result = analyze_pickleball_video("short_pickle.mp4")
        print(f"Score: {result.overall_score}")
        print(result.to_report())
