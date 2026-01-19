
import unittest
from unittest.mock import MagicMock, patch
import json
from pathlib import Path
import os
from video_analyzer.analyzer import load_official_standards, AnalysisConfig, analyze_pickleball_video
from video_analyzer.llm_analyzer import PickleballAnalyzer, ANALYSIS_PROMPT_TEMPLATE, TechniqueAssessment

class TestStandardsIntegration(unittest.TestCase):
    def test_load_official_standards(self):
        # Should load the existing dictionary_official.json in root
        standards = load_official_standards()
        self.assertIn("Ready_Position", standards)
        self.assertIn("Forehand_Drive", standards)
        self.assertEqual(standards["Volley"]["Impact_Extension"]["min"], 130)

    def test_build_prompt_with_standards(self):
        analyzer = PickleballAnalyzer(api_key="fake_key")
        pose_data = {"test": "data"}
        official_standards = {"Ready_Position": {"Knee_Angle": {"min": 130, "max": 160}}}
        
        prompt = analyzer._build_prompt(pose_data, official_standards, "vi")
        
        self.assertIn("TIÊU CHUẨN KỸ THUẬT OFFICIAL", prompt)
        self.assertIn("Ready_Position", prompt)
        self.assertIn("Knee_Angle", prompt)
        self.assertIn("130", prompt)
        self.assertIn("test", prompt)

    def test_technique_assessment_new_fields(self):
        assessment = TechniqueAssessment(
            aspect="Gối",
            score=8,
            observation="Hơi cao",
            recommendation="Hạ thấp",
            actual_value="150",
            standard_range="130-140"
        )
        self.assertEqual(assessment.actual_value, "150")
        self.assertEqual(assessment.standard_range, "130-140")

    @patch('video_analyzer.analyzer.os.path.exists')
    @patch('video_analyzer.llm_analyzer.OpenAI')
    @patch('video_analyzer.analyzer.extract_frames')
    @patch('video_analyzer.analyzer.PoseDetector')
    @patch('video_analyzer.analyzer.get_video_info')
    @patch('video_analyzer.analyzer.load_official_standards')
    def test_analyze_pickleball_video_flow(self, mock_load, mock_info, mock_detector, mock_extract, mock_openai, mock_exists):
        # Mocking all dependencies to check if official_standards is passed correctly
        mock_exists.return_value = True
        mock_load.return_value = {"mock": "standards"}
        mock_info.return_value = {"duration": 10}
        
        mock_pose_instance = mock_detector.return_value
        mock_pose_result = MagicMock()
        mock_pose_result.to_summary_dict.return_value = {"pose": "data"}
        mock_pose_instance.detect_video.return_value = mock_pose_result
        
        # We need to mock the frames list so it doesn't raise ValueError
        frame_mock = MagicMock()
        frame_mock.base64_data = "base64"
        mock_extract.return_value = [frame_mock]
        
        # Mock PickleballAnalyzer.analyze
        with patch('video_analyzer.analyzer.PickleballAnalyzer') as MockAnalyzerClass:
            mock_analyzer_instance = MockAnalyzerClass.return_value
            
            analyze_pickleball_video("fake.mp4")
            
            # Check if load_official_standards was called
            mock_load.assert_called_once()
            
            # Check if analyzer.analyze was called with official_standards
            mock_analyzer_instance.analyze.assert_called_once()
            args, kwargs = mock_analyzer_instance.analyze.call_args
            self.assertEqual(kwargs['official_standards'], {"mock": "standards"})

if __name__ == '__main__':
    unittest.main()
