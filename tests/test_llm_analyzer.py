import os
import unittest
from unittest.mock import MagicMock, patch
from video_analyzer.llm_analyzer import PickleballAnalyzer, AnalysisResult, MoveIdentification, TechniqueAssessment

class TestLLMAnalyzer(unittest.TestCase):

    def setUp(self):
        self.api_key = "test_key"
        os.environ["OPENAI__API_KEY"] = self.api_key
        self.analyzer = PickleballAnalyzer()

    def test_init_no_api_key(self):
        # Temporarily remove API key from env
        old_key = os.environ.pop("OPENAI__API_KEY", None)
        try:
            with self.assertRaises(ValueError):
                PickleballAnalyzer()
        finally:
            if old_key:
                os.environ["OPENAI__API_KEY"] = old_key

    def test_build_prompt(self):
        pose_data = {"average_angles": {"Right Elbow": 120}}
        prompt = self.analyzer._build_prompt(pose_data, "vi")
        self.assertIn('"Right Elbow": 120', prompt)
        self.assertIn("huấn luyện viên pickleball", prompt)

    def test_parse_response(self):
        response_text = """
        {
          "overall_score": 8,
          "skill_level": "Intermediate",
          "summary": "Good overall form with some minor issues.",
          "moves_identified": [
            {"type": "Forehand Volley", "confidence": "high", "timestamps": "0:01-0:03"}
          ],
          "technique_assessments": [
            {"aspect": "Elbow Position", "score": 7, "observation": "Slightly low", "recommendation": "Raise it higher"}
          ],
          "strengths": ["Strong wrist"],
          "areas_to_improve": ["Footwork"],
          "practice_drills": ["Wall drills"]
        }
        """
        result = self.analyzer._parse_response(response_text)
        self.assertEqual(result.overall_score, 8)
        self.assertEqual(result.skill_level, "Intermediate")
        self.assertEqual(len(result.moves_identified), 1)
        self.assertEqual(result.moves_identified[0].move_type, "Forehand Volley")
        self.assertEqual(len(result.technique_assessments), 1)
        self.assertEqual(result.technique_assessments[0].aspect, "Elbow Position")
        self.assertEqual(result.strengths, ["Strong wrist"])

    def test_to_report(self):
        result = AnalysisResult(
            overall_score=8,
            skill_level="Intermediate",
            summary="Test summary",
            moves_identified=[MoveIdentification("Forehand", "high", "0:01")],
            technique_assessments=[TechniqueAssessment("Elbow", 8, "Good", "Keep it up")],
            strengths=["Wrist"],
            areas_to_improve=["Feet"],
            practice_drills=["Drill 1"]
        )
        report = result.to_report()
        self.assertIn("PICKLEBALL TECHNIQUE ANALYSIS REPORT", report)
        self.assertIn("Overall Score: 8/10", report)
        self.assertIn("Skill Level: Intermediate", report)
        self.assertIn("Test summary", report)
        self.assertIn("Forehand", report)
        self.assertIn("Elbow", report)

    @patch('video_analyzer.llm_analyzer.OpenAI')
    def test_analyze_success(self, mock_openai):
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        {
          "overall_score": 9,
          "skill_level": "Advanced",
          "summary": "Excellent play.",
          "moves_identified": [],
          "technique_assessments": [],
          "strengths": [],
          "areas_to_improve": [],
          "practice_drills": []
        }
        """
        mock_client.chat.completions.create.return_value = mock_response
        
        # Re-initialize with mock
        analyzer = PickleballAnalyzer(api_key="test")
        result = analyzer.analyze(["base64data"], {"dummy": "data"})
        
        self.assertEqual(result.overall_score, 9)
        mock_client.chat.completions.create.assert_called_once()

if __name__ == '__main__':
    unittest.main()
