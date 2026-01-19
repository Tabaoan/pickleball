"""
LLM Analyzer Module
OpenAI GPT-4 Vision integration for pickleball analysis
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI


@dataclass
class TechniqueAssessment:
    """Assessment of a specific technique aspect"""
    aspect: str  # e.g., "Elbow Position", "Stance"
    score: int  # 1-10
    observation: str
    recommendation: str
    actual_value: Optional[str] = None      # NEW: Giá trị thực tế
    standard_range: Optional[str] = None    # NEW: Khoảng chuẩn


@dataclass
class MoveIdentification:
    """Identified pickleball move/action"""
    move_type: str  # e.g., "Forehand Volley", "Backhand Drive"
    confidence: str  # "high", "medium", "low"
    timestamps: str  # e.g., "0:02 - 0:05"


@dataclass
class AnalysisResult:
    """Complete analysis result from LLM"""
    # Overall assessment
    overall_score: int  # 1-10
    skill_level: str  # "Beginner", "Intermediate", "Advanced"
    summary: str
    
    # Detailed breakdowns
    moves_identified: List[MoveIdentification] = field(default_factory=list)
    technique_assessments: List[TechniqueAssessment] = field(default_factory=list)
    
    # Coaching feedback
    strengths: List[str] = field(default_factory=list)
    areas_to_improve: List[str] = field(default_factory=list)
    practice_drills: List[str] = field(default_factory=list)
    
    # Raw response for debugging
    raw_response: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "overall_score": self.overall_score,
            "skill_level": self.skill_level,
            "summary": self.summary,
            "moves_identified": [
                {"type": m.move_type, "confidence": m.confidence, "timestamps": m.timestamps}
                for m in self.moves_identified
            ],
            "technique_assessments": [
                {
                    "aspect": t.aspect,
                    "score": t.score,
                    "observation": t.observation,
                    "recommendation": t.recommendation,
                    "actual_value": t.actual_value,
                    "standard_range": t.standard_range
                }
                for t in self.technique_assessments
            ],
            "strengths": self.strengths,
            "areas_to_improve": self.areas_to_improve,
            "practice_drills": self.practice_drills
        }
    
    def to_report(self) -> str:
        """Generate human-readable report"""
        report = []
        report.append("# PICKLEBALL TECHNIQUE ANALYSIS REPORT")
        report.append(f"\nOverall Score: {self.overall_score}/10")
        report.append(f"Skill Level: {self.skill_level}")
        report.append(f"\nSummary: {self.summary}")
        
        report.append("\n## IDENTIFIED MOVES")
        for move in self.moves_identified:
            report.append(f"- {move.move_type} ({move.confidence} confidence) at {move.timestamps}")
            
        report.append("\n## TECHNIQUE ASSESSMENT")
        for assessment in self.technique_assessments:
            report.append(f"### {assessment.aspect} (Score: {assessment.score}/10)")
            report.append(f"**Observation:** {assessment.observation}")
            report.append(f"**Recommendation:** {assessment.recommendation}")
            
        report.append("\n## COACHING FEEDBACK")
        report.append("\n**Strengths:**")
        for strength in self.strengths:
            report.append(f"- {strength}")
            
        report.append("\n**Areas to Improve:**")
        for area in self.areas_to_improve:
            report.append(f"- {area}")
            
        report.append("\n**Suggested Practice Drills:**")
        for drill in self.practice_drills:
            report.append(f"- {drill}")
            
        return "\n".join(report)


# Prompt template
ANALYSIS_PROMPT_TEMPLATE = '''
Bạn là một huấn luyện viên pickleball chuyên nghiệp với 20 năm kinh nghiệm.
Hãy phân tích video pickleball dưới đây và đưa ra nhận xét chi tiết.

## TIÊU CHUẨN KỸ THUẬT OFFICIAL
Dưới đây là các góc khớp CHUẨN cho từng loại động tác pickleball:
```json
{official_standards}
```

## DỮ LIỆU POSE DETECTION (từ video người chơi)
Dưới đây là dữ liệu góc khớp được phát hiện từ video bằng AI:
```json
{pose_data}
```

## YÊU CẦU PHÂN TÍCH
Dựa vào các hình ảnh video VÀ dữ liệu góc khớp ở trên, hãy:

1. **NHẬN DIỆN ĐỘNG TÁC**: Xác định các loại đánh (Forehand_Drive, Volley, Dink, Serve, Ready_Position)

2. **SO SÁNH VỚI TIÊU CHUẨN**: 
   - Với mỗi động tác nhận diện được, so sánh góc thực tế với góc chuẩn trong TIÊU CHUẨN KỸ THUẬT OFFICIAL
   - Chỉ ra cụ thể: "Góc khuỷu tay thực tế là X°, chuẩn là Y°-Z°" 
   - Đánh giá PASS/FAIL cho từng tiêu chí

3. **ĐÁNH GIÁ KỸ THUẬT** (cho mỗi khía cạnh, chấm điểm 1-10):
   - Tư thế chân (stance, footwork)
   - Vị trí khuỷu tay (elbow position) - SO SÁNH với Elbow_Angle chuẩn
   - Góc vai (shoulder rotation)
   - Gối và trọng tâm (knee bend) - SO SÁNH với Knee_Angle chuẩn
   - Kết thúc động tác (follow-through)

4. **ĐIỂM MẠNH**: Liệt kê 2-3 điểm đạt/vượt tiêu chuẩn

5. **CẦN CẢI THIỆN**: Liệt kê 2-3 điểm CHƯA ĐẠT tiêu chuẩn, kèm số liệu cụ thể

6. **BÀI TẬP GỢI Ý**: Đề xuất 2-3 bài tập để đạt được góc chuẩn

## ĐỊNH DẠNG RESPONSE
Trả lời dưới dạng JSON với cấu trúc sau:
```json
{{
  "overall_score": <1-10>,
  "skill_level": "<Beginner/Intermediate/Advanced>",
  "summary": "<Tóm tắt 2-3 câu, đề cập đến so sánh với tiêu chuẩn>",
  "moves_identified": [
    {{"type": "<tên động tác>", "confidence": "<high/medium/low>", "timestamps": "<thời điểm>"}}
  ],
  "technique_assessments": [
    {{
      "aspect": "<khía cạnh>", 
      "score": <1-10>, 
      "actual_value": "<giá trị thực tế>",
      "standard_range": "<khoảng chuẩn>",
      "observation": "<quan sát, so sánh với chuẩn>", 
      "recommendation": "<gợi ý cụ thể>"
    }}
  ],
  "strengths": ["<điểm mạnh 1 - đạt chuẩn X>", "<điểm mạnh 2>"],
  "areas_to_improve": ["<cần cải thiện 1 - thực tế X° vs chuẩn Y°>", "<cần cải thiện 2>"],
  "practice_drills": ["<bài tập 1>", "<bài tập 2>"]
}}
```
'''


class PickleballAnalyzer:
    """OpenAI GPT-4 Vision analyzer for pickleball"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize analyzer
        
        Args:
            api_key: OpenAI API key (default: from OPENAI__API_KEY env var)
            model: Model to use (gpt-4o, gpt-4-vision-preview)
        """
        if api_key is None:
            api_key = os.getenv("OPENAI__API_KEY")
            
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI__API_KEY environment variable.")
            
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def analyze(
        self,
        frames_base64: List[str],
        pose_data: Dict[str, Any],
        official_standards: Optional[Dict[str, Any]] = None,
        language: str = "vi"  # "vi" = Vietnamese, "en" = English
    ) -> AnalysisResult:
        """
        Analyze pickleball video using vision + pose data + official standards
        
        Args:
            frames_base64: List of base64 encoded frame images
            pose_data: Pose detection data dictionary
            official_standards: Official technique standards from dictionary_official.json
            language: Response language (Currently localized in prompt)
            
        Returns:
            AnalysisResult with detailed analysis
            
        Raises:
            ValueError: If no frames provided
            Exception: If API call fails
        """
        if not frames_base64:
            raise ValueError("No frames provided for analysis")
            
        prompt = self._build_prompt(pose_data, official_standards, language)
        
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt}
        ]
        
        for base64_img in frames_base64:
            # Check if base64 data already has the prefix
            if not base64_img.startswith("data:image"):
                base64_img = f"data:image/jpeg;base64,{base64_img}"
                
            content.append({
                "type": "image_url",
                "image_url": {"url": base64_img}
            })
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_completion_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            response_text = response.choices[0].message.content
            if not response_text:
                raise ValueError("Empty response from LLM")
                
            result = self._parse_response(response_text)
            result.raw_response = response_text
            return result
            
        except Exception as e:
            # In a production app, we'd handle specific OpenAI exceptions here
            raise Exception(f"Analysis failed: {str(e)}")
    
    def _build_prompt(
        self, 
        pose_data: Dict[str, Any], 
        official_standards: Optional[Dict[str, Any]],
        language: str
    ) -> str:
        """Build the analysis prompt with pose data and official standards"""
        # Note: language parameter is currently ignored as the template is in Vietnamese
        # In a real app, we'd have multiple templates or translate.
        standards_json = json.dumps(official_standards, indent=2) if official_standards else "{}"
        return ANALYSIS_PROMPT_TEMPLATE.format(
            official_standards=standards_json,
            pose_data=json.dumps(pose_data, indent=2)
        )
    
    def _parse_response(self, response_text: str) -> AnalysisResult:
        """Parse LLM response into structured result"""
        data = json.loads(response_text)
        
        moves = [
            MoveIdentification(
                move_type=m.get("type", "Unknown"),
                confidence=m.get("confidence", "low"),
                timestamps=m.get("timestamps", "N/A")
            )
            for m in data.get("moves_identified", [])
        ]
        
        assessments = [
            TechniqueAssessment(
                aspect=t.get("aspect", "Unknown"),
                score=t.get("score", 0),
                observation=t.get("observation", ""),
                recommendation=t.get("recommendation", ""),
                actual_value=t.get("actual_value"),
                standard_range=t.get("standard_range")
            )
            for t in data.get("technique_assessments", [])
        ]
        
        return AnalysisResult(
            overall_score=data.get("overall_score", 0),
            skill_level=data.get("skill_level", "Unknown"),
            summary=data.get("summary", ""),
            moves_identified=moves,
            technique_assessments=assessments,
            strengths=data.get("strengths", []),
            areas_to_improve=data.get("areas_to_improve", []),
            practice_drills=data.get("practice_drills", [])
        )
