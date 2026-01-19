"""
Main Analyzer Module
Combines all components into a single function
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from .frame_extractor import extract_frames, get_video_info
from .llm_analyzer import AnalysisResult, PickleballAnalyzer
from .pose_detector import PoseDetector

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_official_standards(standards_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load official technique standards from JSON file
    
    Args:
        standards_path: Path to dictionary_official.json
                       Default: looks in project root
    
    Returns:
        Dictionary containing official standards
        
    Raises:
        FileNotFoundError: If standards file not found
    """
    import json
    if standards_path is None:
        # Default path: project root / dictionary_official.json
        standards_path = Path(__file__).parent.parent / "dictionary_official.json"
    
    standards_path = Path(standards_path)
    
    if not standards_path.exists():
        raise FileNotFoundError(f"Official standards file not found: {standards_path}")
    
    with open(standards_path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class AnalysisConfig:
    """Configuration for video analysis"""
    
    # Frame extraction
    num_frames: int = 8
    max_frame_width: int = 1024
    sampling_mode: str = "uniform"  # "uniform" | "keyframes"
    
    # Pose detection
    pose_sample_rate: int = 2  # Process every Nth frame
    pose_confidence: float = 0.5
    
    # LLM
    openai_model: str = "gpt-5.2"
    language: str = "vi"  # "vi" | "en"
    
    # Output
    save_report: bool = False
    report_path: Optional[str] = None
    
    # Standards (NEW)
    use_official_standards: bool = True  # NEW: Enable/disable standards comparison
    standards_path: Optional[str] = None  # NEW: Custom path to standards file
    
    # Debug
    save_frames: bool = False
    frames_dir: Optional[str] = None
    verbose: bool = False


def analyze_pickleball_video(
    video_path: str,
    config: Optional[AnalysisConfig] = None
) -> AnalysisResult:
    """
    Analyze a pickleball video and get coaching feedback.
    
    This function uses a hybrid approach:
    1. Extracts key frames from the video
    2. Runs YOLO pose detection to get joint angles
    3. Sends both frames (visual) and pose data (numerical) to GPT-4 Vision
    4. Returns structured analysis with technique assessment and coaching tips
    
    Args:
        video_path: Path to the video file (.mp4)
        config: Optional configuration. Uses defaults if not provided.
        
    Returns:
        AnalysisResult containing overall score, skill level, summary, etc.
            
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be processed
        EnvironmentError: If OPENAI__API_KEY is not set
    """
    if config is None:
        config = AnalysisConfig()
    
    if config.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Step 1: Validate video
    logger.info(f"📹 Analyzing video: {video_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    video_info = get_video_info(video_path)
    logger.debug(f"Video info: {video_info}")
    
    # Step 2: Extract frames
    logger.info(f"🖼️ Extracting {config.num_frames} frames...")
    frames = extract_frames(
        video_path,
        num_frames=config.num_frames,
        max_width=config.max_frame_width,
        sampling_mode=config.sampling_mode
    )
    logger.info(f"✅ Extracted {len(frames)} frames")
    
    # Optional: Save frames for debugging
    if config.save_frames and config.frames_dir:
        os.makedirs(config.frames_dir, exist_ok=True)
        for i, f in enumerate(frames):
            frame_path = os.path.join(config.frames_dir, f"frame_{i}_{f.frame_number}.jpg")
            f.image.save(frame_path)
        logger.info(f"📁 Saved debug frames to: {config.save_frames}")
    
    # Step 3: Run pose detection
    logger.info("🏃 Running pose detection...")
    detector = PoseDetector()
    pose_data = detector.detect_video(
        video_path,
        sample_rate=config.pose_sample_rate,
        confidence_threshold=config.pose_confidence
    )
    logger.info(f"✅ Processed {pose_data.processed_frames} frames")
    logger.debug(f"Average angles: {pose_data.avg_angles}")
    
    # Step 4: Prepare data for LLM
    frames_base64 = [f.base64_data for f in frames]
    pose_summary = pose_data.to_summary_dict()
    
    # Step 4.5: Load official standards (NEW)
    official_standards = None
    if config.use_official_standards:
        try:
            official_standards = load_official_standards(config.standards_path)
            logger.info(f"📚 Loaded official standards with {len(official_standards)} move types")
        except FileNotFoundError as e:
            logger.warning(f"⚠️ Official standards not found, proceeding without: {e}")
    
    # Step 5: Call LLM for analysis
    logger.info("🤖 Sending to GPT-4 Vision for analysis...")
    analyzer = PickleballAnalyzer(model=config.openai_model)
    result = analyzer.analyze(
        frames_base64=frames_base64,
        pose_data=pose_summary,
        official_standards=official_standards,
        language=config.language
    )
    logger.info(f"✅ Analysis complete! Score: {result.overall_score}/10")
    
    # Step 6: Optional - Save report
    if config.save_report:
        report_path = config.report_path or f"report_{Path(video_path).stem}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(result.to_report())
        logger.info(f"📄 Report saved to: {report_path}")
    
    return result


def quick_analyze(video_path: str, language: str = "vi") -> str:
    """
    Quick analysis - returns just the summary text.
    
    Args:
        video_path: Path to video file
        language: "vi" for Vietnamese, "en" for English
        
    Returns:
        Summary text of the analysis
    """
    config = AnalysisConfig(language=language, num_frames=5)
    result = analyze_pickleball_video(video_path, config)
    return result.summary
