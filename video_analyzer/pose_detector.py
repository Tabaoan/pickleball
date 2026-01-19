"""
Pose Detector Module
YOLO-based pose detection và angle calculation
"""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# COCO pose keypoint indices
KEYPOINT_NAMES = {
    0: "Nose",
    5: "Left Shoulder",
    6: "Right Shoulder",
    7: "Left Elbow",
    8: "Right Elbow",
    9: "Left Wrist",
    10: "Right Wrist",
    11: "Left Hip",
    12: "Right Hip",
    13: "Left Knee",
    14: "Right Knee",
    15: "Left Ankle",
    16: "Right Ankle",
}


@dataclass
class Keypoint:
    """Single keypoint with coordinates and confidence"""

    name: str
    x: float
    y: float
    confidence: float


@dataclass
class JointAngles:
    """Calculated joint angles for a frame"""

    right_elbow: Optional[float] = None
    left_elbow: Optional[float] = None
    right_shoulder: Optional[float] = None
    left_shoulder: Optional[float] = None
    right_knee: Optional[float] = None
    left_knee: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Convert to dictionary"""
        return {
            "Right Elbow": self.right_elbow,
            "Left Elbow": self.left_elbow,
            "Right Shoulder": self.right_shoulder,
            "Left Shoulder": self.left_shoulder,
            "Right Knee": self.right_knee,
            "Left Knee": self.left_knee,
        }


@dataclass
class FramePose:
    """Pose data for a single frame"""

    frame_number: int
    timestamp: float
    keypoints: Dict[str, Keypoint]
    angles: JointAngles
    confidence: float  # Overall detection confidence


@dataclass
class VideoPoseData:
    """Aggregated pose data for entire video"""

    video_path: str
    total_frames: int
    processed_frames: int
    frames: List[FramePose]

    # Summary statistics
    avg_angles: Dict[str, float] = field(default_factory=dict)
    angle_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def to_summary_dict(self) -> Dict:
        """Convert to summary dictionary for LLM"""
        return {
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "average_angles": self.avg_angles,
            "angle_ranges": {k: {"min": v[0], "max": v[1]} for k, v in self.angle_ranges.items()},
            "sample_frames": [
                {
                    "frame": f.frame_number,
                    "timestamp": f.timestamp,
                    "angles": f.angles.to_dict(),
                    "confidence": f.confidence,
                }
                for f in self.frames[:10]  # First 10 frames as sample
            ],
        }


def calculate_angle(
    p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]
) -> float:
    """
    Calculate angle at p2 formed by p1-p2-p3

    Args:
        p1, p2, p3: (x, y) coordinates

    Returns:
        Angle in degrees (0-180)
    """
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return float(np.degrees(angle))


class PoseDetector:
    """YOLO-based pose detector"""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize pose detector

        Args:
            model_path: Path to YOLO model weights
                       Default: looks for yolo26m-pose in project root
        """
        if model_path is None:
            model_path = "yolo26m-pose (1).pt"

        if not os.path.exists(model_path):
            # Try to find it in the project root if it was a relative path
            alt_path = Path(__file__).parent.parent / model_path
            if alt_path.exists():
                model_path = str(alt_path)
            else:
                raise FileNotFoundError(f"YOLO model not found at {model_path}")

        self.model = YOLO(model_path)

    def detect_video(
        self, video_path: str, sample_rate: int = 1, confidence_threshold: float = 0.5
    ) -> VideoPoseData:
        """
        Detect poses in video

        Args:
            video_path: Path to video file
            sample_rate: Process every Nth frame (1 = all frames)
            confidence_threshold: Minimum confidence for detection

        Returns:
            VideoPoseData with all frame poses and statistics
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames_data: List[FramePose] = []
        current_frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame_idx % sample_rate == 0:
                timestamp = current_frame_idx / fps if fps > 0 else 0.0
                pose = self.detect_frame(frame, current_frame_idx, timestamp)

                if pose and pose.confidence >= confidence_threshold:
                    frames_data.append(pose)

            current_frame_idx += 1

        cap.release()

        # Calculate statistics
        summary_data = VideoPoseData(
            video_path=video_path,
            total_frames=total_frames,
            processed_frames=len(frames_data),
            frames=frames_data,
        )

        if frames_data:
            angle_keys = [
                "right_elbow",
                "left_elbow",
                "right_shoulder",
                "left_shoulder",
                "right_knee",
                "left_knee",
            ]
            for key in angle_keys:
                angles = [getattr(f.angles, key) for f in frames_data if getattr(f.angles, key) is not None]
                if angles:
                    summary_data.avg_angles[key] = float(np.mean(angles))
                    summary_data.angle_ranges[key] = (float(np.min(angles)), float(np.max(angles)))

        return summary_data

    def detect_frame(
        self, frame: np.ndarray, frame_number: int = 0, timestamp: float = 0.0
    ) -> Optional[FramePose]:
        """
        Detect pose in single frame

        Args:
            frame: numpy array (BGR format from OpenCV)
            frame_number: Frame index
            timestamp: Frame timestamp in seconds

        Returns:
            FramePose or None if no detection
        """
        results = self.model.predict(frame, verbose=False)

        if not results or len(results[0].keypoints) == 0:
            return None

        # Take the first detection (assume single player as per scope)
        kp_data = results[0].keypoints[0]
        points = kp_data.xy[0].cpu().numpy()  # (N, 2)
        confidences = kp_data.conf[0].cpu().numpy()  # (N,)

        keypoints: Dict[str, Keypoint] = {}
        for idx, name in KEYPOINT_NAMES.items():
            if idx < len(points):
                keypoints[name] = Keypoint(
                    name=name, x=float(points[idx][0]), y=float(points[idx][1]), confidence=float(confidences[idx])
                )

        # Calculate angles
        angles = JointAngles()

        # Right Elbow: Shoulder(6) -> Elbow(8) -> Wrist(10)
        if all(k in keypoints for k in ["Right Shoulder", "Right Elbow", "Right Wrist"]):
            angles.right_elbow = calculate_angle(
                (keypoints["Right Shoulder"].x, keypoints["Right Shoulder"].y),
                (keypoints["Right Elbow"].x, keypoints["Right Elbow"].y),
                (keypoints["Right Wrist"].x, keypoints["Right Wrist"].y),
            )

        # Left Elbow: Shoulder(5) -> Elbow(7) -> Wrist(9)
        if all(k in keypoints for k in ["Left Shoulder", "Left Elbow", "Left Wrist"]):
            angles.left_elbow = calculate_angle(
                (keypoints["Left Shoulder"].x, keypoints["Left Shoulder"].y),
                (keypoints["Left Elbow"].x, keypoints["Left Elbow"].y),
                (keypoints["Left Wrist"].x, keypoints["Left Wrist"].y),
            )

        # Right Shoulder: Elbow(8) -> Shoulder(6) -> Hip(12)
        if all(k in keypoints for k in ["Right Elbow", "Right Shoulder", "Right Hip"]):
            angles.right_shoulder = calculate_angle(
                (keypoints["Right Elbow"].x, keypoints["Right Elbow"].y),
                (keypoints["Right Shoulder"].x, keypoints["Right Shoulder"].y),
                (keypoints["Right Hip"].x, keypoints["Right Hip"].y),
            )

        # Left Shoulder: Elbow(7) -> Shoulder(5) -> Hip(11)
        if all(k in keypoints for k in ["Left Elbow", "Left Shoulder", "Left Hip"]):
            angles.left_shoulder = calculate_angle(
                (keypoints["Left Elbow"].x, keypoints["Left Elbow"].y),
                (keypoints["Left Shoulder"].x, keypoints["Left Shoulder"].y),
                (keypoints["Left Hip"].x, keypoints["Left Hip"].y),
            )

        # Right Knee: Hip(12) -> Knee(14) -> Ankle(16)
        if all(k in keypoints for k in ["Right Hip", "Right Knee", "Right Ankle"]):
            angles.right_knee = calculate_angle(
                (keypoints["Right Hip"].x, keypoints["Right Hip"].y),
                (keypoints["Right Knee"].x, keypoints["Right Knee"].y),
                (keypoints["Right Ankle"].x, keypoints["Right Ankle"].y),
            )

        # Left Knee: Hip(11) -> Knee(13) -> Ankle(15)
        if all(k in keypoints for k in ["Left Hip", "Left Knee", "Left Ankle"]):
            angles.left_knee = calculate_angle(
                (keypoints["Left Hip"].x, keypoints["Left Hip"].y),
                (keypoints["Left Knee"].x, keypoints["Left Knee"].y),
                (keypoints["Left Ankle"].x, keypoints["Left Ankle"].y),
            )

        return FramePose(
            frame_number=frame_number,
            timestamp=timestamp,
            keypoints=keypoints,
            angles=angles,
            confidence=float(np.mean(confidences)),
        )
