"""
Frame Extractor Module
Trích xuất frames từ video cho LLM analysis
"""

import base64
import os
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image


@dataclass
class ExtractedFrame:
    """Represents a single extracted frame"""
    frame_number: int
    timestamp: float  # seconds
    image: Image.Image
    base64_data: str


@dataclass
class VideoInfo:
    """Video metadata"""
    path: str
    total_frames: int
    fps: float
    duration: float  # seconds
    width: int
    height: int


def get_video_info(video_path: str) -> VideoInfo:
    """
    Get video metadata

    Args:
        video_path: Path to video file

    Returns:
        VideoInfo object with video metadata

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be read
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Count actual readable frames (metadata can be unreliable)
    total_frames = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        total_frames += 1
    
    duration = total_frames / fps if fps > 0 else 0.0

    cap.release()

    return VideoInfo(
        path=video_path,
        total_frames=total_frames,
        fps=fps,
        duration=duration,
        width=width,
        height=height
    )


def frame_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """
    Convert PIL Image to base64 string

    Args:
        image: PIL Image object
        format: Image format (JPEG, PNG)
        quality: JPEG quality (1-100)

    Returns:
        Base64 encoded string format: data:image/jpeg;base64,{base64_string}
    """
    buffered = BytesIO()
    image.save(buffered, format=format, quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"


def extract_frames(
    video_path: str,
    num_frames: int = 8,
    max_width: int = 1024,
    sampling_mode: str = "uniform"  # "uniform" | "keyframes"
) -> List[ExtractedFrame]:
    """
    Extract frames from video

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (default 8)
        max_width: Maximum width for resizing (default 1024)
        sampling_mode:
            - "uniform": Extract frames evenly distributed
            - "keyframes": Focus on start, middle, end

    Returns:
        List of ExtractedFrame objects

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be processed
    """
    info = get_video_info(video_path)
    
    if info.total_frames == 0:
        raise ValueError("Video has no frames")

    max_idx = max(0, info.total_frames - 5)
    # Determine frame indices to extract
    if sampling_mode == "uniform":
        indices = np.linspace(0, max_idx, num_frames).astype(int)
    elif sampling_mode == "keyframes":
        if num_frames < 3:
            indices = np.linspace(0, max_idx, num_frames).astype(int)
        else:
            # 25% start, 50% middle, 25% end distribution roughly
            num_start = max(1, num_frames // 4)
            num_end = max(1, num_frames // 4)
            num_mid = num_frames - num_start - num_end
            
            start_indices = np.linspace(0, int(0.1 * info.total_frames), num_start).astype(int)
            mid_start = int(0.4 * info.total_frames)
            mid_end = int(0.6 * info.total_frames)
            mid_indices = np.linspace(mid_start, mid_end, num_mid).astype(int)
            end_start = int(0.9 * info.total_frames)
            end_indices = np.linspace(end_start, max_idx, num_end).astype(int)
            
            indices = np.concatenate([start_indices, mid_indices, end_indices])
            indices = np.unique(indices)
            indices.sort()
            
            # If unique results in fewer frames, fill in with uniform
            if len(indices) < num_frames:
                all_indices = np.linspace(0, max_idx, num_frames).astype(int)
                indices = np.unique(np.concatenate([indices, all_indices]))
                indices.sort()
                indices = indices[:num_frames]
    else:
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")

    extracted_frames: List[ExtractedFrame] = []
    
    cap = cv2.VideoCapture(video_path)
    
    # Convert indices to a set for O(1) lookup
    indices_set = set(indices.tolist())
    target_indices = sorted(indices_set)
    
    # Use sequential reading for more reliable frame extraction
    current_frame = 0
    while current_frame <= max(target_indices):
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame in indices_set:
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            # Resize if width > max_width
            if image.width > max_width:
                aspect_ratio = image.height / image.width
                new_height = int(max_width * aspect_ratio)
                image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
                
            timestamp = current_frame / info.fps if info.fps > 0 else 0.0
            base64_data = frame_to_base64(image)
            
            extracted_frames.append(ExtractedFrame(
                frame_number=current_frame,
                timestamp=timestamp,
                image=image,
                base64_data=base64_data
            ))
        
        current_frame += 1
        
    cap.release()
    
    # Slice to exact num_frames if we got more due to concatenations
    return extracted_frames[:num_frames]
