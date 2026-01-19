#!/usr/bin/env python3
"""
CLI entry point for video analysis
Usage: python analyze_video.py short_pickle.mp4
"""

import sys
from video_analyzer import analyze_pickleball_video, AnalysisConfig


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_video.py <video_path> [--lang=vi|en]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    language = "vi"
    
    for arg in sys.argv[2:]:
        if arg.startswith("--lang="):
            language = arg.split("=")[1]
    
    try:
        config = AnalysisConfig(language=language, verbose=True)
        result = analyze_pickleball_video(video_path, config)
        
        print("\n" + "="*60)
        print("📊 KẾT QUẢ PHÂN TÍCH PICKLEBALL")
        print("="*60)
        print(result.to_report())
        
    except Exception as e:
        print(f"❌ Error analyzing video: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
