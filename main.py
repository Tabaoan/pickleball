import cv2
import base64
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load API Key từ file .env
load_dotenv()
api_key = os.getenv("OPENAI__API_KEY") 

if not api_key:
    raise ValueError("Chưa tìm thấy OPENAI_API_KEY trong file .env")

client = OpenAI(api_key=api_key)

# --- CẤU HÌNH GIÁ (PRICING CONFIG) ---
MODEL_PRICING = {
    "gpt-5.2":  {"input": 1.75, "output": 14},
    "gpt-4o":   {"input": 2.50, "output": 10.00}, 
}

# Chọn model bạn muốn dùng
SELECTED_MODEL = "gpt-5.2" 

def process_video_frames(video_path, seconds_per_frame=0.5):
    """
    Hàm này đọc video và trích xuất khung hình.
    seconds_per_frame: Lấy mẫu mỗi 0.5 giây một lần.
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Không thể mở video: {video_path}")
        return None, 0

    base64Frames = []
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    
    if fps == 0:
        return None, 0

    frame_step = int(fps * seconds_per_frame) 
    if frame_step == 0: frame_step = 1
    
    current_frame = 0
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        if current_frame % frame_step == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

        current_frame += 1

    video.release()
    print(f"Đã trích xuất {len(base64Frames)} frames từ video dài {video_duration:.1f}s.")
    return base64Frames, video_duration

def calculate_cost(usage, model_name):
    """
    Tính toán chi phí dựa trên usage trả về từ API
    """
    prompt_tokens = usage.prompt_tokens     
    completion_tokens = usage.completion_tokens 
    
    # Lấy bảng giá, nếu model không có trong list thì mặc định giá 0
    price_info = MODEL_PRICING.get(model_name, {"input": 0, "output": 0})
    
    # Công thức: (Số token / 1,000,000) * Giá
    input_cost = (prompt_tokens / 1_000_000) * price_info["input"]
    output_cost = (completion_tokens / 1_000_000) * price_info["output"]
    total_cost = input_cost + output_cost
    
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": usage.total_tokens,
        "total_cost_usd": total_cost,
        "model_used": model_name
    }

def analyze_pickleball_video(video_path):
    print("Đang xử lý video... Vui lòng chờ.")
    
    frames, duration = process_video_frames(video_path, seconds_per_frame=0.5)
    
    if not frames:
        return None, None
    
    # --- PROMPT ĐÃ ĐƯỢC CẬP NHẬT ---
    system_prompt = """
    Bạn là một huấn luyện viên Pickleball chuyên nghiệp sử dụng AI.
    Nhiệm vụ của bạn là phân tích chuỗi hình ảnh trích từ video người chơi và đánh giá kỹ thuật theo từng động tác.

    YÊU CẦU ĐẦU RA:
    - Chỉ trả về JSON thuần túy (không markdown, không giải thích ngoài JSON).
    - Phân tích theo từng "move" (động tác/pha bóng) với mốc thời gian rõ ràng.

    CÁCH PHÂN TÍCH LỖI:
    - Nếu động tác SAI (is_correct = false), bạn phải cung cấp 2 thông tin cải thiện riêng biệt:
      1. correction_method: Giải thích kỹ thuật đúng cần thay đổi (Ví dụ: "Xoay hông sớm hơn", "Giữ mặt vợt mở").
      2. drill_suggestion: Một bài tập cụ thể để sửa lỗi này (Ví dụ: "Tập Wall Drills 5 phút", "Shadow Swing trước gương 20 lần").

    CẤU TRÚC JSON MONG MUỐN:
    {
      "analysis_summary": "Nhận xét tổng quan về trình độ, điểm mạnh yếu.",
      "moves": [
        {
          "move_name": "Tên động tác (VD: Forehand Drive, Dinking...)",
          "start_time": "MM:SS",
          "end_time": "MM:SS",
          "is_correct": false,
          "error_details": "Mô tả chi tiết lỗi sai dựa trên hình ảnh (VD: Cổ tay bị gập quá mức).",
          "correction_method": "Hướng dẫn kỹ thuật để sửa lỗi (Lý thuyết).",
          "drill_suggestion": "Tên bài tập và hướng dẫn tập luyện cụ thể (Thực hành)."
        }
      ]
    }

    QUY TẮC:
    - Nếu is_correct = true: Các trường error_details, correction_method, drill_suggestion để chuỗi rỗng "".
    - Nếu is_correct = false: BẮT BUỘC phải điền đầy đủ 3 trường trên.
    """

    print(f"Đang gửi dữ liệu lên OpenAI ({SELECTED_MODEL})...")
    
    try:
        response = client.chat.completions.create(
            model=SELECTED_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        "Đây là video quay chậm các động tác của tôi. Hãy phân tích giúp tôi.",
                        *map(lambda x: {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{x}"}}, frames),
                    ],
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        # 1. Lấy nội dung trả về
        result_content = response.choices[0].message.content
        json_result = json.loads(result_content)

        # 2. Lấy thông tin Usage và tính tiền
        cost_report = calculate_cost(response.usage, SELECTED_MODEL)

        return json_result, cost_report

    except Exception as e:
        print(f"Có lỗi xảy ra khi gọi API: {e}")
        return None, None

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    video_file = r"C:\Users\tabao\Downloads\2.mp4"

    if os.path.exists(video_file):
        analysis_result, cost_info = analyze_pickleball_video(video_file)
        
        if analysis_result:
            print("\n" + "="*40)
            print(" KẾT QUẢ PHÂN TÍCH CHI PHÍ & TOKEN")
            print("="*40)
            print(f"Model sử dụng:      {cost_info['model_used']}")
            print(f"Input Tokens (Ảnh): {cost_info['prompt_tokens']:,}")
            print(f"Output Tokens:      {cost_info['completion_tokens']:,}")
            print(f"Tổng Tokens:        {cost_info['total_tokens']:,}")
            print("-" * 40)
            print(f"CHI PHÍ ƯỚC TÍNH:   ${cost_info['total_cost_usd']:.6f}")
            print("="*40 + "\n")

            # In kết quả JSON (tóm tắt)
            print("--- KẾT QUẢ CHUYÊN MÔN ---")
            print(json.dumps(analysis_result, indent=4, ensure_ascii=False))
            
            # Lưu file
            with open("gpt_analysis_result.json", "w", encoding="utf-8") as f:
                json.dump(analysis_result, f, indent=4, ensure_ascii=False)
            
            # Lưu file log chi phí riêng
            with open("cost_log.json", "w", encoding="utf-8") as f:
                json.dump(cost_info, f, indent=4)
                
            print("\nĐã lưu kết quả vào file.")
    else:
        print("Không tìm thấy file video. Vui lòng kiểm tra đường dẫn.")