import os
import json
import glob
import time
from openai import OpenAI
from dotenv import load_dotenv

# ==========================================
# 1. CẤU HÌNH & KHỞI TẠO
# ==========================================
load_dotenv(override=True)
api_key = os.getenv("OPENAI__API_KEY") 

if not api_key:
    print(" Lỗi: Chưa tìm thấy API Key.")
    exit()

client = OpenAI(api_key=api_key)

# Folder chứa các file JSON động tác (Output của bước trước)
DATA_FOLDER = r"C:\Users\tabao\OneDrive\Desktop\Vitex\test_keypoint\moves_data"
STANDARDS_PATH = r"C:\Users\tabao\OneDrive\Desktop\Vitex\test_keypoint\dictionary_official.json"
REPORT_FOLDER = r"C:\Users\tabao\OneDrive\Desktop\Vitex\test_keypoint\reports" 

# Tạo folder báo cáo nếu chưa có
if not os.path.exists(REPORT_FOLDER):
    os.makedirs(REPORT_FOLDER)

# ==========================================
# 2. HÀM TIỆN ÍCH
# ==========================================
def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f" Lỗi đọc file {path}: {e}")
        return None

def optimize_payload(frames_data):
    """
    Nén dữ liệu: Chỉ giữ timestamp và angles, làm tròn số để tiết kiệm token.
    """
    optimized = []
    for item in frames_data:
        compact = {
            "t": item.get("timestamp", 0),
            "a": {}
        }
        # Chỉ lấy angles, bỏ keypoints
        if "angles" in item:
            for joint, angle in item["angles"].items():
                compact["a"][joint] = round(angle, 1)
        
        optimized.append(compact)
    return optimized

def analyze_single_file(file_path, standards_data):
    """Gửi 1 file JSON lên GPT để phân tích"""
    file_name = os.path.basename(file_path)
    print(f"\n Đang xử lý file: {file_name}...")
    
    user_data = load_json(file_path)
    if not user_data: return

    # Lấy thông tin metadata từ file JSON
    move_type = user_data.get("type", "Unknown")
    move_id = user_data.get("id", "0")
    raw_frames = user_data.get("frames", [])

    # Tối ưu dữ liệu frames
    compact_frames = optimize_payload(raw_frames)
    
    print(f"📡 Gửi {len(compact_frames)} frames lên Server...")

    # --- PROMPT ---
    prompt = f"""
    Bạn là Huấn luyện viên Pickleball AI. Hãy chấm điểm kỹ thuật cho pha bóng này.

    THÔNG TIN PHA BÓNG:
    - Loại động tác (AI nhận diện sơ bộ): {move_type}
    - ID: {move_id}

    DỮ LIỆU TIÊU CHUẨN (Dictionary):
    {json.dumps(standards_data, ensure_ascii=False)}

    DỮ LIỆU THỰC TẾ (Time-series của người chơi):
    {json.dumps(compact_frames, ensure_ascii=False)}

    --- YÊU CẦU PHÂN TÍCH ---
    Hãy kiểm tra xem người chơi có thực hiện ĐÚNG kỹ thuật của động tác "{move_type}" hay không dựa trên Dictionary chuẩn.

    HÃY TRẢ LỜI NGẮN GỌN, SÚC TÍCH THEO MẪU SAU:

    ##  PHÂN TÍCH PHA BÓNG #{move_id} ({move_type})
    
    1. **Độ ổn định:** (Nhận xét về sự mượt mà của biểu đồ góc)
    2. **Lỗi vi phạm (Timeline):**
       - **t=[Giây]:** [Tên lỗi] (Góc đo được: ... | Chuẩn: ...).
       (Chỉ liệt kê nếu có lỗi nghiêm trọng vượt ngưỡng)
    
    3. **Kết luận:** [ĐẠT / KHÔNG ĐẠT]
    4. **Lời khuyên:** [1 câu sửa lỗi]
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia phân tích dữ liệu thể thao khắt khe."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        report_content = response.choices[0].message.content
        
        # In ra màn hình
        print("-" * 40)
        print(report_content)
        print("-" * 40)

        # Lưu ra file text
        report_filename = f"report_{file_name.replace('.json', '.txt')}"
        save_path = os.path.join(REPORT_FOLDER, report_filename)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        print(f" Đã lưu báo cáo: {save_path}")

    except Exception as e:
        print(f" Lỗi API khi xử lý file {file_name}: {e}")

# ==========================================
# 3. CHƯƠNG TRÌNH CHÍNH
# ==========================================
def main():
    print(" BẮT ĐẦU QUÉT FOLDER DỮ LIỆU...")
    
    # 1. Load tiêu chuẩn
    standards_data = load_json(STANDARDS_PATH)
    if not standards_data:
        print(" Không tìm thấy file Dictionary chuẩn.")
        return

    # 2. Tìm tất cả file json bắt đầu bằng 'action_'
    # (Do code trước lưu file dạng action_1_xxx.json)
    search_pattern = os.path.join(DATA_FOLDER, "action_*.json")
    json_files = glob.glob(search_pattern)

    if not json_files:
        print(f"Không tìm thấy file JSON nào trong {DATA_FOLDER}")
        return

    print(f" Tìm thấy {len(json_files)} pha bóng cần phân tích.")

    # 3. Lặp qua từng file
    for file_path in json_files:
        analyze_single_file(file_path, standards_data)
        
        # Nghỉ 1 chút để tránh rate limit (nếu dùng free tier)
        time.sleep(1) 

    print("\n HOÀN TẤT TOÀN BỘ QUÁ TRÌNH PHÂN TÍCH!")
    print(f" Xem kết quả tại folder: {REPORT_FOLDER}")

if __name__ == "__main__":
    main()