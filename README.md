---
title: Vietnamese Food Recognition
emoji: 🍜
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# 🍜 Vietnamese Food Recognition & Analysis (YOLO & Gemini AI)

Hệ thống nhận diện món ăn Việt Nam sử dụng mô hình **YOLOv8** kết hợp độc quyền với **Google Gemini AI**, tích hợp sẵn cơ chế **tự động xoay mô hình (Model Auto-Rotation / Failover)** để luôn đảm bảo tính khả dụng cao nhất kể cả khi gặp lỗi giới hạn tốc độ (Rate Limit 429), hết hạn ngạch hoặc lỗi dịch vụ.

---

## ✨ Tính năng nổi bật
- 🎯 **Nhận diện món ăn (YOLOv8)**: Nhận diện các món ăn Việt Nam phổ biến (Phở, Bún bò Huế, Bánh mì, Bánh xèo, Bún chả, Bánh cuốn, Bún đậu...).
- ✨ **Google Gemini AI với cơ chế xoay vòng thông minh**:
  - Hỗ trợ toàn bộ danh sách mô hình Gemini thế hệ mới (Gemini 3.8 Flash, Gemini 3.7 Flash, Gemini 3.6 Flash, Gemini 3.5 Flash, Gemini 2.5 Flash, Nano Banana...).
  - **Tự động chuyển model**: Khi một model gặp lỗi (429 Rate Limit, Quota Exceeded, 503 Overload, 404...), hệ thống lập tức tự động xoay sang model tiếp theo trong danh sách mà không làm gián đoạn trải nghiệm người dùng.
- 🖥️ **Web UI trực quan**: Giao diện kiểm tra nhanh ngay trên trình duyệt với tính năng vẽ Bounding Box trực tiếp trên ảnh và hiển thị thông tin món ăn từ Gemini.
- 🚀 **RESTful API**: Tích hợp Swagger UI (`/docs`) dễ dàng kết nối với ứng dụng Mobile hoặc Web.

---

## 📡 Danh sách API Endpoints

| Phương thức | Đường dẫn | Chức năng |
|---|---|---|
| `GET` | `/` | Giao diện Web trực quan (hoặc JSON nếu gọi từ API client) |
| `GET` | `/health` | Kiểm tra trạng thái YOLO model và Gemini active model |
| `GET` | `/docs` | Tài liệu Swagger UI kiểm thử API trực tiếp |
| `POST` | `/predict` | Nhận diện món ăn trong ảnh (YOLO Detection) |
| `POST` | `/food-info` | Lấy thông tin ẩm thực chi tiết từ Gemini theo tên món |
| `POST` | `/analyze` | Phân tích toàn diện: YOLO nhận diện + Gemini phân tích chi tiết |

---

## 🔐 Cấu hình Biến môi trường trên Hugging Face Spaces (Variables and secrets)

Vào **Settings** -> **Variables and secrets** -> Thêm Secret:

### 1. Bắt buộc để dùng Gemini AI:
- `GEMINI_API_KEY` (hoặc `GOOGLE_API_KEY`): API key lấy miễn phí tại [Google AI Studio](https://aistudio.google.com/).

### 2. Tùy chọn cấu hình thêm (Optional):
- `GEMINI_MODEL`: Mô hình Gemini bạn muốn ưu tiên chạy đầu tiên (ví dụ: `gemini-3.8-flash` hoặc `gemini-2.5-flash`).
- `GEMINI_BASE_URL`: Đổi URL endpoint nếu bạn sử dụng proxy hoặc gateway riêng (mặc định: Google AI Studio endpoint).
- `GEMINI_TIMEOUT`: Thời gian chờ tối đa cho mỗi model trước khi xoay sang model kế tiếp (mặc định: `15.0` giây).
- `CONFIDENCE_THRESHOLD`: Ngưỡng tự tin phát hiện của YOLO (mặc định: `0.25`).
- `IMAGE_SIZE`: Kích cỡ ảnh resize cho YOLO (mặc định: `640`).

---

## 🔄 Danh sách mô hình Gemini xoay vòng tự động

Hệ thống được cấu hình sẵn danh sách xoay vòng theo thứ tự ưu tiên tối ưu cho sinh văn bản & JSON:
1. `gemini-3.8-flash`
2. `gemini-3.7-flash`
3. `gemini-3.6-flash`
4. `gemini-3.5-flash`
5. `gemini-2.5-flash`
6. `gemini-3.5-flash-lite`
7. `gemini-3.1-flash-lite`
8. `gemini-3.1-pro-preview`
9. `gemini-3-flash-preview`
10. `gemini-omni-1.1-flash`
11. `gemini-3.8-live-extended-thinking`
12. `gemini-3.8-live`
13. `gemini-3.1-flash-live-preview`
14. `gemini-2.5-flash-native-audio-preview-12-2025`
15. `gemini-3.5-live-translate-preview`
16. `gemini-3.1-flash-image`
17. `gemini-3.1-flash-lite-image`
18. `gemini-3-pro-image`
19. `gemini-2.5-flash-image`
20. `gemini-3.8-flash-tts`
21. `gemini-3.8-flash-lite-tts`
22. `gemini-3.1-flash-tts-preview`
23. `gemini-2.5-flash-preview-tts`
24. `gemini-3.5-transcribe`
25. `gemini-3.5-transcribe-live`

---

## 💻 Chạy thử nghiệm trên máy cục bộ (Local)

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Thiết lập API Key Gemini
$env:GEMINI_API_KEY="your-gemini-api-key"

# Chạy server FastAPI
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```
Truy cập: `http://localhost:7860`
