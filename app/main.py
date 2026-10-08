import io
import os
import sys
from pathlib import Path
from typing import List, Optional

# Đảm bảo Python luôn tìm thấy các package trong app dù chạy từ đâu
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, ImageOps

try:
    from app.detector import FoodDetector, format_food_name
    from app.llm import FoodLLM
    from app.schemas import (
        AnalyzeResponse,
        FoodInfoRequest,
        FoodInfoResponse,
        PredictResponse,
    )
except ImportError:
    from detector import FoodDetector, format_food_name
    from llm import FoodLLM
    from schemas import (
        AnalyzeResponse,
        FoodInfoRequest,
        FoodInfoResponse,
        PredictResponse,
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Vietnamese Food Recognition API",
    description=(
        "Hệ thống nhận diện món ăn Việt Nam sử dụng mô hình học máy thị giác máy tính "
        "kết hợp hệ thống tri thức ẩm thực thông minh."
    ),
    version="1.0.0",
)


# ============================================================
# LOAD YOLO MODEL & CONFIGURATION
# ============================================================

MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))

detector: Optional[FoodDetector] = None
detector_error: Optional[str] = None


def get_detector() -> Optional[FoodDetector]:
    global detector, detector_error
    if detector is not None:
        return detector
    try:
        detector = FoodDetector(
            model_path=MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            image_size=IMAGE_SIZE,
        )
        detector_error = None
        print(f"[Detector] Da tai thanh cong YOLO model tu: {MODEL_PATH}")
    except Exception as e:
        detector_error = str(e)
        print(f"[Detector] Chua the khoi tao FoodDetector: {e}".encode("ascii", errors="replace").decode("ascii"))
    return detector


# Khởi tạo detector lúc khởi động
get_detector()


# ============================================================
# LOAD AI KNOWLEDGE SYSTEM
# ============================================================

llm: Optional[FoodLLM] = None
llm_error: Optional[str] = None


def get_llm() -> Optional[FoodLLM]:
    """
    Lazy load hệ thống AI tri thức món ăn khi có API Key.
    """
    global llm, llm_error
    if llm is not None:
        return llm

    has_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if has_api_key:
        try:
            llm = FoodLLM()
            llm_error = None
            print("[AI System] Khoi tao he thong tri thuc am thuc thanh cong.")
        except Exception as e:
            llm_error = str(e)
            print(f"[AI System] Khong the khoi tao he thong: {e}".encode("ascii", errors="replace").decode("ascii"))

    return llm


# Thử load AI lúc khởi động
get_llm()


# ============================================================
# HELPER FUNCTIONS (TỐI ƯU BỘ NHỚ TRÁNH LỖI 502 TRÊN RENDER)
# ============================================================

async def read_image(file: UploadFile) -> tuple[Image.Image, tuple[int, int]]:
    """
    Đọc file upload và chuyển thành PIL Image chuẩn RGB.
    Tự động xoay chuẩn theo EXIF từ máy ảnh điện thoại và
    giới hạn kích thước tối đa (max 1280px) để chống tràn RAM (OOM) trên Render 512MB.
    Trả về (image_đã_tối_ưu, (orig_w, orig_h)) để giữ độ chính xác tuyệt đối của Bounding Box.
    """
    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Không xác định được loại file.",
        )

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/jpg",
    }

    if file.content_type.lower() not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Chỉ hỗ trợ file ảnh định dạng JPG, JPEG, PNG hoặc WEBP.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="File ảnh rỗng.",
        )

    # Giới hạn kích thước file upload 15MB
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Dung lượng file ảnh quá lớn (vui lòng chọn ảnh dưới 15MB).",
        )

    try:
        raw_image = Image.open(io.BytesIO(content))
        image = ImageOps.exif_transpose(raw_image)
        if image is None:
            image = raw_image
        image = image.convert("RGB")

        orig_size = image.size  # (orig_w, orig_h) kích thước thực tế sau khi đã chuẩn hóa EXIF

        # Giữ độ phân giải sắc nét (max 2048px) cho chi tiết hạt cơm và thức ăn, vẫn kiểm soát RAM an toàn
        max_dim = 2048
        if max(image.size) > max_dim:
            image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        return image, orig_size
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="File không phải ảnh hợp lệ.",
        ) from exc


# ============================================================
# WEB UI TEMPLATE (ĐA NGÔN NGỮ VI/EN, LỊCH SỬ LOCAL STORAGE & THỐNG KÊ)
# ============================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Nhận Diện & Phân Tích Món Ăn Việt Nam</title>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <!-- Lucide Icons Library -->
  <script src="https://unpkg.com/lucide@latest"></script>
  <style>
    :root {
      --primary: #e63946;
      --primary-hover: #d62828;
      --primary-light: #fef2f2;
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --text: #0f172a;
      --text-muted: #64748b;
      --border: #e2e8f0;
      --radius: 16px;
      --shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      padding: 32px 16px;
    }

    .container {
      max-width: 1120px;
      margin: 0 auto;
    }

    /* LUCIDE ICONS */
    .lucide {
      width: 18px;
      height: 18px;
      vertical-align: -3px;
      display: inline-block;
      stroke-width: 2.2;
    }

    .upload-icon {
      width: 44px;
      height: 44px;
      color: #94a3b8;
      margin-bottom: 8px;
      stroke-width: 1.8;
    }

    .empty-icon {
      width: 44px;
      height: 44px;
      color: #cbd5e1;
      margin-bottom: 8px;
      stroke-width: 1.8;
    }

    /* HEADER & LANGUAGE SWITCHER */
    header {
      text-align: center;
      margin-bottom: 24px;
    }

    header h1 {
      font-size: 2.1rem;
      font-weight: 800;
      color: var(--text);
      display: flex;
      align-items: center;
      justify-content: center;
      letter-spacing: -0.02em;
    }

    .lang-switch-wrap {
      display: flex;
      justify-content: center;
      margin-top: 14px;
    }

    .lang-switch {
      display: inline-flex;
      background: #ffffff;
      border: 1px solid var(--border);
      padding: 3px;
      border-radius: 30px;
      box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
      gap: 2px;
    }

    .lang-btn {
      background: transparent;
      border: none;
      padding: 6px 16px;
      border-radius: 20px;
      font-size: 0.84rem;
      font-weight: 600;
      color: var(--text-muted);
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      transition: all 0.2s;
      font-family: inherit;
    }

    .lang-btn:hover {
      color: var(--text);
    }

    .lang-btn.active {
      background: var(--primary);
      color: #ffffff;
      box-shadow: 0 2px 8px rgba(230, 57, 70, 0.25);
    }

    /* MAIN GRID (INPUT / OUTPUT) */
    .main-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      align-items: start;
    }

    @media (max-width: 860px) {
      .main-grid { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--card-bg);
      border-radius: var(--radius);
      padding: 24px;
      box-shadow: var(--shadow);
      border: 1px solid var(--border);
    }

    .card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 18px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border);
    }

    .card-title {
      font-size: 1.15rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--text);
    }

    /* LEFT: UPLOAD BOX & PREVIEW */
    .upload-zone {
      border: 2px dashed #cbd5e1;
      border-radius: 14px;
      padding: 38px 20px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
      background: #f8fafc;
    }
    .upload-zone:hover, .upload-zone.dragover {
      border-color: var(--primary);
      background: var(--primary-light);
    }

    .preview-box {
      border-radius: 14px;
      border: 1px solid var(--border);
      background: #f8fafc;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    .preview-img-wrap {
      width: 100%;
      height: 280px;
      max-height: 280px;
      background: #0f172a;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }

    .preview-img-wrap img {
      max-width: 100%;
      max-height: 280px;
      object-fit: contain;
      display: block;
    }

    .preview-footer {
      padding: 12px 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: #ffffff;
      border-top: 1px solid var(--border);
    }

    .file-name {
      font-size: 0.85rem;
      font-weight: 600;
      color: #334155;
      max-width: 220px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .btn-change {
      background: #f1f5f9;
      color: #334155;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      padding: 6px 12px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }
    .btn-change:hover {
      background: #e2e8f0;
      color: #0f172a;
    }

    /* CONTROLS */
    .controls {
      margin-top: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .slider-group {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
    }
    .slider-group label { font-size: 0.88rem; font-weight: 600; color: #334155; }
    .conf-badge {
      font-size: 0.85rem;
      font-weight: 700;
      color: var(--primary);
      background: var(--primary-light);
      padding: 2px 8px;
      border-radius: 6px;
    }

    input[type=range] {
      width: 100%;
      accent-color: var(--primary);
      cursor: pointer;
    }

    .mode-select {
      display: flex;
      flex-direction: column;
      gap: 8px;
      background: #f8fafc;
      padding: 12px 14px;
      border-radius: 12px;
      border: 1px solid var(--border);
    }

    .mode-option {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.88rem;
      font-weight: 500;
      cursor: pointer;
    }
    .mode-option input { cursor: pointer; accent-color: var(--primary); }

    .btn-submit {
      background: var(--primary);
      color: white;
      border: none;
      border-radius: 12px;
      padding: 14px 20px;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      box-shadow: 0 4px 14px rgba(230, 57, 70, 0.25);
      transition: all 0.2s;
    }
    .btn-submit:hover { background: var(--primary-hover); }
    .btn-submit:disabled { opacity: 0.6; cursor: not-allowed; box-shadow: none; }

    /* RIGHT: RESULT CONTAINER (CÙNG KÍCH THƯỚC CHUẨN VỚI Ô BÊN TRÁI) */
    .result-canvas-wrap {
      width: 100%;
      height: 280px;
      max-height: 280px;
      background: #0f172a;
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    canvas {
      max-width: 100%;
      max-height: 280px;
      object-fit: contain;
      display: block;
    }

    .empty-state {
      text-align: center;
      padding: 48px 16px;
      color: #94a3b8;
    }
    .empty-state p { font-size: 0.95rem; font-weight: 500; }

    .tags-container {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 16px 0;
    }

    .detection-chip {
      padding: 6px 12px;
      border-radius: 8px;
      font-weight: 700;
      font-size: 0.88rem;
      display: inline-flex;
      align-items: center;
    }

    /* CULINARY KNOWLEDGE CARDS */
    .info-card {
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px;
      margin-top: 14px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    }

    .info-card-header {
      display: flex;
      align-items: center;
      margin-bottom: 14px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border);
    }

    .info-card-header h3 {
      font-size: 1.2rem;
      font-weight: 800;
      color: #0f172a;
      display: flex;
      align-items: center;
    }

    .info-item {
      margin-bottom: 12px;
      font-size: 0.92rem;
      line-height: 1.55;
    }
    .info-item strong {
      color: #334155;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      margin-bottom: 2px;
    }

    .ingredients-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 6px;
    }
    .ingredient-badge {
      background: #ffffff;
      border: 1px solid #cbd5e1;
      padding: 3px 10px;
      border-radius: 6px;
      font-size: 0.82rem;
      font-weight: 600;
      color: #334155;
    }

    /* HISTORY & STATS SECTION */
    .history-grid {
      display: grid;
      grid-template-columns: 310px 1fr;
      gap: 24px;
      margin-top: 24px;
      align-items: start;
    }

    @media (max-width: 860px) {
      .history-grid { grid-template-columns: 1fr; }
    }

    /* STATS CARD */
    .stats-card {
      background: var(--card-bg);
    }

    .stats-body {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .stat-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #f8fafc;
      padding: 12px 14px;
      border-radius: 12px;
      border: 1px solid var(--border);
    }

    .stat-label {
      font-size: 0.88rem;
      font-weight: 600;
      color: #334155;
    }

    .stat-number {
      font-size: 1.45rem;
      font-weight: 800;
      color: var(--primary);
    }

    .stat-recent-section {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .recent-foods-list {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    .recent-food-badge {
      background: #f1f5f9;
      border: 1px solid var(--border);
      padding: 4px 10px;
      border-radius: 8px;
      font-size: 0.84rem;
      font-weight: 600;
      color: #1e293b;
      display: inline-flex;
      align-items: center;
    }

    .stat-empty {
      font-size: 0.84rem;
      color: #94a3b8;
      font-style: italic;
    }

    /* HISTORY CARD */
    .history-card {
      background: var(--card-bg);
    }

    .history-count-badge {
      font-size: 0.78rem;
      font-weight: 700;
      background: var(--primary-light);
      color: var(--primary);
      padding: 2px 8px;
      border-radius: 12px;
      margin-left: 6px;
    }

    .btn-clear-history {
      background: #fef2f2;
      color: #dc2626;
      border: 1px solid #fecaca;
      border-radius: 8px;
      padding: 6px 12px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }
    .btn-clear-history:hover {
      background: #dc2626;
      color: white;
    }

    .history-list {
      max-height: 480px;
      overflow-y: auto;
      padding-right: 4px;
    }

    .history-empty {
      text-align: center;
      padding: 40px 16px;
      color: #94a3b8;
    }

    .history-empty-icon {
      width: 40px;
      height: 40px;
      color: #cbd5e1;
      margin-bottom: 8px;
      stroke-width: 1.8;
    }

    .history-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 12px;
      border-bottom: 1px solid var(--border);
      transition: background 0.15s ease;
      gap: 14px;
    }
    .history-item:last-child {
      border-bottom: none;
    }
    .history-item:hover {
      background: #f8fafc;
      border-radius: 10px;
    }

    .history-main {
      display: flex;
      align-items: center;
      gap: 14px;
      flex: 1;
      min-width: 0;
    }

    .history-thumb {
      width: 62px;
      height: 62px;
      border-radius: 10px;
      object-fit: cover;
      border: 1px solid var(--border);
      background: #0f172a;
      flex-shrink: 0;
    }

    .history-details {
      display: flex;
      flex-direction: column;
      gap: 3px;
      min-width: 0;
    }

    .history-food-title {
      font-size: 1.02rem;
      font-weight: 700;
      color: var(--text);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .history-meta-row {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 0.82rem;
      color: var(--text-muted);
    }

    .history-conf-tag {
      font-weight: 700;
      color: #059669;
      background: #ecfdf5;
      padding: 1px 7px;
      border-radius: 6px;
      font-size: 0.8rem;
    }

    .history-date {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      font-size: 0.8rem;
      color: #64748b;
    }

    .history-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-shrink: 0;
    }

    .btn-history-view {
      background: var(--primary-light);
      color: var(--primary);
      border: 1px solid rgba(230, 57, 70, 0.2);
      border-radius: 8px;
      padding: 7px 12px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 5px;
      transition: all 0.15s;
    }
    .btn-history-view:hover {
      background: var(--primary);
      color: white;
    }

    .btn-history-del {
      background: #f1f5f9;
      color: #64748b;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 7px 10px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      transition: all 0.15s;
    }
    .btn-history-del:hover {
      background: #fee2e2;
      color: #dc2626;
      border-color: #fca5a5;
    }

    /* TOAST NOTIFICATION */
    .toast-box {
      position: fixed;
      bottom: 24px;
      right: 24px;
      background: #0f172a;
      color: #ffffff;
      padding: 12px 20px;
      border-radius: 10px;
      font-size: 0.88rem;
      font-weight: 600;
      box-shadow: 0 10px 25px rgba(0,0,0,0.2);
      display: flex;
      align-items: center;
      gap: 8px;
      opacity: 0;
      transform: translateY(20px);
      transition: all 0.25s ease;
      z-index: 9999;
      pointer-events: none;
    }
    .toast-box.show {
      opacity: 1;
      transform: translateY(0);
    }

    .spinner {
      border: 3px solid rgba(255,255,255,0.3);
      border-radius: 50%;
      border-top: 3px solid white;
      width: 20px;
      height: 20px;
      animation: spin 1s linear infinite;
    }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>
        <span data-i18n="appTitle">Nhận Diện & Phân Tích Món Ăn Việt Nam</span>
      </h1>
      
      <!-- BỘ CHUYỂN ĐỔI NGÔN NGỮ (VI / EN) -->
      <div class="lang-switch-wrap">
        <div class="lang-switch">
          <button type="button" class="lang-btn active" id="btn-lang-vi" onclick="setLanguage('vi')">
            Tiếng Việt
          </button>
          <button type="button" class="lang-btn" id="btn-lang-en" onclick="setLanguage('en')">
            English
          </button>
        </div>
      </div>
    </header>

    <div class="main-grid" id="main-grid">
      <!-- CỘT TRÁI: DỮ LIỆU ĐẦU VÀO (INPUT) -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">
            <i data-lucide="camera"></i> <span data-i18n="step1Title">1. Tải Ảnh Đầu Vào</span>
          </div>
        </div>

        <!-- Khung chưa chọn ảnh -->
        <div class="upload-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
          <i data-lucide="upload-cloud" class="upload-icon"></i>
          <p><strong data-i18n="uploadTitle">Bấm để chọn ảnh</strong> <span data-i18n="uploadOrDrag">hoặc kéo thả file vào đây</span></p>
          <p style="font-size:0.82rem; color:#94a3b8; margin-top:4px;" data-i18n="uploadSub">Hỗ trợ: JPG, JPEG, PNG, WEBP</p>
        </div>

        <!-- Khung hiển thị ảnh đã chọn bên cột trái -->
        <div class="preview-box" id="preview-box" style="display:none;">
          <div class="preview-img-wrap">
            <img id="input-preview-img" src="" alt="Ảnh đầu vào">
          </div>
          <div class="preview-footer">
            <span class="file-name" id="file-name-text">anh_mon_an.jpg</span>
            <button type="button" class="btn-change" onclick="document.getElementById('file-input').click()">
              <i data-lucide="refresh-cw"></i> <span data-i18n="changeImg">Chọn ảnh khác</span>
            </button>
          </div>
        </div>

        <input type="file" id="file-input" accept="image/*" style="display:none">

        <div class="controls">
          <div>
            <div class="slider-group">
              <label for="conf-slider" data-i18n="confLabel">Độ tin cậy tối thiểu (Confidence):</label>
              <span class="conf-badge" id="conf-val">20%</span>
            </div>
            <input type="range" id="conf-slider" min="0.05" max="0.95" step="0.05" value="0.20" oninput="document.getElementById('conf-val').innerText = Math.round(this.value * 100) + '%'">
          </div>

          <div class="mode-select">
            <label class="mode-option">
              <input type="radio" name="api-mode" value="analyze" checked>
              <span data-i18n="modeAnalyze">Phân tích toàn diện (Nhận diện + Tri thức ẩm thực chi tiết)</span>
            </label>
            <label class="mode-option">
              <input type="radio" name="api-mode" value="predict">
              <span data-i18n="modePredict">Chỉ nhận diện vật thể (YOLO Detection)</span>
            </label>
          </div>

          <button class="btn-submit" id="submit-btn" onclick="processImage()">
            <span id="btn-text" data-i18n="btnSubmit">Bắt Đầu Phân Tích</span>
            <div class="spinner" id="btn-spinner" style="display:none;"></div>
          </button>
        </div>
      </div>

      <!-- CỘT PHẢI: KẾT QUẢ ĐẦU RA (OUTPUT) -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">
            <i data-lucide="layers"></i> <span data-i18n="step2Title">2. Kết Quả Nhận Diện & Phân Tích</span>
          </div>
        </div>

        <!-- Trạng thái trống lúc chưa phân tích -->
        <div class="empty-state" id="empty-state">
          <i data-lucide="image" class="empty-icon"></i>
          <p id="empty-state-text" data-i18n="emptyWait">Vui lòng tải ảnh ở ô bên trái và bấm "Bắt Đầu Phân Tích"</p>
        </div>

        <!-- Khu vực kết quả hiển thị sau khi bấm phân tích -->
        <div id="result-content" style="display:none;">
          <div class="result-canvas-wrap">
            <canvas id="result-canvas"></canvas>
          </div>

          <div class="tags-container" id="tags-container"></div>
          <div id="culinary-info-container"></div>
        </div>
      </div>
    </div>

    <!-- KHU VỰC LỊCH SỬ & THỐNG KÊ NHẬN DIỆN -->
    <div class="history-grid">
      <!-- KHUNG THỐNG KÊ NHỎ -->
      <div class="card stats-card">
        <div class="card-header">
          <div class="card-title">
            <i data-lucide="bar-chart-3"></i> <span data-i18n="statsTitle">Thống Kê Nhận Diện</span>
          </div>
        </div>
        <div class="stats-body">
          <div class="stat-item">
            <span class="stat-label" data-i18n="statsTotal">Tổng số lần nhận diện:</span>
            <span class="stat-number" id="stats-total-count">0</span>
          </div>
          <div class="stat-recent-section">
            <span class="stat-label" data-i18n="statsRecent">Món được nhận diện gần đây:</span>
            <div class="recent-foods-list" id="stats-recent-list">
              <span class="stat-empty" data-i18n="statsNone">Chưa có dữ liệu</span>
            </div>
          </div>
        </div>
      </div>

      <!-- KHUNG LỊCH SỬ NHẬN DIỆN -->
      <div class="card history-card">
        <div class="card-header">
          <div class="card-title">
            <i data-lucide="history"></i> <span data-i18n="historyTitle">Lịch Sử Nhận Diện</span>
            <span class="history-count-badge" id="history-badge">0</span>
          </div>
          <button type="button" class="btn-clear-history" id="btn-clear-history" onclick="clearAllHistory()">
            <i data-lucide="trash-2"></i> <span data-i18n="historyClearAll">Xóa toàn bộ lịch sử</span>
          </button>
        </div>

        <!-- Danh sách mục lịch sử -->
        <div class="history-list" id="history-items-container">
          <div class="history-empty" id="history-empty-placeholder">
            <i data-lucide="clock" class="history-empty-icon"></i>
            <p data-i18n="historyEmpty">Chưa có lịch sử nhận diện nào.</p>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    // BẢNG DỊCH ĐA NGÔN NGỮ (VI / EN)
    const TRANSLATIONS = {
      vi: {
        appTitle: "Nhận Diện & Phân Tích Món Ăn Việt Nam",
        step1Title: "1. Tải Ảnh Đầu Vào",
        uploadTitle: "Bấm để chọn ảnh",
        uploadOrDrag: "hoặc kéo thả file vào đây",
        uploadSub: "Hỗ trợ: JPG, JPEG, PNG, WEBP",
        changeImg: "Chọn ảnh khác",
        confLabel: "Độ tin cậy tối thiểu (Confidence):",
        modeAnalyze: "Phân tích toàn diện (Nhận diện + Tri thức ẩm thực chi tiết)",
        modePredict: "Chỉ nhận diện vật thể (YOLO Detection)",
        btnSubmit: "Bắt Đầu Phân Tích",
        btnProcessing: "Đang xử lý...",
        step2Title: "2. Kết Quả Nhận Diện & Phân Tích",
        emptyWait: 'Vui lòng tải ảnh ở ô bên trái và bấm "Bắt Đầu Phân Tích"',
        emptyReady: 'Ảnh đã sẵn sàng. Nhấn "Bắt Đầu Phân Tích" để nhận diện món ăn!',
        noDetection: "Không tìm thấy món ăn nào với độ tin cậy &ge; {conf}%. Hãy thử kéo giảm thanh Confidence!",
        statsTitle: "Thống Kê Nhận Diện",
        statsTotal: "Tổng số lần nhận diện:",
        statsRecent: "Món được nhận diện gần đây:",
        statsNone: "Chưa có dữ liệu",
        historyTitle: "Lịch Sử Nhận Diện",
        historyClearAll: "Xóa toàn bộ lịch sử",
        historyEmpty: "Chưa có lịch sử nhận diện nào.",
        viewDetail: "Xem chi tiết",
        deleteItem: "Xóa",
        confirmClearAll: "Bạn có chắc chắn muốn xóa toàn bộ lịch sử nhận diện không?",
        confirmDelete: "Bạn có chắc muốn xóa mục này khỏi lịch sử?",
        historyLoaded: "Đã tải lại kết quả từ lịch sử!",
        introLabel: "Giới thiệu:",
        originLabel: "Nguồn gốc & Văn hóa:",
        ingredientsLabel: "Nguyên liệu chính:",
        tasteLabel: "Hương vị đặc trưng:",
        prepLabel: "Cách chế biến & Thưởng thức:",
        noteLabel: "Lưu ý ẩm thực:",
        updating: "Đang cập nhật...",
        noInfo: "Chưa có thông tin.",
        selectFileWarn: "Vui lòng chọn một bức ảnh trước!",
        serverErrNetwork: "Không thể kết nối đến máy chủ (máy chủ đang khởi động lại hoặc mạng gián đoạn). Vui lòng đợi 15-30 giây và thử lại!",
        serverErr502: "Máy chủ đang thức dậy từ chế độ ngủ hoặc tạm quá tải (502 Bad Gateway). Vui lòng đợi 15-30 giây và nhấn thử lại!",
        serverErr504: "Thời gian xử lý quá lâu (504 Gateway Timeout). Vui lòng thử lại với ảnh dung lượng nhỏ hơn!",
        serverErr500: "Máy chủ gặp sự cố nội bộ khi xử lý (500). Vui lòng thử lại!"
      },
      en: {
        appTitle: "Vietnamese Food Recognition & Analysis",
        step1Title: "1. Upload Input Image",
        uploadTitle: "Click to choose an image",
        uploadOrDrag: "or drag & drop file here",
        uploadSub: "Supports: JPG, JPEG, PNG, WEBP",
        changeImg: "Change image",
        confLabel: "Minimum Confidence Score:",
        modeAnalyze: "Comprehensive Analysis (Detection + Detailed Culinary Knowledge)",
        modePredict: "Object Detection Only (YOLO Detection)",
        btnSubmit: "Start Analysis",
        btnProcessing: "Processing...",
        step2Title: "2. Detection & Analysis Results",
        emptyWait: 'Please upload an image on the left and click "Start Analysis"',
        emptyReady: 'Image ready. Click "Start Analysis" to recognize dishes!',
        noDetection: "No food detected with confidence &ge; {conf}%. Try lowering the Confidence slider!",
        statsTitle: "Recognition Statistics",
        statsTotal: "Total recognitions:",
        statsRecent: "Recently recognized dishes:",
        statsNone: "No data available",
        historyTitle: "Recognition History",
        historyClearAll: "Clear All History",
        historyEmpty: "No recognition history yet.",
        viewDetail: "View Details",
        deleteItem: "Delete",
        confirmClearAll: "Are you sure you want to delete all recognition history?",
        confirmDelete: "Are you sure you want to delete this item?",
        historyLoaded: "Loaded results from history!",
        introLabel: "Introduction:",
        originLabel: "Origin & Culture:",
        ingredientsLabel: "Main Ingredients:",
        tasteLabel: "Flavor Profile:",
        prepLabel: "Preparation & Serving:",
        noteLabel: "Culinary Notes:",
        updating: "Updating...",
        noInfo: "No information available.",
        selectFileWarn: "Please select an image first!",
        serverErrNetwork: "Cannot connect to server (server may be waking up or updating). Please wait 15-30 seconds and retry!",
        serverErr502: "Server is waking up from sleep or temporarily overloaded (502 Bad Gateway). Please wait 15-30 seconds and retry!",
        serverErr504: "Processing took too long (504 Gateway Timeout). Please retry with a smaller image!",
        serverErr500: "Server encountered an internal error (500). Please retry!"
      }
    };

    let currentLang = localStorage.getItem('food_app_lang') || 'vi';

    // BẢNG CHUẨN HÓA TIẾNG VIỆT CÓ DẤU
    const VIETNAMESE_NAMES = {
      'Banh canh': 'Bánh canh', 'Banh chung': 'Bánh chưng', 'Banh cuon': 'Bánh cuốn',
      'Banh khot': 'Bánh khọt', 'Banh mi': 'Bánh mì', 'Banh trang': 'Bánh tráng',
      'Banh trang tron': 'Bánh tráng trộn', 'Banh xeo': 'Bánh xèo', 'Bo kho': 'Bò kho',
      'Bo la lot': 'Bò lá lốt', 'Bong cai': 'Bông cải', 'Bun': 'Bún',
      'Bun bo Hue': 'Bún bò Huế', 'Bun cha': 'Bún chả', 'Bun dau': 'Bún đậu',
      'Bun mam': 'Bún mắm', 'Bun rieu': 'Bún riêu', 'Ca': 'Cá',
      'Ca chua': 'Cà chua', 'Ca phao': 'Cà pháo', 'Ca rot': 'Cà rốt',
      'Canh': 'Canh', 'Cha': 'Chả', 'Cha gio': 'Chả giò', 'Chanh': 'Chanh',
      'Com': 'Cơm', 'Com tam': 'Cơm tấm', 'Con nguoi': 'Người',
      'Cu kieu': 'Củ kiệu', 'Cua': 'Cua', 'Dau hu': 'Đậu hũ', 'Dua chua': 'Dưa chua',
      'Dua leo': 'Dưa leo', 'Goi cuon': 'Gỏi cuốn', 'Hamburger': 'Hamburger',
      'Heo quay': 'Heo quay', 'Hu tieu': 'Hủ tiếu', 'Kho qua thit': 'Khổ qua nhồi thịt',
      'Khoai tay chien': 'Khoai tây chiên', 'Lau': 'Lẩu', 'Long heo': 'Lòng heo',
      'Mi': 'Mì', 'Muc': 'Mực', 'Nam': 'Nấm', 'Oc': 'Ốc',
      'Ot chuong': 'Ớt chuông', 'Pho': 'Phở', 'Pho mai': 'Phô mai', 'Rau': 'Rau',
      'Salad': 'Salad', 'Thit bo': 'Thịt bò', 'Thit ga': 'Thịt gà', 'Thit heo': 'Thịt heo',
      'Thit kho': 'Thịt kho', 'Thit nuong': 'Thịt nướng', 'Tom': 'Tôm',
      'Trung': 'Trứng', 'Xoi': 'Xôi', 'Banh beo': 'Bánh bèo', 'Cao lau': 'Cao lầu',
      'Mi Quang': 'Mì Quảng', 'Com chien duong chau': 'Cơm chiên Dương Châu',
      'Bun cha ca': 'Bún chả cá', 'Com chien ga': 'Cơm chiên gà', 'Chao long': 'Cháo lòng',
      'Nom hoa chuoi': 'Nộm hoa chuối', 'Nui xao bo': 'Nui xào bò', 'Sup cua': 'Súp cua'
    };

    // BẢNG DỊCH TIẾNG ANH KÈM TÊN GỐC
    const FOOD_NAMES_EN = {
      'Bánh canh': 'Thick Noodle Soup (Bánh canh)',
      'Bánh chưng': 'Square Sticky Rice Cake (Bánh chưng)',
      'Bánh cuốn': 'Steamed Rice Rolls (Bánh cuốn)',
      'Bánh khọt': 'Mini Crispy Pancakes (Bánh khọt)',
      'Bánh mì': 'Vietnamese Baguette (Bánh mì)',
      'Bánh tráng': 'Rice Paper (Bánh tráng)',
      'Bánh tráng trộn': 'Rice Paper Salad (Bánh tráng trộn)',
      'Bánh xèo': 'Crispy Vietnamese Pancake (Bánh xèo)',
      'Bò kho': 'Beef Stew (Bò kho)',
      'Bò lá lốt': 'Beef in Betel Leaves (Bò lá lốt)',
      'Bông cải': 'Broccoli',
      'Bún': 'Rice Vermicelli',
      'Bún bò Huế': 'Hue Spicy Beef Noodles (Bún bò Huế)',
      'Bún chả': 'Grilled Pork Noodles (Bún chả)',
      'Bún đậu': 'Tofu & Noodle Platter (Bún đậu)',
      'Bún mắm': 'Fermented Fish Noodles (Bún mắm)',
      'Bún riêu': 'Crab Noodle Soup (Bún riêu)',
      'Cá': 'Fish',
      'Cà chua': 'Tomato',
      'Cà pháo': 'Pickled Eggplant',
      'Cà rốt': 'Carrot',
      'Canh': 'Soup',
      'Chả': 'Pork Roll (Chả)',
      'Chả giò': 'Crispy Spring Rolls (Chả giò)',
      'Chanh': 'Lime',
      'Cơm': 'Steamed Rice',
      'Cơm tấm': 'Broken Rice (Cơm tấm)',
      'Người': 'Person',
      'Củ kiệu': 'Pickled Allium',
      'Cua': 'Crab',
      'Đậu hũ': 'Tofu',
      'Dưa chua': 'Pickled Greens',
      'Dưa leo': 'Cucumber',
      'Gỏi cuốn': 'Fresh Spring Rolls (Gỏi cuốn)',
      'Hamburger': 'Hamburger',
      'Heo quay': 'Roast Pork',
      'Hủ tiếu': 'Kuy Teav Noodles (Hủ tiếu)',
      'Khổ qua nhồi thịt': 'Stuffed Bitter Melon Soup',
      'Khoai tây chiên': 'French Fries',
      'Lẩu': 'Hotpot',
      'Lòng heo': 'Pork Offal',
      'Mì': 'Egg Noodles',
      'Mực': 'Squid',
      'Nấm': 'Mushroom',
      'Ốc': 'Sea Snails (Ốc)',
      'Ớt chuông': 'Bell Pepper',
      'Phở': 'Pho Noodle Soup (Phở)',
      'Phô mai': 'Cheese',
      'Rau': 'Herbs & Greens',
      'Salad': 'Salad',
      'Thịt bò': 'Beef',
      'Thịt gà': 'Chicken',
      'Thịt heo': 'Pork',
      'Thịt kho': 'Braised Pork (Thịt kho)',
      'Thịt nướng': 'Grilled Meat',
      'Tôm': 'Shrimp',
      'Trứng': 'Egg',
      'Xôi': 'Sticky Rice (Xôi)',
      'Bánh bèo': 'Water Fern Cake (Bánh bèo)',
      'Cao lầu': 'Cao Lau Noodles',
      'Mì Quảng': 'Quang Style Noodles (Mì Quảng)',
      'Cơm chiên Dương Châu': 'Yangzhou Fried Rice',
      'Bún chả cá': 'Fish Cake Noodles',
      'Cơm chiên gà': 'Chicken Fried Rice',
      'Cháo lòng': 'Pork Congee',
      'Nộm hoa chuối': 'Banana Blossom Salad',
      'Nui xào bò': 'Stir-fried Macaroni with Beef',
      'Súp cua': 'Crab Asparagus Soup'
    };

    function formatFoodName(name) {
      if (!name) return '';
      const clean = name.trim();
      return VIETNAMESE_NAMES[clean] || clean;
    }

    function getDisplayName(rawOrViName) {
      const vi = formatFoodName(rawOrViName);
      if (currentLang === 'en' && FOOD_NAMES_EN[vi]) {
        return FOOD_NAMES_EN[vi];
      }
      return vi;
    }

    // ĐỔI NGÔN NGỮ
    function setLanguage(lang) {
      currentLang = lang;
      localStorage.setItem('food_app_lang', lang);

      const viBtn = document.getElementById('btn-lang-vi');
      const enBtn = document.getElementById('btn-lang-en');
      if (viBtn && enBtn) {
        if (lang === 'vi') {
          viBtn.classList.add('active');
          enBtn.classList.remove('active');
        } else {
          enBtn.classList.add('active');
          viBtn.classList.remove('active');
        }
      }

      // Cập nhật tất cả text data-i18n
      document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (TRANSLATIONS[lang] && TRANSLATIONS[lang][key]) {
          el.innerText = TRANSLATIONS[lang][key];
        }
      });

      // Nếu đang có kết quả hiển thị, re-render để cập nhật nhãn và nội dung
      if (window.currentResultData) {
        renderResults(window.currentResultData, false);
      }

      renderHistoryUI();
      if (window.lucide) lucide.createIcons();
    }

    function t(key, replacements = {}) {
      let str = (TRANSLATIONS[currentLang] && TRANSLATIONS[currentLang][key]) || key;
      for (const [k, v] of Object.entries(replacements)) {
        str = str.replace(`{${k}}`, v);
      }
      return str;
    }

    // TOAST THÔNG BÁO
    function showToast(msg) {
      let toast = document.getElementById('toast-notification');
      if (!toast) {
        toast = document.createElement('div');
        toast.id = 'toast-notification';
        toast.className = 'toast-box';
        document.body.appendChild(toast);
      }
      toast.innerHTML = `<i data-lucide="check-circle-2"></i> ${msg}`;
      toast.classList.add('show');
      if (window.lucide) lucide.createIcons();
      setTimeout(() => { toast.classList.remove('show'); }, 2500);
    }

    let selectedFile = null;
    let loadedImage = null;
    window.currentResultData = null;

    // QUẢN LÝ LỊCH SỬ & LOCAL STORAGE
    function formatHistoryTime(d) {
      const day = ('0' + d.getDate()).slice(-2);
      const mon = ('0' + (d.getMonth() + 1)).slice(-2);
      const yr = d.getFullYear();
      const hr = ('0' + d.getHours()).slice(-2);
      const min = ('0' + d.getMinutes()).slice(-2);
      return `${day}/${mon}/${yr} - ${hr}:${min}`;
    }

    function createCompressedImage(img, maxDim, quality) {
      try {
        const canvas = document.createElement('canvas');
        let w = img.naturalWidth || img.width;
        let h = img.naturalHeight || img.height;
        if (!w || !h) return '';
        if (w > h) {
          if (w > maxDim) {
            h = Math.round((h * maxDim) / w);
            w = maxDim;
          }
        } else {
          if (h > maxDim) {
            w = Math.round((w * maxDim) / h);
            h = maxDim;
          }
        }
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, w, h);
        return canvas.toDataURL('image/jpeg', quality);
      } catch (err) {
        console.warn("Resize image error:", err);
        return '';
      }
    }

    function getHistoryList() {
      try {
        const raw = localStorage.getItem('food_recognition_history');
        return raw ? JSON.parse(raw) : [];
      } catch (_) {
        return [];
      }
    }

    function safeSaveHistory(list) {
      try {
        localStorage.setItem('food_recognition_history', JSON.stringify(list));
      } catch (e) {
        console.warn("Storage quota warning, trimming history...", e);
        while (list.length > 5) {
          list.splice(list.length - 4, 4);
          try {
            localStorage.setItem('food_recognition_history', JSON.stringify(list));
            break;
          } catch (_) {}
        }
      }
    }

    function saveToHistory(item) {
      let list = getHistoryList();
      list.unshift(item);
      if (list.length > 25) list = list.slice(0, 25);
      safeSaveHistory(list);
      renderHistoryUI();
    }

    function deleteHistoryItem(id) {
      if (!confirm(t('confirmDelete'))) return;
      let list = getHistoryList().filter(h => h.id !== id);
      safeSaveHistory(list);
      renderHistoryUI();
    }

    function clearAllHistory() {
      if (!confirm(t('confirmClearAll'))) return;
      localStorage.removeItem('food_recognition_history');
      renderHistoryUI();
    }

    function loadHistoryItem(id) {
      const list = getHistoryList();
      const item = list.find(h => h.id === id);
      if (!item) return;

      const img = new Image();
      img.onload = () => {
        loadedImage = img;
        inputPreviewImg.src = img.src;
        dropZone.style.display = 'none';
        previewBox.style.display = 'flex';
        fileNameText.innerText = `${getDisplayName(item.primaryFood)} (${item.timestamp})`;

        if (item.confThreshold) {
          const slider = document.getElementById('conf-slider');
          if (slider) {
            slider.value = item.confThreshold;
            document.getElementById('conf-val').innerText = Math.round(item.confThreshold * 100) + '%';
          }
        }

        renderResults(item.data, false);
        showToast(t('historyLoaded'));

        const mainGrid = document.getElementById('main-grid');
        if (mainGrid) {
          mainGrid.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      };
      img.src = item.previewImage || item.thumbnail;
    }

    function renderHistoryUI() {
      const list = getHistoryList();
      const countBadge = document.getElementById('history-badge');
      const totalCount = document.getElementById('stats-total-count');
      const recentContainer = document.getElementById('stats-recent-list');
      const historyContainer = document.getElementById('history-items-container');

      if (countBadge) countBadge.innerText = list.length;
      if (totalCount) totalCount.innerText = list.length;

      // Cập nhật thống kê món gần đây (không dùng emoji)
      if (recentContainer) {
        if (list.length === 0) {
          recentContainer.innerHTML = `<span class="stat-empty">${t('statsNone')}</span>`;
        } else {
          const seen = new Set();
          const recentFoods = [];
          for (const item of list) {
            const foods = item.allFoods || [item.primaryFood];
            for (const f of foods) {
              if (f && !seen.has(f)) {
                seen.add(f);
                recentFoods.push(f);
                if (recentFoods.length >= 8) break;
              }
            }
            if (recentFoods.length >= 8) break;
          }

          recentContainer.innerHTML = recentFoods.map(f => {
            const dName = getDisplayName(f);
            return `<span class="recent-food-badge">${dName}</span>`;
          }).join('');
        }
      }

      // Cập nhật danh sách lịch sử (không dùng emoji)
      if (historyContainer) {
        if (list.length === 0) {
          historyContainer.innerHTML = `
            <div class="history-empty">
              <i data-lucide="clock" class="history-empty-icon"></i>
              <p>${t('historyEmpty')}</p>
            </div>
          `;
        } else {
          let html = '';
          list.forEach(item => {
            const foodName = getDisplayName(item.primaryFood);
            const viewText = t('viewDetail');
            const delText = t('deleteItem');

            html += `
              <div class="history-item">
                <div class="history-main">
                  <img src="${item.thumbnail}" class="history-thumb" alt="${foodName}">
                  <div class="history-details">
                    <div class="history-food-title">
                      <span>${foodName}</span>
                    </div>
                    <div class="history-meta-row">
                      <span class="history-conf-tag">${item.primaryConf}</span>
                      <span class="history-date"><i data-lucide="calendar"></i> ${item.timestamp}</span>
                    </div>
                  </div>
                </div>
                <div class="history-actions">
                  <button type="button" class="btn-history-view" onclick="loadHistoryItem('${item.id}')">
                    <i data-lucide="eye"></i> ${viewText}
                  </button>
                  <button type="button" class="btn-history-del" onclick="deleteHistoryItem('${item.id}')" title="${delText}">
                    <i data-lucide="trash-2"></i> ${delText}
                  </button>
                </div>
              </div>
            `;
          });
          historyContainer.innerHTML = html;
        }
      }

      if (window.lucide) lucide.createIcons();
    }

    // KHỞI TẠO
    window.addEventListener('DOMContentLoaded', () => {
      setLanguage(currentLang);
      renderHistoryUI();
      if (window.lucide) lucide.createIcons();
    });

    const fileInput = document.getElementById('file-input');
    const dropZone = document.getElementById('drop-zone');
    const previewBox = document.getElementById('preview-box');
    const inputPreviewImg = document.getElementById('input-preview-img');
    const fileNameText = document.getElementById('file-name-text');

    ['dragenter', 'dragover'].forEach(e => {
      dropZone.addEventListener(e, (ev) => { ev.preventDefault(); dropZone.classList.add('dragover'); });
    });
    ['dragleave', 'drop'].forEach(e => {
      dropZone.addEventListener(e, (ev) => { ev.preventDefault(); dropZone.classList.remove('dragover'); });
    });

    dropZone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length) handleFile(e.target.files[0]);
    });

    // XỬ LÝ KHI CHỌN ẢNH: HIỂN THỊ Ở CỘT TRÁI
    function handleFile(file) {
      selectedFile = file;
      fileNameText.innerText = `${file.name} (${Math.round(file.size / 1024)} KB)`;

      const reader = new FileReader();
      reader.onload = (e) => {
        loadedImage = new Image();
        loadedImage.onload = () => {
          inputPreviewImg.src = e.target.result;
          dropZone.style.display = 'none';
          previewBox.style.display = 'flex';

          document.getElementById('result-content').style.display = 'none';
          document.getElementById('empty-state').style.display = 'block';
          document.getElementById('empty-state-text').innerText = t('emptyReady');
          if (window.lucide) lucide.createIcons();
        };
        loadedImage.src = e.target.result;
      };
      reader.readAsDataURL(file);
    }

    // BẮT ĐẦU PHÂN TÍCH (TIẾN TRÌNH 2 PHA TIẾT KIỆM THỜI GIAN & TĂNG TỐC TRẢI NGHIỆM)
    async function processImage() {
      if (!selectedFile) {
        alert(t('selectFileWarn'));
        return;
      }

      const mode = document.querySelector('input[name="api-mode"]:checked').value;
      const confSlider = document.getElementById('conf-slider');
      const conf = confSlider ? parseFloat(confSlider.value) : 0.25;

      const btn = document.getElementById('submit-btn');
      const btnText = document.getElementById('btn-text');
      const spinner = document.getElementById('btn-spinner');

      btn.disabled = true;
      btnText.innerText = (currentLang === 'vi' ? 'Đang nhận diện món...' : 'Detecting dishes...');
      spinner.style.display = "inline-block";

      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('confidence', conf);

      try {
        // PHA 1: NẾU CHẾ ĐỘ PHÂN TÍCH, GỌI /predict ĐỂ VẼ BOUNDING BOX NGAY TRONG 1S
        if (mode === 'analyze') {
          let shouldContinueToAnalyze = true;
          try {
            const predResp = await fetch('/predict', { method: 'POST', body: formData });
            if (predResp.ok) {
              const predData = await predResp.json();
              renderResults(predData, false);

              const validDets = (predData.detections || []).filter(d => d.confidence >= conf);
              if (validDets.length > 0) {
                const infoContainer = document.getElementById('culinary-info-container');
                if (infoContainer) {
                  const loadMsg = currentLang === 'vi' ? 'Đang trích xuất tri thức ẩm thực chuyên sâu...' : 'Extracting culinary knowledge...';
                  infoContainer.innerHTML = `
                    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:14px; padding:24px; text-align:center; margin-top:14px;">
                      <div class="spinner" style="border-top-color:var(--primary); width:26px; height:26px; margin:0 auto 10px;"></div>
                      <p style="font-size:0.92rem; font-weight:600; color:#334155;">${loadMsg}</p>
                    </div>
                  `;
                }
                btnText.innerText = (currentLang === 'vi' ? 'Đang trích xuất tri thức...' : 'Extracting knowledge...');
              } else {
                // KHÔNG TÌM THẤY MÓN ĂN NÀO ĐẠT CONFIDENCE YÊU CẦU -> DỪNG NGAY, KHÔNG GỌI LLM TRI THỨC!
                shouldContinueToAnalyze = false;
              }
            }
          } catch (predErr) {
            console.warn("Fast predict warning:", predErr);
          }

          if (!shouldContinueToAnalyze) {
            return;
          }
        }

        // PHA 2: GỌI ANALYZE ĐỂ LẤY TOÀN BỘ TRI THỨC VÀ CÔNG THỨC (CHỈ CHẠY KHI CÓ MÓN ĂN THỎA ĐIỀU KIỆN)
        const endpoint = mode === 'analyze' ? '/analyze' : '/predict';
        const response = await fetch(endpoint, {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          let errorMsg = `Server error (${response.status})`;
          try {
            const errData = await response.json();
            if (errData && errData.detail) errorMsg = errData.detail;
          } catch (_) {
            if (response.status === 502) errorMsg = t('serverErr502');
            else if (response.status === 504) errorMsg = t('serverErr504');
            else if (response.status === 500) errorMsg = t('serverErr500');
          }
          throw new Error(errorMsg);
        }

        const data = await response.json();
        window.currentResultData = data;
        renderResults(data, true); // true = lưu vào lịch sử
      } catch (err) {
        let errMsg = err.message || "";
        if (errMsg.includes("Failed to fetch") || errMsg.includes("NetworkError")) {
          alert(t('serverErrNetwork'));
        } else {
          alert("Error: " + errMsg);
        }
      } finally {
        btn.disabled = false;
        btnText.innerText = t('btnSubmit');
        spinner.style.display = "none";
        if (window.lucide) lucide.createIcons();
      }
    }

    // HIỂN THỊ KẾT QUẢ SANG CỘT PHẢI
    function renderResults(data, shouldSaveHistory = false) {
      if (!data || !loadedImage) return;
      window.currentResultData = data;

      document.getElementById('empty-state').style.display = 'none';
      const resultContent = document.getElementById('result-content');
      resultContent.style.display = 'block';

      const canvas = document.getElementById('result-canvas');
      const canvasW = loadedImage.naturalWidth || loadedImage.width;
      const canvasH = loadedImage.naturalHeight || loadedImage.height;
      canvas.width = canvasW;
      canvas.height = canvasH;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(loadedImage, 0, 0, canvasW, canvasH);

      // Tự động scale tọa độ nếu canvas khác kích thước gốc (hỗ trợ chính xác cả khi xem lại từ history thumbnail)
      const baseW = data.image_width || canvasW;
      const baseH = data.image_height || canvasH;
      const scaleX = (baseW > 0) ? (canvasW / baseW) : 1.0;
      const scaleY = (baseH > 0) ? (canvasH / baseH) : 1.0;

      const currentConf = parseFloat(document.getElementById('conf-slider').value) || 0.25;
      const rawDetections = data.detections || [];
      const detections = rawDetections.filter(d => d.confidence >= currentConf);

      const tagsContainer = document.getElementById('tags-container');
      const infoContainer = document.getElementById('culinary-info-container');
      tagsContainer.innerHTML = '';
      infoContainer.innerHTML = '';

      if (detections.length === 0) {
        const percentText = Math.round(currentConf * 100);
        tagsContainer.innerHTML = `<span style="color:#64748b; font-style:italic;">${t('noDetection', { conf: percentText })}</span>`;
      } else {
        const colors = ['#e63946', '#2a9d8f', '#e76f51', '#457b9d', '#9b5de5', '#f4a261'];
        detections.forEach((d, idx) => {
          const color = colors[idx % colors.length];
          const b = d.box;
          const confPercent = Math.round(d.confidence * 100);
          const displayName = getDisplayName(d.class_name);

          const x1 = b.x1 * scaleX;
          const y1 = b.y1 * scaleY;
          const boxW = (b.x2 - b.x1) * scaleX;
          const boxH = (b.y2 - b.y1) * scaleY;

          // Vẽ Bounding Box chính xác ôm trọn món ăn
          ctx.lineWidth = Math.max(3, Math.round(canvas.width / 240));
          ctx.strokeStyle = color;
          ctx.strokeRect(x1, y1, boxW, boxH);

          // Nhãn trên Canvas (theo ngôn ngữ đã chọn)
          const label = `${displayName} (${confPercent}%)`;
          const fontSize = Math.max(14, Math.round(canvas.width / 36));
          ctx.font = `bold ${fontSize}px 'Plus Jakarta Sans', sans-serif`;
          const textWidth = ctx.measureText(label).width;
          const textHeight = fontSize + 4;

          ctx.fillStyle = color;
          ctx.fillRect(x1, Math.max(0, y1 - textHeight - 6), textWidth + 12, textHeight + 6);
          ctx.fillStyle = '#ffffff';
          ctx.fillText(label, x1 + 6, Math.max(textHeight, y1 - 4));

          // Detection Chip (không có icon emoji)
          const chip = document.createElement('div');
          chip.className = 'detection-chip';
          chip.style.backgroundColor = color + '20';
          chip.style.color = color;
          chip.innerText = `${displayName}: ${confPercent}%`;
          tagsContainer.appendChild(chip);
        });

        // LƯU VÀO LOCAL STORAGE NẾU LÀ LẦN CHẠY MỚI
        if (shouldSaveHistory) {
          const bestDet = detections.reduce((max, d) => d.confidence > max.confidence ? d : max, detections[0]);
          const primaryFoodVi = formatFoodName(bestDet.class_name);
          const primaryConfStr = (bestDet.confidence * 100).toFixed(1) + '%';
          const allFoodsList = [...new Set(detections.map(d => formatFoodName(d.class_name)))];

          const historyItem = {
            id: 'hist_' + Date.now(),
            primaryFood: primaryFoodVi,
            primaryConf: primaryConfStr,
            allFoods: allFoodsList,
            timestamp: formatHistoryTime(new Date()),
            thumbnail: createCompressedImage(loadedImage, 85, 0.55),
            previewImage: createCompressedImage(loadedImage, 520, 0.65),
            confThreshold: currentConf,
            data: data
          };
          saveToHistory(historyItem);
        }
      }

      // THẺ TRI THỨC ẨM THỰC (CHUYỂN TOÀN BỘ NỘI DUNG SANG TIẾNG ANH KHI CHỌN ENGLISH)
      const rawFoodsList = (data.foods_info && data.foods_info.length > 0)
        ? data.foods_info
        : (data.food_info ? [data.food_info] : []);

      let foodsList = [];
      if (detections.length > 0 && rawFoodsList.length > 0) {
        const detectedNames = detections.map(d => formatFoodName(d.class_name).toLowerCase().trim());
        foodsList = rawFoodsList.filter(info => {
          const infoName = formatFoodName(info.food_name).toLowerCase().trim();
          // Kiểm tra khớp trực tiếp hoặc bao hàm 2 chiều (ví dụ: "trứng ốp la" <-> "trứng", "thịt nướng" <-> "thịt heo")
          return detectedNames.some(dName => 
            infoName === dName || 
            infoName.includes(dName) || 
            dName.includes(infoName)
          );
        });

        // Nếu vì lý do đặt tên của LLM mà filter không bắt được nhưng server đã phân tích trả về
        if (foodsList.length === 0) {
          foodsList = rawFoodsList;
        }
      }

      if (foodsList.length > 0) {
        let cardsHtml = '';
        foodsList.forEach((info, idx) => {
          const displayName = getDisplayName(info.food_name);
          const isEn = (currentLang === 'en');
          const enInfo = (info.en && typeof info.en === 'object') ? info.en : null;

          // Nội dung chi tiết: tiếng Anh nếu chọn English và có trường en, ngược lại tiếng Việt
          const descText = (isEn && enInfo && enInfo.description) ? enInfo.description : (info.description || t('noInfo'));
          const originText = (isEn && enInfo && enInfo.origin) ? enInfo.origin : (info.origin || t('noInfo'));
          const rawIngredients = (isEn && enInfo && Array.isArray(enInfo.ingredients) && enInfo.ingredients.length > 0)
            ? enInfo.ingredients
            : (info.ingredients || []);
          const tasteText = (isEn && enInfo && enInfo.taste) ? enInfo.taste : (info.taste || t('updating'));
          const prepText = (isEn && enInfo && enInfo.preparation) ? enInfo.preparation : (info.preparation || t('updating'));
          const noteText = (isEn && enInfo && enInfo.note) ? enInfo.note : (info.note || '');

          const ingBadges = rawIngredients.map(i => `<span class="ingredient-badge">${i}</span>`).join('');

          cardsHtml += `
            <div class="info-card" style="margin-top: ${idx === 0 ? '14px' : '20px'};">
              <div class="info-card-header">
                <h3>${displayName}</h3>
              </div>
              <div class="info-item">
                <strong><i data-lucide="book-open"></i> ${t('introLabel')}</strong>
                <p>${descText}</p>
              </div>
              <div class="info-item">
                <strong><i data-lucide="map-pin"></i> ${t('originLabel')}</strong>
                <p>${originText}</p>
              </div>
              <div class="info-item">
                <strong><i data-lucide="layers"></i> ${t('ingredientsLabel')}</strong>
                <div class="ingredients-tags">${ingBadges || t('updating')}</div>
              </div>
              <div class="info-item">
                <strong><i data-lucide="sparkles"></i> ${t('tasteLabel')}</strong>
                <p>${tasteText}</p>
              </div>
              <div class="info-item">
                <strong><i data-lucide="chef-hat"></i> ${t('prepLabel')}</strong>
                <p>${prepText}</p>
              </div>
              ${noteText ? `
              <div class="info-item">
                <strong><i data-lucide="info"></i> ${t('noteLabel')}</strong>
                <p>${noteText}</p>
              </div>` : ''}
            </div>
          `;
        });
        infoContainer.innerHTML = cardsHtml;
      } else if (data.message && detections.length > 0) {
        infoContainer.innerHTML = `<div style="font-size:0.88rem; color:#64748b; margin-top:8px;">${data.message}</div>`;
      }

      if (window.lucide) lucide.createIcons();
    }
  </script>
</body>
</html>
"""
# ============================================================
# ROOT ROUTE (HTML UI + JSON API COMPATIBILITY)
# ============================================================

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def root(request: Request):
    """
    Trả về Web UI trực quan nếu truy cập từ trình duyệt,
    hoặc JSON nếu gọi từ API client, hỗ trợ HEAD cho Render Health Check.
    """
    if request.method == "HEAD":
        return Response(status_code=200)

    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return JSONResponse(
            content={
                "message": "Vietnamese Food Recognition API",
                "version": "1.0.0",
                "docs": "/docs",
                "health": "/health",
            }
        )
    return HTMLResponse(content=HTML_PAGE)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.api_route("/health", methods=["GET", "HEAD"])
def health(request: Request):
    if request.method == "HEAD":
        return Response(status_code=200)

    active_det = get_detector()
    active_llm = get_llm()
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "model_loaded": active_det is not None,
        "model_error": detector_error,
        "ai_knowledge_ready": active_llm is not None,
        "ai_error": llm_error,
    }


# ============================================================
# PREDICT (CHỈ CHẠY YOLO - NHẬN ĐÚNG NGƯỠNG CONFIDENCE)
# ============================================================

@app.post(
    "/predict",
    response_model=PredictResponse,
)
async def predict(
    file: UploadFile = File(...),
    confidence: Optional[float] = Form(None),
):
    """
    Nhận ảnh và chạy YOLO object detection theo ngưỡng confidence yêu cầu.
    """
    active_det = get_detector()
    if active_det is None:
        raise HTTPException(
            status_code=500,
            detail=f"Mô hình YOLO chưa sẵn sàng: {detector_error or 'Không xác định'}",
        )

    image, orig_size = await read_image(file)

    try:
        detections = active_det.predict(
            image,
            confidence_threshold=confidence,
            original_size=orig_size,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi nhận diện hình ảnh: {str(exc)}",
        ) from exc

    if not detections:
        return PredictResponse(
            success=True,
            detections=[],
            image_width=orig_size[0],
            image_height=orig_size[1],
            message=(
                "Không phát hiện được món ăn "
                "với ngưỡng confidence hiện tại."
            ),
        )

    return PredictResponse(
        success=True,
        detections=detections,
        image_width=orig_size[0],
        image_height=orig_size[1],
        message="Nhận diện thành công.",
    )


# ============================================================
# FOOD INFO (TRA CỨU TRI THỨC MÓN ĂN THEO TÊN)
# ============================================================

@app.post(
    "/food-info",
    response_model=FoodInfoResponse,
)
def food_info(
    request: FoodInfoRequest,
):
    """
    Nhận tên món ăn và tra cứu thông tin ẩm thực chi tiết.
    """
    active_llm = get_llm()
    if active_llm is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Hệ thống tri thức ẩm thực chưa được kích hoạt. "
                "Hãy thiết lập API KEY trong biến môi trường."
            ),
        )

    # Chuẩn hóa tên món ăn sang tiếng Việt có dấu trước khi tra cứu
    food_name_vn = format_food_name(request.food_name)

    try:
        res = active_llm.get_food_info(food_name_vn)
        res.food_name = format_food_name(res.food_name)
        return res
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi tra cứu thông tin ẩm thực: {str(exc)}",
        ) from exc


# ============================================================
# ANALYZE (PIPELINE TOÀN DIỆN - NHẬN ĐÚNG NGƯỠNG CONFIDENCE)
# ============================================================

@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
)
async def analyze(
    file: UploadFile = File(...),
    confidence: Optional[float] = Form(None),
):
    """
    Pipeline phân tích hoàn chỉnh:
    Ảnh -> YOLO nhận diện theo confidence -> Lọc các món duy nhất -> LLM phân tích chi tiết công thức cho TẤT CẢ các món phát hiện được.
    """
    active_det = get_detector()
    if active_det is None:
        raise HTTPException(
            status_code=500,
            detail=f"Mô hình YOLO chưa sẵn sàng: {detector_error or 'Không xác định'}",
        )

    image, orig_size = await read_image(file)

    # 1. YOLO Detect theo đúng ngưỡng confidence
    try:
        detections = active_det.predict(
            image,
            confidence_threshold=confidence,
            original_size=orig_size,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi nhận diện hình ảnh: {str(exc)}",
        ) from exc

    if not detections:
        return AnalyzeResponse(
            success=True,
            detections=[],
            food_info=None,
            foods_info=[],
            image_width=orig_size[0],
            image_height=orig_size[1],
            message="Không phát hiện được món ăn trong ảnh.",
        )

    # 2. Lấy danh sách tất cả các món ăn duy nhất phát hiện được (sắp xếp theo độ tin cậy)
    sorted_detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    unique_food_names: List[str] = []
    seen = set()
    for d in sorted_detections:
        formatted_name = format_food_name(d.class_name)
        if formatted_name.lower() not in seen:
            seen.add(formatted_name.lower())
            unique_food_names.append(formatted_name)

    # 3. Tra cứu tri thức ẩm thực cho TẤT CẢ các món phát hiện được
    active_llm = get_llm()
    if active_llm is None:
        return AnalyzeResponse(
            success=True,
            detections=detections,
            food_info=None,
            foods_info=[],
            image_width=orig_size[0],
            image_height=orig_size[1],
            message=(
                f"Đã nhận diện {len(unique_food_names)} món: {', '.join(unique_food_names)}. "
                "Hệ thống tri thức chưa được cấu hình API Key."
            ),
        )

    foods_info: List[FoodInfoResponse] = []
    try:
        foods_info = active_llm.get_foods_info(unique_food_names)
        for item in foods_info:
            item.food_name = format_food_name(item.food_name)
    except Exception as exc:
        print(f"[Analyze] Loi khi lay thong tin danh sach mon: {exc}")

    primary_food_info = foods_info[0] if foods_info else None

    return AnalyzeResponse(
        success=True,
        detections=detections,
        food_info=primary_food_info,
        foods_info=foods_info,
        image_width=orig_size[0],
        image_height=orig_size[1],
        message=f"Đã phân tích hoàn tất {len(foods_info)} món ăn." if foods_info else "Phân tích hoàn tất.",
    )