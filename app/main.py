import io
import os
import sys
from pathlib import Path
from typing import Optional

# Đảm bảo Python luôn tìm thấy các package trong app dù chạy từ đâu
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, ImageOps

try:
    from app.detector import FoodDetector
    from app.llm import FoodLLM
    from app.schemas import (
        AnalyzeResponse,
        FoodInfoRequest,
        FoodInfoResponse,
        PredictResponse,
    )
except ImportError:
    from detector import FoodDetector
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
# HELPER FUNCTIONS
# ============================================================

async def read_image(file: UploadFile) -> Image.Image:
    """
    Đọc file upload và chuyển thành PIL Image chuẩn RGB,
    tự động căn chỉnh góc xoay theo EXIF từ máy ảnh điện thoại.
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

    try:
        raw_image = Image.open(io.BytesIO(content))
        image = ImageOps.exif_transpose(raw_image)
        if image is None:
            image = raw_image
        return image.convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="File không phải ảnh hợp lệ.",
        ) from exc


# ============================================================
# WEB UI TEMPLATE (TỐI ƯU GIAO DIỆN HIỆN ĐẠI, GỌN GÀNG, BẢO MẬT NGUỒN AI)
# ============================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Nhận Diện & Phân Tích Món Ăn Việt Nam | AI Food Recognition</title>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --primary: #e63946;
      --primary-hover: #d62828;
      --primary-light: #fef2f2;
      --accent: #2563eb;
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --text: #0f172a;
      --text-muted: #64748b;
      --border: #e2e8f0;
      --radius: 16px;
      --shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.06);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      padding: 24px 16px;
    }

    .container {
      max-width: 1120px;
      margin: 0 auto;
    }

    /* HEADER */
    header {
      text-align: center;
      margin-bottom: 28px;
    }

    header h1 {
      font-size: 2.1rem;
      font-weight: 800;
      color: var(--text);
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      letter-spacing: -0.02em;
    }

    header p {
      color: var(--text-muted);
      margin-top: 6px;
      font-size: 1rem;
    }

    .badges {
      display: flex;
      justify-content: center;
      gap: 10px;
      margin-top: 14px;
      flex-wrap: wrap;
    }

    .badge {
      font-size: 0.82rem;
      padding: 5px 14px;
      border-radius: 999px;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      text-decoration: none;
    }
    .badge-model { background: #e0f2fe; color: #0284c7; }
    .badge-ai { background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }
    .badge-docs { background: #f1f5f9; color: #475569; border: 1px solid var(--border); transition: all 0.2s; }
    .badge-docs:hover { background: #e2e8f0; color: #0f172a; }

    /* GRID */
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
      padding: 36px 20px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
      background: #f8fafc;
    }
    .upload-zone:hover, .upload-zone.dragover {
      border-color: var(--primary);
      background: var(--primary-light);
    }
    .upload-zone svg {
      width: 44px;
      height: 44px;
      color: #94a3b8;
      margin-bottom: 8px;
    }

    /* INPUT PREVIEW INSIDE LEFT BOX */
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
      padding: 5px 12px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
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

    /* RIGHT: RESULT CONTAINER */
    .result-canvas-wrap {
      width: 100%;
      min-height: 260px;
      background: #0f172a;
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    canvas {
      max-width: 100%;
      height: auto;
      display: block;
    }

    .empty-state {
      text-align: center;
      padding: 48px 16px;
      color: #94a3b8;
    }
    .empty-state svg {
      width: 48px;
      height: 48px;
      margin-bottom: 10px;
      opacity: 0.6;
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
      gap: 6px;
    }

    /* CULINARY KNOWLEDGE CARD */
    .info-card {
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px;
      margin-top: 14px;
    }

    .info-card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 14px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border);
    }

    .info-card-header h3 {
      font-size: 1.15rem;
      font-weight: 800;
      color: #0f172a;
    }

    .info-tag {
      font-size: 0.75rem;
      background: #dcfce7;
      color: #15803d;
      font-weight: 700;
      padding: 3px 10px;
      border-radius: 999px;
    }

    .info-item {
      margin-bottom: 12px;
      font-size: 0.92rem;
      line-height: 1.55;
    }
    .info-item strong {
      color: #334155;
      display: inline-block;
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
      <h1>🍜 Nhận Diện & Phân Tích Món Ăn Việt Nam</h1>
      <p>Hệ thống Thị giác máy tính nhận diện thị giác kết hợp Trí tuệ nhân tạo phân tích ẩm thực chuyên sâu</p>
      <div class="badges">
        <span class="badge badge-model" id="model-status-badge">🎯 YOLO Model: Đang tải...</span>
        <span class="badge badge-ai" id="ai-status-badge">✨ Hệ Thống Tri Thức: Sẵn sàng</span>
        <a href="/docs" target="_blank" class="badge badge-docs">📖 Swagger API Docs</a>
        <a href="/health" target="_blank" class="badge badge-docs">🩺 Kiểm tra Hệ Thống</a>
      </div>
    </header>

    <div class="main-grid">
      <!-- CỘT TRÁI: DỮ LIỆU ĐẦU VÀO (INPUT) -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">📷 1. Tải Ảnh Đầu Vào</div>
        </div>

        <!-- Khung chưa chọn ảnh -->
        <div class="upload-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>
          <p><strong>Bấm để chọn ảnh</strong> hoặc kéo thả file vào đây</p>
          <p style="font-size:0.82rem; color:#94a3b8; margin-top:4px;">Hỗ trợ: JPG, JPEG, PNG, WEBP</p>
        </div>

        <!-- Khung hiển thị ảnh đã chọn bên cột trái -->
        <div class="preview-box" id="preview-box" style="display:none;">
          <div class="preview-img-wrap">
            <img id="input-preview-img" src="" alt="Ảnh đầu vào">
          </div>
          <div class="preview-footer">
            <span class="file-name" id="file-name-text">anh_mon_an.jpg</span>
            <button type="button" class="btn-change" onclick="document.getElementById('file-input').click()">🔄 Chọn ảnh khác</button>
          </div>
        </div>

        <input type="file" id="file-input" accept="image/*" style="display:none">

        <div class="controls">
          <div>
            <div class="slider-group">
              <label for="conf-slider">Độ tin cậy tối thiểu (Confidence):</label>
              <span class="conf-badge" id="conf-val">25%</span>
            </div>
            <input type="range" id="conf-slider" min="0.05" max="0.95" step="0.05" value="0.25" oninput="document.getElementById('conf-val').innerText = Math.round(this.value * 100) + '%'">
          </div>

          <div class="mode-select">
            <label class="mode-option">
              <input type="radio" name="api-mode" value="analyze" checked>
              <span>Phân tích toàn diện (Nhận diện + Tri thức ẩm thực chi tiết)</span>
            </label>
            <label class="mode-option">
              <input type="radio" name="api-mode" value="predict">
              <span>Chỉ nhận diện vật thể (YOLO Detection)</span>
            </label>
          </div>

          <button class="btn-submit" id="submit-btn" onclick="processImage()">
            <span id="btn-text">🚀 Bắt Đầu Phân Tích</span>
            <div class="spinner" id="btn-spinner" style="display:none;"></div>
          </button>
        </div>
      </div>

      <!-- CỘT PHẢI: KẾT QUẢ ĐẦU RA (OUTPUT) -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">📊 2. Kết Quả Nhận Diện & Phân Tích</div>
        </div>

        <!-- Trạng thái trống lúc chưa phân tích -->
        <div class="empty-state" id="empty-state">
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
          <p id="empty-state-text">Vui lòng tải ảnh ở ô bên trái và bấm "Bắt Đầu Phân Tích"</p>
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
  </div>

  <script>
    let selectedFile = null;
    let loadedImage = null;

    // Kiểm tra trạng thái hệ thống lúc tải trang
    fetch('/health').then(r => r.json()).then(data => {
      const modelBadge = document.getElementById('model-status-badge');
      if (data.model_loaded) {
        modelBadge.innerText = '🎯 YOLO Model: Sẵn sàng';
        modelBadge.style.background = '#e0f2fe';
        modelBadge.style.color = '#0284c7';
      } else {
        modelBadge.innerText = '⚠️ YOLO Model: Chưa nạp';
        modelBadge.style.background = '#fee2e2';
        modelBadge.style.color = '#991b1b';
      }

      const aiBadge = document.getElementById('ai-status-badge');
      if (data.ai_knowledge_ready) {
        aiBadge.innerText = '✨ Hệ Thống Tri Thức: Sẵn sàng';
        aiBadge.style.background = '#f0fdf4';
        aiBadge.style.color = '#16a34a';
      } else {
        aiBadge.innerText = '⚠️ Hệ Thống Tri Thức: Cần cấu hình Key';
        aiBadge.style.background = '#fef9c3';
        aiBadge.style.color = '#854d0e';
      }
    }).catch(() => {});

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

    // XỬ LÝ KHI CHỌN ẢNH: HIỂN THỊ Ở CỘT TRÁI (KHÔNG HIỂN THỊ SANG CỘT PHẢI KHI CHƯA PHÂN TÍCH)
    function handleFile(file) {
      selectedFile = file;
      fileNameText.innerText = `${file.name} (${Math.round(file.size / 1024)} KB)`;

      const reader = new FileReader();
      reader.onload = (e) => {
        loadedImage = new Image();
        loadedImage.onload = () => {
          // 1. Hiển thị ảnh xem trước ngay tại cột trái
          inputPreviewImg.src = e.target.result;
          dropZone.style.display = 'none';
          previewBox.style.display = 'flex';

          // 2. Cột phải giữ nguyên trạng thái chờ phân tích
          document.getElementById('result-content').style.display = 'none';
          document.getElementById('empty-state').style.display = 'block';
          document.getElementById('empty-state-text').innerText = 'Ảnh đã sẵn sàng. Nhấn "Bắt Đầu Phân Tích" để nhận diện món ăn!';
        };
        loadedImage.src = e.target.result;
      };
      reader.readAsDataURL(file);
    }

    // BẮT ĐẦU PHÂN TÍCH
    async function processImage() {
      if (!selectedFile) {
        alert("Vui lòng chọn một bức ảnh trước!");
        return;
      }

      const mode = document.querySelector('input[name="api-mode"]:checked').value;
      const conf = document.getElementById('conf-slider').value;
      const endpoint = mode === 'analyze' ? '/analyze' : '/predict';

      const btn = document.getElementById('submit-btn');
      const btnText = document.getElementById('btn-text');
      const spinner = document.getElementById('btn-spinner');

      btn.disabled = true;
      btnText.innerText = "Đang xử lý...";
      spinner.style.display = "inline-block";

      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const response = await fetch(endpoint, {
          method: 'POST',
          body: formData
        });
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || 'Lỗi xử lý hệ thống');
        }

        renderResults(data);
      } catch (err) {
        alert("Lỗi: " + err.message);
      } finally {
        btn.disabled = false;
        btnText.innerText = "🚀 Bắt Đầu Phân Tích";
        spinner.style.display = "none";
      }
    }

    // HIỂN THỊ KẾT QUẢ SANG CỘT PHẢI
    function renderResults(data) {
      document.getElementById('empty-state').style.display = 'none';
      const resultContent = document.getElementById('result-content');
      resultContent.style.display = 'block';

      const canvas = document.getElementById('result-canvas');
      canvas.width = loadedImage.naturalWidth;
      canvas.height = loadedImage.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(loadedImage, 0, 0);

      const detections = data.detections || [];
      const tagsContainer = document.getElementById('tags-container');
      const infoContainer = document.getElementById('culinary-info-container');
      tagsContainer.innerHTML = '';
      infoContainer.innerHTML = '';

      if (detections.length === 0) {
        tagsContainer.innerHTML = '<span style="color:#64748b; font-style:italic;">Không tìm thấy món ăn nào với độ tin cậy này. Thử giảm Confidence!</span>';
      } else {
        const colors = ['#e63946', '#2a9d8f', '#e76f51', '#457b9d', '#9b5de5', '#f4a261'];
        detections.forEach((d, idx) => {
          const color = colors[idx % colors.length];
          const b = d.box;
          const confPercent = Math.round(d.confidence * 100);

          // Vẽ Bounding Box
          ctx.lineWidth = Math.max(3, Math.round(canvas.width / 240));
          ctx.strokeStyle = color;
          ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);

          // Nhãn tên món
          const label = `${d.class_name} (${confPercent}%)`;
          const fontSize = Math.max(14, Math.round(canvas.width / 36));
          ctx.font = `bold ${fontSize}px 'Plus Jakarta Sans', sans-serif`;
          const textWidth = ctx.measureText(label).width;
          const textHeight = fontSize + 4;

          ctx.fillStyle = color;
          ctx.fillRect(b.x1, Math.max(0, b.y1 - textHeight - 6), textWidth + 12, textHeight + 6);
          ctx.fillStyle = '#ffffff';
          ctx.fillText(label, b.x1 + 6, Math.max(textHeight, b.y1 - 4));

          // Tag kết quả
          const chip = document.createElement('div');
          chip.className = 'detection-chip';
          chip.style.backgroundColor = color + '20';
          chip.style.color = color;
          chip.innerText = `🥢 ${d.class_name}: ${confPercent}%`;
          tagsContainer.appendChild(chip);
        });
      }

      // THẺ TRI THỨC ẨM THỰC (KHÔNG LỘ NGUỒN AI)
      if (data.food_info) {
        const info = data.food_info;
        const ingBadges = (info.ingredients || []).map(i => `<span class="ingredient-badge">${i}</span>`).join('');

        infoContainer.innerHTML = `
          <div class="info-card">
            <div class="info-card-header">
              <h3>🍲 ${info.food_name || 'Thông Tin Món Ăn'}</h3>
              <span class="info-tag">✨ Phân Tích Chuyên Sâu</span>
            </div>
            <div class="info-item"><strong>📖 Giới thiệu:</strong> <p>${info.description || 'Chưa có thông tin.'}</p></div>
            <div class="info-item"><strong>📍 Nguồn gốc & Văn hóa:</strong> <p>${info.origin || 'Chưa có thông tin.'}</p></div>
            <div class="info-item"><strong>🥩 Nguyên liệu chính:</strong><div class="ingredients-tags">${ingBadges || 'Đang cập nhật'}</div></div>
            <div class="info-item"><strong>👅 Hương vị đặc trưng:</strong> <p>${info.taste || 'Đang cập nhật.'}</p></div>
            <div class="info-item"><strong>👨‍🍳 Cách chế biến & Thưởng thức:</strong> <p>${info.preparation || 'Đang cập nhật.'}</p></div>
            ${info.note ? `<div class="info-item"><strong>💡 Lưu ý ẩm thực:</strong> <p>${info.note}</p></div>` : ''}
          </div>
        `;
      } else if (data.message && detections.length > 0) {
        infoContainer.innerHTML = `<div style="font-size:0.88rem; color:#64748b; margin-top:8px;">ℹ️ ${data.message}</div>`;
      }
    }
  </script>
</body>
</html>
"""


# ============================================================
# ROOT ROUTE (HTML UI + JSON API COMPATIBILITY)
# ============================================================

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    """
    Trả về Web UI trực quan nếu truy cập từ trình duyệt,
    hoặc JSON nếu gọi từ API client.
    """
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

@app.get("/health")
def health():
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
# PREDICT (CHỈ CHẠY YOLO)
# ============================================================

@app.post(
    "/predict",
    response_model=PredictResponse,
)
async def predict(
    file: UploadFile = File(...),
):
    """
    Nhận ảnh và chạy YOLO object detection.
    """
    active_det = get_detector()
    if active_det is None:
        raise HTTPException(
            status_code=500,
            detail=f"Mô hình YOLO chưa sẵn sàng: {detector_error or 'Không xác định'}",
        )

    image = await read_image(file)

    try:
        detections = active_det.predict(image)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi nhận diện hình ảnh: {str(exc)}",
        ) from exc

    if not detections:
        return PredictResponse(
            success=True,
            detections=[],
            message=(
                "Không phát hiện được món ăn "
                "với ngưỡng confidence hiện tại."
            ),
        )

    return PredictResponse(
        success=True,
        detections=detections,
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

    try:
        return active_llm.get_food_info(
            request.food_name
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi tra cứu thông tin ẩm thực: {str(exc)}",
        ) from exc


# ============================================================
# ANALYZE (PIPELINE TOÀN DIỆN)
# ============================================================

@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
)
async def analyze(
    file: UploadFile = File(...),
):
    """
    Pipeline phân tích hoàn chỉnh:
    Ảnh -> YOLO nhận diện món -> Chọn món tin cậy nhất -> Phân tích tri thức ẩm thực chi tiết.
    """
    active_det = get_detector()
    if active_det is None:
        raise HTTPException(
            status_code=500,
            detail=f"Mô hình YOLO chưa sẵn sàng: {detector_error or 'Không xác định'}",
        )

    image = await read_image(file)

    # 1. YOLO Detect
    try:
        detections = active_det.predict(image)
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
            message="Không phát hiện được món ăn trong ảnh.",
        )

    # 2. Chọn món có confidence cao nhất
    best_detection = max(
        detections,
        key=lambda d: d.confidence,
    )
    food_name = best_detection.class_name

    # 3. Tra cứu tri thức ẩm thực
    active_llm = get_llm()
    if active_llm is None:
        return AnalyzeResponse(
            success=True,
            detections=detections,
            food_info=None,
            message=(
                f"Đã nhận diện: {food_name}. "
                "Hệ thống tri thức chưa được cấu hình API Key."
            ),
        )

    try:
        information = active_llm.get_food_info(food_name)
    except Exception as exc:
        return AnalyzeResponse(
            success=True,
            detections=detections,
            food_info=None,
            message=(
                f"Nhận diện được {food_name}, "
                f"nhưng chưa thể phân tích thông tin chi tiết: {str(exc)}"
            ),
        )

    return AnalyzeResponse(
        success=True,
        detections=detections,
        food_info=information,
        message="Phân tích hoàn tất.",
    )