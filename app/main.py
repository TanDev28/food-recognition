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
        "API nhận diện món ăn Việt Nam bằng YOLO "
        "và cung cấp thông tin món ăn độc quyền bằng Google Gemini (kèm cơ chế tự xoay mô hình khi lỗi)."
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
# LOAD GEMINI LLM (VỚI CƠ CHẾ TỰ XOAY MÔ HÌNH)
# ============================================================

llm: Optional[FoodLLM] = None
llm_error: Optional[str] = None


def get_llm() -> Optional[FoodLLM]:
    """
    Lazy load Gemini LLM để nhận diện khi biến môi trường được thiết lập.
    Chỉ sử dụng Google Gemini (GEMINI_API_KEY hoặc GOOGLE_API_KEY).
    """
    global llm, llm_error
    if llm is not None:
        return llm

    has_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if has_api_key:
        try:
            llm = FoodLLM()
            llm_error = None
            print(f"[LLM] Khoi tao Gemini LLM thanh cong (Active model: {llm.get_active_model()}).")
        except Exception as e:
            llm_error = str(e)
            print(f"[LLM] Khong the khoi tao Gemini LLM: {e}".encode("ascii", errors="replace").decode("ascii"))

    return llm


# Thử load Gemini lúc khởi động
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
# WEB UI TEMPLATE
# ============================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Nhận Diện Món Ăn Việt Nam | YOLO & Gemini AI</title>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --primary: #e63946;
      --primary-hover: #d62828;
      --secondary: #1a73e8;
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --text: #1e293b;
      --text-muted: #64748b;
      --border: #e2e8f0;
      --radius: 16px;
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
      max-width: 1080px;
      margin: 0 auto;
    }

    header {
      text-align: center;
      margin-bottom: 32px;
    }

    header h1 {
      font-size: 2.2rem;
      font-weight: 800;
      color: #0f172a;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 12px;
      flex-wrap: wrap;
    }

    header p {
      color: var(--text-muted);
      margin-top: 8px;
      font-size: 1.05rem;
    }

    .badges {
      display: flex;
      justify-content: center;
      gap: 12px;
      margin-top: 14px;
      flex-wrap: wrap;
    }

    .badge {
      font-size: 0.85rem;
      padding: 6px 14px;
      border-radius: 999px;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      text-decoration: none;
    }
    .badge-yolo { background: #e0f2fe; color: #0369a1; }
    .badge-gemini { background: #e8f0fe; color: #1a73e8; border: 1px solid #c2e7ff; }
    .badge-docs { background: #f1f5f9; color: #334155; border: 1px solid var(--border); }
    .badge-docs:hover { background: #e2e8f0; }

    .main-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
    }

    @media (max-width: 860px) {
      .main-grid { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--card-bg);
      border-radius: var(--radius);
      padding: 24px;
      box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
      border: 1px solid var(--border);
    }

    .card h2 {
      font-size: 1.25rem;
      font-weight: 700;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .upload-zone {
      border: 2px dashed #cbd5e1;
      border-radius: 12px;
      padding: 32px 20px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
      background: #f8fafc;
    }
    .upload-zone:hover, .upload-zone.dragover {
      border-color: var(--primary);
      background: #fff5f5;
    }
    .upload-zone svg {
      width: 48px;
      height: 48px;
      color: #94a3b8;
      margin-bottom: 10px;
    }

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
    }
    .slider-group label { font-size: 0.9rem; font-weight: 600; }
    .slider-group span { font-weight: 700; color: var(--primary); }

    input[type=range] {
      width: 100%;
      accent-color: var(--primary);
      cursor: pointer;
    }

    .btn {
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
      transition: background 0.2s;
    }
    .btn:hover { background: var(--primary-hover); }
    .btn:disabled { opacity: 0.6; cursor: not-allowed; }

    #canvas-container {
      position: relative;
      width: 100%;
      min-height: 240px;
      background: #f1f5f9;
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
      border-radius: 12px;
    }

    .result-section {
      margin-top: 20px;
    }

    .detection-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 16px;
    }

    .detection-chip {
      background: #fee2e2;
      color: #991b1b;
      padding: 6px 12px;
      border-radius: 8px;
      font-weight: 700;
      font-size: 0.9rem;
    }

    .food-card {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 18px;
      margin-top: 14px;
    }

    .food-card h3 {
      color: #0f172a;
      font-size: 1.15rem;
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 6px;
    }

    .model-tag {
      font-size: 0.75rem;
      background: #e8f0fe;
      color: #1a73e8;
      padding: 3px 10px;
      border-radius: 999px;
      font-weight: 600;
    }

    .info-row {
      margin-bottom: 10px;
      font-size: 0.92rem;
    }
    .info-row strong { color: #334155; }

    .ingredients-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 4px;
    }
    .ingredient-badge {
      background: #ffffff;
      border: 1px solid #cbd5e1;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 0.82rem;
      color: #0f172a;
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

    .empty-state {
      color: #94a3b8;
      text-align: center;
      padding: 40px 10px;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>🍜 Nhận Diện Món Ăn Việt Nam</h1>
      <p>Nhận diện món ăn tự động bằng mô hình YOLO & tra cứu ẩm thực chi tiết với Google Gemini AI</p>
      <div class="badges">
        <span class="badge badge-yolo" id="model-status-badge">🎯 YOLO Model</span>
        <span class="badge badge-gemini" id="llm-status-badge">✨ Gemini AI (Auto-Rotation)</span>
        <a href="/docs" target="_blank" class="badge badge-docs">📖 Swagger API Docs</a>
        <a href="/health" target="_blank" class="badge badge-docs">🩺 Health</a>
      </div>
    </header>

    <div class="main-grid">
      <!-- CỘT TRÁI: UPLOAD & CÀI ĐẶT -->
      <div class="card">
        <h2>📷 Tải ảnh món ăn</h2>
        <div class="upload-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>
          <p><strong>Bấm để chọn ảnh</strong> hoặc kéo thả ảnh vào đây</p>
          <p style="font-size:0.85rem; color:#94a3b8; margin-top:4px;">JPG, PNG, WEBP (Phở, Bún bò, Bánh mì, Bánh xèo...)</p>
          <input type="file" id="file-input" accept="image/*" style="display:none">
        </div>

        <div class="controls">
          <div>
            <div class="slider-group">
              <label for="conf-slider">Độ tin cậy tối thiểu (Confidence):</label>
              <span id="conf-val">0.25</span>
            </div>
            <input type="range" id="conf-slider" min="0.05" max="0.95" step="0.05" value="0.25" oninput="document.getElementById('conf-val').innerText = this.value">
          </div>

          <div style="display: flex; gap: 12px;">
            <label style="display:flex; align-items:center; gap:6px; font-size:0.9rem; cursor:pointer;">
              <input type="radio" name="api-mode" value="analyze" checked> Phân tích đầy đủ (YOLO + Gemini)
            </label>
            <label style="display:flex; align-items:center; gap:6px; font-size:0.9rem; cursor:pointer;">
              <input type="radio" name="api-mode" value="predict"> Chỉ nhận diện YOLO
            </label>
          </div>

          <button class="btn" id="submit-btn" onclick="processImage()">
            <span id="btn-text">🔍 Phân Tích Món Ăn</span>
            <div class="spinner" id="btn-spinner" style="display:none;"></div>
          </button>
        </div>
      </div>

      <!-- CỘT PHẢI: KẾT QUẢ -->
      <div class="card">
        <h2>📊 Kết quả nhận diện</h2>
        <div id="canvas-container">
          <div class="empty-state" id="empty-state">
            <p>Chưa có ảnh được chọn</p>
          </div>
          <canvas id="result-canvas" style="display:none;"></canvas>
        </div>

        <div class="result-section" id="result-details" style="display:none;">
          <h3 style="font-size: 1rem; margin-bottom: 8px;">Món ăn phát hiện được:</h3>
          <div class="detection-tags" id="tags-container"></div>
          <div id="llm-info-container"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let selectedFile = null;
    let loadedImage = null;

    // Check health on load
    fetch('/health').then(r => r.json()).then(data => {
      const llmBadge = document.getElementById('llm-status-badge');
      if (data.llm_enabled) {
        llmBadge.innerText = `✨ Gemini (${data.llm_active_model || 'Auto-Rotate'})`;
        llmBadge.style.background = '#e8f0fe';
        llmBadge.style.color = '#1a73e8';
      } else {
        llmBadge.innerText = '⚠️ Chưa cấu hình GEMINI_API_KEY';
        llmBadge.style.background = '#fef9c3';
        llmBadge.style.color = '#854d0e';
      }

      const modelBadge = document.getElementById('model-status-badge');
      if (data.model_loaded) {
        modelBadge.innerText = '🎯 YOLO Ready';
        modelBadge.style.background = '#e0f2fe';
        modelBadge.style.color = '#0369a1';
      } else {
        modelBadge.innerText = '⚠️ YOLO Chưa tải';
        modelBadge.style.background = '#fee2e2';
        modelBadge.style.color = '#991b1b';
      }
    }).catch(() => {});

    const fileInput = document.getElementById('file-input');
    const dropZone = document.getElementById('drop-zone');

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

    function handleFile(file) {
      selectedFile = file;
      const reader = new FileReader();
      reader.onload = (e) => {
        loadedImage = new Image();
        loadedImage.onload = () => {
          showImagePreview();
        };
        loadedImage.src = e.target.result;
      };
      reader.readAsDataURL(file);
    }

    function showImagePreview() {
      const canvas = document.getElementById('result-canvas');
      const emptyState = document.getElementById('empty-state');
      canvas.width = loadedImage.naturalWidth;
      canvas.height = loadedImage.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(loadedImage, 0, 0);
      canvas.style.display = 'block';
      emptyState.style.display = 'none';
      document.getElementById('result-details').style.display = 'none';
    }

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
      btnText.innerText = "Đang phân tích...";
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
          throw new Error(data.detail || 'Lỗi xử lý');
        }

        renderResults(data);
      } catch (err) {
        alert("Lỗi: " + err.message);
      } finally {
        btn.disabled = false;
        btnText.innerText = "🔍 Phân Tích Món Ăn";
        spinner.style.display = "none";
      }
    }

    function renderResults(data) {
      const canvas = document.getElementById('result-canvas');
      const ctx = canvas.getContext('2d');
      ctx.drawImage(loadedImage, 0, 0);

      const detections = data.detections || [];
      const tagsContainer = document.getElementById('tags-container');
      const llmContainer = document.getElementById('llm-info-container');
      tagsContainer.innerHTML = '';
      llmContainer.innerHTML = '';

      if (detections.length === 0) {
        tagsContainer.innerHTML = '<span style="color:#64748b; font-style:italic;">Không tìm thấy món ăn nào với độ tin cậy này. Thử giảm confidence!</span>';
      } else {
        const colors = ['#e63946', '#2a9d8f', '#e76f51', '#457b9d', '#9b5de5'];
        detections.forEach((d, idx) => {
          const color = colors[idx % colors.length];
          const b = d.box;
          const confPercent = Math.round(d.confidence * 100);

          // Vẽ box
          ctx.lineWidth = Math.max(3, Math.round(canvas.width / 250));
          ctx.strokeStyle = color;
          ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);

          // Nhãn
          const label = `${d.class_name} (${confPercent}%)`;
          ctx.font = `bold ${Math.max(14, Math.round(canvas.width / 35))}px 'Plus Jakarta Sans', sans-serif`;
          const textWidth = ctx.measureText(label).width;
          const textHeight = Math.max(16, Math.round(canvas.width / 35));

          ctx.fillStyle = color;
          ctx.fillRect(b.x1, Math.max(0, b.y1 - textHeight - 6), textWidth + 10, textHeight + 6);
          ctx.fillStyle = '#ffffff';
          ctx.fillText(label, b.x1 + 5, Math.max(textHeight, b.y1 - 4));

          // Tag
          const chip = document.createElement('div');
          chip.className = 'detection-chip';
          chip.style.backgroundColor = color + '22';
          chip.style.color = color;
          chip.innerText = `🥢 ${d.class_name}: ${confPercent}%`;
          tagsContainer.appendChild(chip);
        });
      }

      // Thông tin Gemini nếu có
      if (data.food_info) {
        const info = data.food_info;
        const ingBadges = (info.ingredients || []).map(i => `<span class="ingredient-badge">${i}</span>`).join('');
        const modelLabel = info.model_used ? `<span class="model-tag">✨ Model: ${info.model_used}</span>` : '';

        llmContainer.innerHTML = `
          <div class="food-card">
            <h3>
              <span>🍜 ${info.food_name || 'Thông tin món ăn'}</span>
              ${modelLabel}
            </h3>
            <div class="info-row"><strong>📖 Mô tả:</strong> ${info.description || 'Không có thông tin'}</div>
            <div class="info-row"><strong>📍 Nguồn gốc:</strong> ${info.origin || 'Không có thông tin'}</div>
            <div class="info-row"><strong>🥩 Nguyên liệu chính:</strong><div class="ingredients-tags">${ingBadges || 'Đang cập nhật'}</div></div>
            <div class="info-row"><strong>👅 Hương vị:</strong> ${info.taste || 'Đang cập nhật'}</div>
            <div class="info-row"><strong>👨‍🍳 Cách chế biến:</strong> ${info.preparation || 'Đang cập nhật'}</div>
            ${info.note ? `<div class="info-row"><strong>💡 Ghi chú:</strong> ${info.note}</div>` : ''}
          </div>
        `;
      } else if (data.message && detections.length > 0) {
        llmContainer.innerHTML = `<div style="font-size:0.85rem; color:#64748b; margin-top:8px;">ℹ️ ${data.message}</div>`;
      }

      document.getElementById('result-details').style.display = 'block';
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
                "llm_provider": "Google Gemini",
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
        "llm_provider": "Google Gemini",
        "llm_enabled": active_llm is not None,
        "llm_active_model": active_llm.get_active_model() if active_llm else None,
        "llm_error": llm_error,
    }


# ============================================================
# PREDICT
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
            detail=f"Model YOLO chưa sẵn sàng: {detector_error or 'Không xác định'}",
        )

    image = await read_image(file)

    try:
        detections = active_det.predict(image)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi chạy YOLO: {str(exc)}",
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
# FOOD INFO
# ============================================================

@app.post(
    "/food-info",
    response_model=FoodInfoResponse,
)
def food_info(
    request: FoodInfoRequest,
):
    """
    Nhận tên món ăn và gọi Gemini LLM lấy thông tin chi tiết (tự động xoay model nếu lỗi).
    """
    active_llm = get_llm()
    if active_llm is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini LLM chưa được cấu hình. "
                "Hãy thiết lập biến môi trường GEMINI_API_KEY hoặc GOOGLE_API_KEY."
            ),
        )

    try:
        return active_llm.get_food_info(
            request.food_name
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi gọi Gemini LLM: {str(exc)}",
        ) from exc


# ============================================================
# ANALYZE
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
    Ảnh -> YOLO nhận diện món -> Chọn món tin cậy nhất -> Gemini LLM phân tích chi tiết.
    """
    active_det = get_detector()
    if active_det is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model YOLO chưa sẵn sàng: {detector_error or 'Không xác định'}",
        )

    image = await read_image(file)

    # 1. YOLO Detect
    try:
        detections = active_det.predict(image)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi chạy YOLO: {str(exc)}",
        ) from exc

    if not detections:
        return AnalyzeResponse(
            success=True,
            detections=[],
            food_info=None,
            message="Không phát hiện được món ăn.",
        )

    # 2. Chọn món có confidence cao nhất
    best_detection = max(
        detections,
        key=lambda d: d.confidence,
    )
    food_name = best_detection.class_name

    # 3. Gemini LLM Info
    active_llm = get_llm()
    if active_llm is None:
        return AnalyzeResponse(
            success=True,
            detections=detections,
            food_info=None,
            message=(
                f"Đã nhận diện: {food_name}. "
                "Gemini LLM chưa được cấu hình (thiết lập GEMINI_API_KEY hoặc GOOGLE_API_KEY để xem chi tiết)."
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
                f"YOLO nhận diện được {food_name}, "
                f"nhưng không lấy được thông tin từ Gemini: {str(exc)}"
            ),
        )

    return AnalyzeResponse(
        success=True,
        detections=detections,
        food_info=information,
        message="Phân tích hoàn tất.",
    )