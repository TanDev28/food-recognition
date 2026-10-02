import json
import os
import re
from typing import Any, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from app.schemas import FoodInfoResponse
except ImportError:
    from schemas import FoodInfoResponse


# ============================================================
# DANH SÁCH TOÀN BỘ MÔ HÌNH GEMINI ĐƯỢC CẤU HÌNH XOAY VÒNG
# ============================================================
# Thứ tự ưu tiên: các mô hình Text/Flash/Pro hiệu năng cao xếp đầu
# để tối ưu tốc độ và độ chính xác tạo JSON, sau đó đến các mô hình mở rộng.
GEMINI_MODELS_POOL: List[str] = [
    # Nhóm 1: Mô hình Flash & Pro xử lý Text/JSON tối ưu nhất
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-omni-1.1-flash",
    # Nhóm 2: Live, Extended Thinking, Multimodal
    "gemini-3.8-live-extended-thinking",
    "gemini-3.8-live",
    "gemini-3.1-flash-live-preview",
    "gemini-2.5-flash-native-audio-preview-12-2025",
    "gemini-3.5-live-translate-preview",
    # Nhóm 3: Image (Nano Banana series)
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image",
    "gemini-3-pro-image",
    "gemini-2.5-flash-image",
    # Nhóm 4: TTS & Speech
    "gemini-3.8-flash-tts",
    "gemini-3.8-flash-lite-tts",
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-preview-tts",
    # Nhóm 5: Transcribe
    "gemini-3.5-transcribe",
    "gemini-3.5-transcribe-live",
]


class FoodLLM:
    """
    Class chịu trách nhiệm gọi duy nhất Google Gemini LLM
    với cơ chế xoay mô hình tự động (Model Fallback / Rotation) khi gặp lỗi.
    """

    def __init__(self):
        if OpenAI is None:
            raise ImportError(
                "Thu vien 'openai' chua duoc cai dat. Hay chay: pip install openai"
            )

        # Lấy API Key từ các biến môi trường của Gemini / Google
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise RuntimeError(
                "Chua cau hinh GEMINI_API_KEY hoac GOOGLE_API_KEY."
            )

        self.api_key = api_key
        self.base_url = os.getenv(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )

        # Cấu hình danh sách mô hình
        custom_models_env = os.getenv("GEMINI_MODELS")
        if custom_models_env:
            self.models = [m.strip() for m in custom_models_env.split(",") if m.strip()]
        else:
            self.models = list(GEMINI_MODELS_POOL)

        # Nếu người dùng chỉ định một model cụ thể qua GEMINI_MODEL, ưu tiên đặt lên đầu
        preferred_model = os.getenv("GEMINI_MODEL")
        if preferred_model:
            preferred_model = preferred_model.strip()
            if preferred_model in self.models:
                self.models.remove(preferred_model)
            self.models.insert(0, preferred_model)

        timeout = float(os.getenv("GEMINI_TIMEOUT", "15.0"))
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
        )

        self.current_model_idx = 0

    def get_active_model(self) -> str:
        """
        Trả về model hiện đang được kích hoạt mặc định.
        """
        if 0 <= self.current_model_idx < len(self.models):
            return self.models[self.current_model_idx]
        return self.models[0]

    def get_food_info(
        self,
        food_name: str,
    ) -> FoodInfoResponse:
        """
        Nhận tên món ăn và gọi Gemini để lấy thông tin.
        Tự động xoay sang mô hình khác trong danh sách nếu mô hình hiện tại gặp lỗi
        (Rate limit 429, 404, Quota exceeded, Timeout, 503, v.v.).
        """
        food_name = food_name.strip()
        if not food_name:
            raise ValueError("Tên món ăn không được để trống.")

        prompt = f"""
Bạn là chuyên gia ẩm thực Việt Nam.

Hãy cung cấp thông tin về món ăn sau:
{food_name}

Trả lời bằng JSON hợp lệ duy nhất với đúng cấu trúc sau (không kèm văn bản khác ngoài JSON):
{{
    "food_name": "{food_name}",
    "description": "...",
    "origin": "...",
    "ingredients": ["...", "..."],
    "taste": "...",
    "preparation": "...",
    "note": "..."
}}

Yêu cầu:
- Viết bằng tiếng Việt chuẩn.
- Nội dung súc tích, hấp dẫn, dễ hiểu.
- ingredients phải là một danh sách (mảng) các nguyên liệu chính.
- Chỉ trả về chuỗi JSON thuần túy, không định dạng markdown.
"""

        total_models = len(self.models)
        errors = []

        # Xoay vòng qua danh sách model bắt đầu từ model hiện tại
        for attempt in range(total_models):
            idx = (self.current_model_idx + attempt) % total_models
            model_name = self.models[idx]

            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "Bạn là chuyên gia ẩm thực Việt Nam chỉ trả về định dạng JSON hợp lệ duy nhất.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                )

                output = response.choices[0].message.content or ""
                output = output.strip()

                # Làm sạch markdown code block (```json ... ```) nếu có
                if output.startswith("```"):
                    lines = output.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    output = "\n".join(lines).strip()

                # Regex tìm chuỗi JSON giữa dấu ngoặc nhọn { ... }
                match = re.search(r"\{.*\}", output, re.DOTALL)
                if match:
                    output = match.group(0)

                data: dict[str, Any] = json.loads(output)

                # Thành công: Cập nhật current_model_idx và trả kết quả
                self.current_model_idx = idx
                print(f"[Gemini] Thanh cong voi model: {model_name}")

                return FoodInfoResponse(
                    food_name=data.get("food_name", food_name),
                    description=data.get(
                        "description",
                        "Chưa có thông tin."
                    ),
                    origin=data.get(
                        "origin",
                        "Chưa có thông tin."
                    ),
                    ingredients=data.get(
                        "ingredients",
                        []
                    ),
                    taste=data.get(
                        "taste",
                        "Chưa có thông tin."
                    ),
                    preparation=data.get(
                        "preparation",
                        "Chưa có thông tin."
                    ),
                    note=data.get("note"),
                    model_used=model_name,
                )

            except Exception as exc:
                err_str = str(exc)
                errors.append(f"[{model_name}]: {err_str[:120]}")
                print(
                    f"[Gemini Rotate] Model '{model_name}' loi ({err_str[:60]}...). "
                    f"Dang tu dong chuyen sang model tiep theo..."
                )
                continue

        # Nếu toàn bộ các mô hình trong danh sách đều gặp sự cố
        error_summary = " | ".join(errors[-3:])
        raise RuntimeError(
            f"Tất cả {total_models} mô hình Gemini đều gặp lỗi. Lỗi gần nhất: {error_summary}"
        )