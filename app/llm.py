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
    Class chịu trách nhiệm gọi Google Gemini LLM
    với cơ chế xoay mô hình tự động (Model Fallback / Rotation) khi gặp lỗi,
    hỗ trợ tra cứu thông tin và công thức cho một hoặc nhiều món ăn cùng lúc.
    """

    def __init__(self):
        if OpenAI is None:
            raise ImportError(
                "Thu vien 'openai' chua duoc cai dat. Hay chay: pip install openai"
            )

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
        Nhận tên món ăn và gọi Gemini để lấy thông tin chi tiết.
        Tự động xoay sang mô hình khác trong danh sách nếu mô hình hiện tại gặp lỗi.
        """
        food_name = food_name.strip()
        if not food_name:
            raise ValueError("Tên món ăn không được để trống.")

        prompt = f"""
Bạn là chuyên gia ẩm thực Việt Nam.

Hãy cung cấp thông tin và công thức chế biến về món ăn sau:
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

                if output.startswith("```"):
                    lines = output.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    output = "\n".join(lines).strip()

                match = re.search(r"\{.*\}", output, re.DOTALL)
                if match:
                    output = match.group(0)

                data: dict[str, Any] = json.loads(output)

                self.current_model_idx = idx
                print(f"[Gemini] Thanh cong voi model: {model_name}")

                return FoodInfoResponse(
                    food_name=data.get("food_name", food_name),
                    description=data.get("description", "Chưa có thông tin."),
                    origin=data.get("origin", "Chưa có thông tin."),
                    ingredients=data.get("ingredients", []),
                    taste=data.get("taste", "Chưa có thông tin."),
                    preparation=data.get("preparation", "Chưa có thông tin."),
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

        error_summary = " | ".join(errors[-3:])
        raise RuntimeError(
            f"Tất cả {total_models} mô hình Gemini đều gặp lỗi. Lỗi gần nhất: {error_summary}"
        )

    def get_foods_info(
        self,
        food_names: List[str],
    ) -> List[FoodInfoResponse]:
        """
        Nhận danh sách nhiều món ăn (khi phát hiện 2, 3 món hoặc nhiều hơn)
        và gọi Gemini lấy công thức, thông tin chi tiết cho TẤT CẢ các món đó.
        """
        # Lọc bỏ trùng lặp
        unique_names: List[str] = []
        seen = set()
        for name in food_names:
            clean = name.strip()
            if clean and clean.lower() not in seen:
                seen.add(clean.lower())
                unique_names.append(clean)

        if not unique_names:
            return []

        if len(unique_names) == 1:
            try:
                return [self.get_food_info(unique_names[0])]
            except Exception:
                return []

        # Xây dựng prompt lấy thông tin nhiều món trong 1 request
        list_str = "\n".join([f"{i+1}. {name}" for i, name in enumerate(unique_names)])
        prompt = f"""
Bạn là chuyên gia ẩm thực Việt Nam.

Hãy cung cấp thông tin chi tiết và công thức chế biến cho từng món ăn trong danh sách sau:
{list_str}

Trả lời bằng một mảng JSON (JSON array) hợp lệ duy nhất, mỗi phần tử tương ứng với một món ăn theo đúng cấu trúc sau:
[
  {{
    "food_name": "Tên món ăn",
    "description": "Mô tả ngắn gọn về món ăn...",
    "origin": "Nguồn gốc, xuất xứ, nét văn hóa...",
    "ingredients": ["Nguyên liệu 1", "Nguyên liệu 2", "..."],
    "taste": "Hương vị đặc trưng...",
    "preparation": "Cách chế biến và thưởng thức...",
    "note": "Lưu ý hoặc mẹo khi nấu/ăn (nếu có)..."
  }}
]

Yêu cầu:
- Viết bằng tiếng Việt chuẩn.
- Bắt buộc trả về đầy đủ thông tin cho tất cả các món ăn trong danh sách trên.
- ingredients phải là một danh sách các nguyên liệu chính.
- Chỉ trả về chuỗi JSON thuần túy, không định dạng markdown.
"""

        total_models = len(self.models)
        errors = []

        for attempt in range(total_models):
            idx = (self.current_model_idx + attempt) % total_models
            model_name = self.models[idx]

            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "Bạn là chuyên gia ẩm thực Việt Nam chỉ trả về mảng JSON hợp lệ duy nhất.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                )

                output = response.choices[0].message.content or ""
                output = output.strip()

                if output.startswith("```"):
                    lines = output.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    output = "\n".join(lines).strip()

                raw_data = None
                match_arr = re.search(r"\[.*\]", output, re.DOTALL)
                if match_arr:
                    try:
                        raw_data = json.loads(match_arr.group(0))
                    except Exception:
                        raw_data = None

                if raw_data is None:
                    match_obj = re.search(r"\{.*\}", output, re.DOTALL)
                    if match_obj:
                        try:
                            parsed_obj = json.loads(match_obj.group(0))
                            for val in parsed_obj.values():
                                if isinstance(val, list):
                                    raw_data = val
                                    break
                        except Exception:
                            raw_data = None

                if not isinstance(raw_data, list):
                    raise ValueError(f"Kết quả không phải mảng JSON hợp lệ: {output[:100]}")

                results: List[FoodInfoResponse] = []
                for item in raw_data:
                    if isinstance(item, dict):
                        results.append(
                            FoodInfoResponse(
                                food_name=item.get("food_name", "Món ăn"),
                                description=item.get("description", "Chưa có thông tin."),
                                origin=item.get("origin", "Chưa có thông tin."),
                                ingredients=item.get("ingredients", []),
                                taste=item.get("taste", "Chưa có thông tin."),
                                preparation=item.get("preparation", "Chưa có thông tin."),
                                note=item.get("note"),
                                model_used=model_name,
                            )
                        )

                if results:
                    self.current_model_idx = idx
                    print(f"[Gemini] Thanh cong lay thong tin {len(results)} mon voi model: {model_name}")
                    return results

            except Exception as exc:
                err_str = str(exc)
                errors.append(f"[{model_name}]: {err_str[:120]}")
                print(
                    f"[Gemini Rotate] Model '{model_name}' loi phan tich danh sach ({err_str[:60]}...). "
                    f"Dang chuyen sang model tiep theo..."
                )
                continue

        # Nếu batch call thất bại qua các model, gọi fallback tuần tự từng món
        print("[Gemini Fallback] Batch request that bai, thu goi tuan tu tung mon...")
        fallback_results: List[FoodInfoResponse] = []
        for name in unique_names:
            try:
                info = self.get_food_info(name)
                fallback_results.append(info)
            except Exception as e:
                print(f"[Gemini Fallback] Khong the lay thong tin cho '{name}': {e}")

        if fallback_results:
            return fallback_results

        error_summary = " | ".join(errors[-3:])
        raise RuntimeError(
            f"Không thể lấy thông tin món ăn từ các mô hình Gemini. Lỗi: {error_summary}"
        )