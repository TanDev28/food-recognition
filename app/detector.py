from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import torch
    # Giới hạn số threads CPU để tiết kiệm tài nguyên và RAM trên Render 512MB
    torch.set_num_threads(2)
except Exception:
    torch = None

try:
    from app.schemas import BoundingBox, Detection
except ImportError:
    from schemas import BoundingBox, Detection


# Bảng chuẩn hóa tên món ăn sang tiếng Việt có dấu
VIETNAMESE_FOOD_NAMES = {
    "Banh canh": "Bánh canh",
    "Banh chung": "Bánh chưng",
    "Banh cuon": "Bánh cuốn",
    "Banh khot": "Bánh khọt",
    "Banh mi": "Bánh mì",
    "Banh trang": "Bánh tráng",
    "Banh trang tron": "Bánh tráng trộn",
    "Banh xeo": "Bánh xèo",
    "Bo kho": "Bò kho",
    "Bo la lot": "Bò lá lốt",
    "Bong cai": "Bông cải",
    "Bun": "Bún",
    "Bun bo Hue": "Bún bò Huế",
    "Bun cha": "Bún chả",
    "Bun dau": "Bún đậu",
    "Bun mam": "Bún mắm",
    "Bun rieu": "Bún riêu",
    "Ca": "Cá",
    "Ca chua": "Cà chua",
    "Ca phao": "Cà pháo",
    "Ca rot": "Cà rốt",
    "Canh": "Canh",
    "Cha": "Chả",
    "Cha gio": "Chả giò",
    "Chanh": "Chanh",
    "Com": "Cơm",
    "Com tam": "Cơm tấm",
    "Con nguoi": "Người",
    "Cu kieu": "Củ kiệu",
    "Cua": "Cua",
    "Dau hu": "Đậu hũ",
    "Dua chua": "Dưa chua",
    "Dua leo": "Dưa leo",
    "Goi cuon": "Gỏi cuốn",
    "Hamburger": "Hamburger",
    "Heo quay": "Heo quay",
    "Hu tieu": "Hủ tiếu",
    "Kho qua thit": "Khổ qua nhồi thịt",
    "Khoai tay chien": "Khoai tây chiên",
    "Lau": "Lẩu",
    "Long heo": "Lòng heo",
    "Mi": "Mì",
    "Muc": "Mực",
    "Nam": "Nấm",
    "Oc": "Ốc",
    "Ot chuong": "Ớt chuông",
    "Pho": "Phở",
    "Pho mai": "Phô mai",
    "Rau": "Rau",
    "Salad": "Salad",
    "Thit bo": "Thịt bò",
    "Thit ga": "Thịt gà",
    "Thit heo": "Thịt heo",
    "Thit kho": "Thịt kho",
    "Thit nuong": "Thịt nướng",
    "Tom": "Tôm",
    "Trung": "Trứng",
    "Xoi": "Xôi",
    "Banh beo": "Bánh bèo",
    "Cao lau": "Cao lầu",
    "Mi Quang": "Mì Quảng",
    "Com chien duong chau": "Cơm chiên Dương Châu",
    "Bun cha ca": "Bún chả cá",
    "Com chien ga": "Cơm chiên gà",
    "Chao long": "Cháo lòng",
    "Nom hoa chuoi": "Nộm hoa chuối",
    "Nui xao bo": "Nui xào bò",
    "Sup cua": "Súp cua",
}


def format_food_name(name: str) -> str:
    """Chuẩn hóa tên món ăn sang tiếng Việt có dấu."""
    if not name:
        return name
    name_clean = name.strip()
    return VIETNAMESE_FOOD_NAMES.get(name_clean, name_clean)


class FoodDetector:
    """
    Class chịu trách nhiệm:
    - Load model YOLO
    - Nhận ảnh
    - Chạy object detection
    - Chuyển kết quả YOLO thành dữ liệu BoundingBox & Detection có dấu tiếng Việt
    """

    def __init__(
        self,
        model_path: str = "models/best.pt",
        confidence_threshold: float = 0.20,
        image_size: int = 640,
    ):
        if YOLO is None:
            raise ImportError(
                "Thu vien 'ultralytics' chua duoc cai dat. Hay chay: pip install ultralytics"
            )

        model_file = Path(model_path)

        if not model_file.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file model YOLO: {model_file.resolve()}"
            )

        self.model = YOLO(str(model_file))
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size

    def predict(
        self,
        image: Image.Image,
        confidence_threshold: Optional[float] = None,
        original_size: Optional[tuple[int, int]] = None,
    ) -> List[Detection]:
        """
        Nhận một ảnh PIL và trả về danh sách detection với tên tiếng Việt có dấu,
        áp dụng đúng ngưỡng confidence và tự động scale tọa độ bounding box về kích thước ảnh gốc.
        """
        confidence = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else self.confidence_threshold
        )

        # Đảm bảo ảnh ở dạng RGB
        image = image.convert("RGB")
        curr_w, curr_h = image.size
        orig_w, orig_h = original_size if original_size else (curr_w, curr_h)
        scale_x = (orig_w / curr_w) if curr_w > 0 else 1.0
        scale_y = (orig_h / curr_h) if curr_h > 0 else 1.0

        # Tối ưu hóa bộ nhớ inference
        ctx = torch.inference_mode() if torch is not None else nullcontext()
        with ctx:
            results = self.model.predict(
                source=image,
                imgsz=self.image_size,
                conf=confidence,
                verbose=False,
            )

        detections: List[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                confidence_score = float(boxes.conf[i].item())

                # Đảm bảo lọc chặt chẽ theo đúng ngưỡng confidence
                if confidence_score < confidence:
                    continue

                xyxy = boxes.xyxy[i].tolist()
                x1, y1, x2, y2 = xyxy

                # Scale tọa độ Bounding Box về kích thước ảnh gốc hiển thị trên Canvas
                scaled_x1 = max(0.0, min(float(orig_w), float(x1) * scale_x))
                scaled_y1 = max(0.0, min(float(orig_h), float(y1) * scale_y))
                scaled_x2 = max(0.0, min(float(orig_w), float(x2) * scale_x))
                scaled_y2 = max(0.0, min(float(orig_h), float(y2) * scale_y))

                # Lấy tên class từ model và chuẩn hóa sang tiếng Việt có dấu
                raw_class_name = self.model.names.get(class_id, f"Class {class_id}")
                class_name = format_food_name(raw_class_name)

                detection = Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence_score,
                    box=BoundingBox(
                        x1=scaled_x1,
                        y1=scaled_y1,
                        x2=scaled_x2,
                        y2=scaled_y2,
                    ),
                )

                detections.append(detection)

        return detections

    def get_class_names(self) -> dict:
        """
        Trả về danh sách class đã được chuẩn hóa tiếng Việt.
        """
        if hasattr(self.model, "names"):
            return {
                cid: format_food_name(name)
                for cid, name in self.model.names.items()
            }
        return {}