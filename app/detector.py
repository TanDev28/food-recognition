from pathlib import Path
from typing import List, Optional

from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from app.schemas import BoundingBox, Detection
except ImportError:
    from schemas import BoundingBox, Detection


class FoodDetector:
    """
    Class chịu trách nhiệm:
    - Load model YOLO
    - Nhận ảnh
    - Chạy object detection
    - Chuyển kết quả YOLO thành dữ liệu BoundingBox & Detection
    """

    def __init__(
        self,
        model_path: str = "models/best.pt",
        confidence_threshold: float = 0.25,
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
    ) -> List[Detection]:
        """
        Nhận một ảnh PIL và trả về danh sách detection.
        """
        confidence = (
            confidence_threshold
            if confidence_threshold is not None
            else self.confidence_threshold
        )

        # Đảm bảo ảnh ở dạng RGB
        image = image.convert("RGB")

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

                xyxy = boxes.xyxy[i].tolist()
                x1, y1, x2, y2 = xyxy

                # Lấy tên class từ model
                class_name = self.model.names.get(class_id, f"Class {class_id}")

                detection = Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence_score,
                    box=BoundingBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                    ),
                )

                detections.append(detection)

        return detections

    def get_class_names(self) -> dict:
        """
        Trả về danh sách class mà model nhận biết.
        """
        if hasattr(self.model, "names"):
            return self.model.names
        return {}