from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: BoundingBox


class PredictResponse(BaseModel):
    success: bool
    detections: List[Detection]
    message: Optional[str] = None


class FoodInfoRequest(BaseModel):
    food_name: str


class FoodInfoResponse(BaseModel):
    food_name: str
    description: str
    origin: str
    ingredients: List[str]
    taste: str
    preparation: str
    note: Optional[str] = None
    model_used: Optional[str] = None
    en: Optional[dict] = None


class AnalyzeResponse(BaseModel):
    success: bool
    detections: List[Detection]
    food_info: Optional[FoodInfoResponse] = None
    foods_info: List[FoodInfoResponse] = Field(default_factory=list)
    message: Optional[str] = None