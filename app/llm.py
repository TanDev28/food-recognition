import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from app.schemas import FoodInfoResponse
except ImportError:
    from schemas import FoodInfoResponse


# ============================================================
# DANH SÁCH MÔ HÌNH GEMINI (ƯU TIÊN MÔ HÌNH NHANH & XOAY VÒNG THÔNG MINH)
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
    # Nhóm dự phòng chính thức siêu tốc độ (<1.5s response)
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]


# ============================================================
# BỘ NHỚ ĐỆM TRI THỨC ẨM THỰC (IN-MEMORY CACHE - PHẢN HỒI 0MS)
# ============================================================
_PRESEEDED_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
    "cơm chiên dương châu": {
        "food_name": "Cơm chiên Dương Châu",
        "description": "Món cơm chiên trứ danh kết hợp hài hòa giữa hạt cơm tơi xốp cùng lạp xưởng, tôm, trứng và các loại rau củ rực rỡ sắc màu.",
        "origin": "Bắt nguồn từ vùng Dương Châu (Trung Quốc) và đã trở thành món ăn quen thuộc, rất được yêu thích trong văn hóa ẩm thực Việt Nam.",
        "ingredients": ["Cơm nguội", "Lạp xưởng", "Tôm tươi", "Trứng gà", "Đậu Hà Lan", "Cà rốt", "Hành lá", "Tỏi phi"],
        "taste": "Vị mặn ngọt béo bùi đan xen, hạt cơm săn giòn vàng ươm, dậy mùi thơm nức của lạp xưởng và hành tỏi.",
        "preparation": "Xào chín tôm và lạp xưởng thái hạt lựu, sau đó cho cơm nguội vào đảo đều cùng trứng đánh tan trên lửa lớn đến khi hạt cơm săn lại.",
        "note": "Nên dùng cơm nguội để trong ngăn mát tủ lạnh qua đêm để khi chiên hạt cơm tơi xốp, không bị nhão.",
        "en": {
            "food_name": "Yangzhou Fried Rice",
            "description": "A famous savory fried rice dish tossed with seasoned eggs, Chinese sausage, shrimp, and diced colorful vegetables.",
            "origin": "Originated in Yangzhou, China, and has become a beloved staple in Vietnamese culinary culture.",
            "ingredients": ["Cold cooked rice", "Chinese sausage", "Shrimp", "Eggs", "Green peas", "Carrots", "Scallions"],
            "taste": "Savory, aromatic, and rich with tender, fluffy separated rice grains.",
            "preparation": "Sauté diced meats and vegetables, then stir-fry with chilled rice and beaten eggs over high heat until grains are firm and fragrant.",
            "note": "Using chilled day-old rice prevents sogginess and guarantees a fluffy, separate grain texture."
        }
    },
    "phở": {
        "food_name": "Phở",
        "description": "Quốc hồn quốc túy của ẩm thực Việt Nam với bánh phở mềm mịn, nước dùng trong vắt thanh ngọt được ninh từ xương bò cùng các loại thảo mộc quý.",
        "origin": "Xuất phát từ miền Bắc (Nam Định và Hà Nội) từ đầu thế kỷ 20, hiện là biểu tượng ẩm thực đại diện cho Việt Nam trên toàn cầu.",
        "ingredients": ["Bánh phở", "Xương ống bò", "Thịt bò (tái/nạm/gầu)", "Hoa hồi", "Quế chi", "Thảo quả", "Gừng nướng", "Hành hoa", "Rau mùi"],
        "taste": "Nước dùng ngọt thanh đậm đà từ tủy xương, dậy mùi thơm thảo mộc ấm áp, hòa quyện với vị mềm ngọt của thịt bò.",
        "preparation": "Ninh xương bò nhiều giờ cùng gia vị hồi, quế, thảo quả nướng; chần bánh phở nóng, xếp thịt bò thái mỏng lên trên rồi chan nước dùng sôi sùng sục.",
        "note": "Thưởng thức ngay khi còn nóng hổi kèm chanh tươi, ớt hiểm, dấm tỏi và quẩy giòn để cảm nhận trọn vẹn hương vị.",
        "en": {
            "food_name": "Pho Noodle Soup (Phở)",
            "description": "Vietnam's national dish consisting of delicate flat rice noodles served in a rich, clear broth simmered for hours with beef bones and fragrant spices.",
            "origin": "Originated in Northern Vietnam (Hanoi and Nam Dinh) in the early 20th century, celebrated worldwide.",
            "ingredients": ["Rice noodles", "Beef marrow bones", "Beef cuts (brisket/flank/sirloin)", "Star anise", "Cinnamon", "Cardamom", "Charred ginger", "Scallions", "Cilantro"],
            "taste": "Rich yet light and clear broth with deep umami sweetness, aromatic warm spices, and tender beef.",
            "preparation": "Simmer beef bones and charred spices for 8-12 hours; scald fresh noodles, top with sliced beef, and pour piping hot broth over.",
            "note": "Best enjoyed steaming hot with fresh lime, chili slices, and fragrant herbs."
        }
    },
    "cơm tấm": {
        "food_name": "Cơm tấm",
        "description": "Món ăn đặc sản Nam Bộ với hạt gạo tấm dẻo bùi, ăn kèm sườn heo nướng mật ong thơm lừng, bì dai giòn, chả trứng béo ngậy và nước mắm chua ngọt.",
        "origin": "Đặc trưng của Sài Gòn và các tỉnh Tây Nam Bộ, từ món ăn bình dân của người lao động nay đã trở thành nét văn hóa đặc sắc.",
        "ingredients": ["Gạo tấm", "Sườn cốt lết", "Bì heo", "Trứng gà", "Thịt băm", "Mộc nhĩ", "Mỡ hành", "Đồ chua", "Nước mắm tỏi ớt"],
        "taste": "Thịt sườn nướng mặn ngọt đậm đà xém cạnh thơm phức, chan mỡ hành béo ngậy cùng nước mắm chua ngọt cay nhẹ kích thích vị giác.",
        "preparation": "Nấu cơm từ gạo tấm; sườn ướp gia vị đậm đà nướng trên than hồng; chả trứng hấp chín vàng; xếp lên đĩa cùng đồ chua và rưới mỡ hành.",
        "note": "Linh hồn của món ăn nằm ở bát nước mắm chua ngọt pha sánh kẹo với ớt băm và đồ chua củ cải cà rốt giòn sần sật.",
        "en": {
            "food_name": "Broken Rice (Cơm tấm)",
            "description": "Iconic southern Vietnamese specialty made from fractured rice grains, served with charcoal-grilled pork chops, steamed egg meatloaf, shredded pork skin, and sweet fish sauce.",
            "origin": "Originated in Saigon and Southern Vietnam, evolving from a humble workers' meal into a world-famous culinary icon.",
            "ingredients": ["Broken rice", "Pork chops", "Shredded pork skin", "Eggs", "Minced pork", "Wood ear mushrooms", "Scallion oil", "Pickled vegetables", "Garlic chili fish sauce"],
            "taste": "Smoky sweet-savory grilled pork with crispy caramelized edges, creamy egg meatloaf, and zesty dipping sauce.",
            "preparation": "Steam broken rice until fluffy; grill marinated pork over charcoal embers; steam meatloaf; assemble with scallion oil and pickled daikon.",
            "note": "The dish is defined by the sweet-savory garlic-chili dipping fish sauce."
        }
    },
    "bánh mì": {
        "food_name": "Bánh mì",
        "description": "Bánh mì giòn rụm bên ngoài, xốp mềm bên trong, kẹp nhân pate gan béo ngậy, chả lụa, thịt nướng, đồ chua giòn ngọt và sốt đậm đà.",
        "origin": "Giao thoa văn hóa giữa ẩm thực Pháp và khẩu vị Việt Nam, được vinh danh là một trong những món ăn đường phố ngon nhất thế giới.",
        "ingredients": ["Vỏ bánh mì", "Pate gan", "Chả lụa", "Thịt nướng/xá xíu", "Dưa leo", "Đồ chua", "Rau mùi", "Ớt tươi", "Nước sốt đặc biệt"],
        "taste": "Giòn rụm, béo ngậy của pate, chua ngọt sảng khoái của dưa góp và cay the ấm áp của ớt tươi.",
        "preparation": "Nướng giòn vỏ bánh, rạch một bên sườn, phết đều pate và bơ trứng, xếp lớp thịt, giò chả, rau thơm dưa leo và rưới sốt cay ngọt.",
        "note": "Bánh mì ngon nhất khi thưởng thức ngay lúc vừa nướng nóng giòn.",
        "en": {
            "food_name": "Vietnamese Baguette (Bánh mì)",
            "description": "Crispy French-influenced baguette with airy interior, packed with savory liver pâté, Vietnamese cold cuts, pickled vegetables, and fresh herbs.",
            "origin": "A culinary fusion born in Saigon, globally acclaimed as one of the best street sandwiches.",
            "ingredients": ["Crispy baguette", "Liver pâté", "Vietnamese pork roll", "Grilled pork", "Cucumber", "Pickled carrots & daikon", "Cilantro", "Fresh chili", "Savory sauce"],
            "taste": "Remarkable contrast of textures: crunchy crust, rich creamy pâté, refreshing sour pickles, and aromatic fresh herbs.",
            "preparation": "Toast baguette until shatteringly crisp, slice lengthwise, spread rich pâté and mayonnaise, layer meats, pickles, cilantro, and chili sauce.",
            "note": "Best eaten freshly made while the crust remains piping hot and crispy."
        }
    },
    "bánh xèo": {
        "food_name": "Bánh xèo",
        "description": "Vỏ bánh vàng ươm mỏng giòn rụm từ bột gạo và nước cốt dừa, cuộn nhân tôm, thịt ba chỉ, giá đỗ, cuốn cùng các loại rau rừng và chấm mắm chua ngọt.",
        "origin": "Món ăn dân dã phổ biến khắp miền Trung và miền Nam với đặc trưng tiếng 'xèo xèo' vui tai khi tráng bánh trên chảo dầu nóng.",
        "ingredients": ["Bột gạo", "Bột nghệ", "Nước cốt dừa", "Tôm đất", "Thịt ba chỉ", "Giá đỗ", "Hành lá", "Rau cải xanh", "Rau thơm", "Nước mắm chua ngọt"],
        "taste": "Vỏ bánh giòn rụm béo ngậy hương dừa, nhân tôm thịt ngọt tươi, ăn kèm rau xanh mát và nước chấm hài hòa không hề ngấy.",
        "preparation": "Pha bột gạo với nghệ và nước cốt dừa; cho tôm thịt vào chảo nóng, múc bột tráng mỏng quanh chảo, rải giá đỗ lên trên và đậy nắp đến khi vỏ vàng giòn.",
        "note": "Bánh xèo ngon nhất khi cuốn cùng lá cải xanh, rau xà lách, chấm ngập trong nước mắm tỏi ớt chua ngọt.",
        "en": {
            "food_name": "Crispy Vietnamese Pancake (Bánh xèo)",
            "description": "Sizzling crispy crepe made from rice flour, turmeric, and coconut milk, stuffed with shrimp, pork belly, and bean sprouts.",
            "origin": "Beloved folk delicacy across Central and Southern Vietnam, named after the loud sizzling sound made when batter hits the hot skillet.",
            "ingredients": ["Rice flour", "Turmeric powder", "Coconut milk", "Fresh shrimp", "Pork belly", "Bean sprouts", "Scallions", "Mustard greens", "Fresh herbs", "Sweet-sour dipping sauce"],
            "taste": "Super crisp exterior with delicate coconut fragrance, savory sweet shrimp and pork fillings, balanced by crisp bitter mustard leaves.",
            "preparation": "Sauté shrimp and pork in a sizzling pan, ladle thin batter in a swirling motion, add bean sprouts, cover until crust turns golden brown and crispy.",
            "note": "Wrap wedges in fresh mustard leaves and herbs, dip generously in garlic-lime fish sauce."
        }
    },
    "bún bò huế": {
        "food_name": "Bún bò Huế",
        "description": "Món bún trứ danh Cố đô Huế với sợi bún to tròn, nước dùng đậm đà thơm ngát hương sả và mắm ruốc, ăn cùng bắp bò, chả cua và tiết luộc.",
        "origin": "Đỉnh cao ẩm thực Cung đình và dân gian xứ Huế, mang hương vị nồng nàn đặc trưng của miền Trung nắng gió.",
        "ingredients": ["Bún sợi to", "Bắp bò", "Gân bò", "Giò heo", "Chả cua", "Huyết luộc", "Sả cây", "Mắm ruốc Huế", "Hạt điều màu", "Rau hoa chuối", "Rau muống chẻ"],
        "taste": "Vị cay nồng ấm, ngọt đậm từ tủy xương và mắm ruốc Huế, dậy mùi thơm lừng của sả phi dầu điều.",
        "preparation": "Hầm bắp bò và giò heo với nhiều sả đập dập; nêm mắm ruốc đã lọc trong; chưng dầu điều tạo màu đỏ cam hấp dẫn; chần bún và chan nước dùng cùng rau sống.",
        "note": "Không thể thiếu đĩa rau sống gồm hoa chuối thái mỏng, rau muống chẻ và vài lát chanh ớt cay xé lưỡi.",
        "en": {
            "food_name": "Hue Spicy Beef Noodles (Bún bò Huế)",
            "description": "Renowned royal court noodle soup featuring thick round rice noodles, tender beef shank, and a fiery, lemongrass-infused broth seasoned with fermented shrimp paste.",
            "origin": "Ancient imperial capital Hue, representing the pinnacle of Central Vietnamese royal culinary artistry.",
            "ingredients": ["Thick rice noodles", "Beef shank", "Pork knuckle", "Crab meatball", "Congealed pork blood", "Lemongrass stalks", "Hue shrimp paste", "Annatto oil", "Banana blossoms", "Morning glory"],
            "taste": "Bold, spicy, deeply aromatic with lemongrass fragrance, rich umami depth from fermented shrimp paste, and vibrant red chili oil.",
            "preparation": "Simmer beef shank and pork with bruised lemongrass; season with strained shrimp paste and red annatto oil; ladle over warm noodles with fresh herbs.",
            "note": "Serve piping hot with shredded banana blossoms, water spinach, and freshly squeezed lime."
        }
    },
    "trứng": {
        "food_name": "Trứng",
        "description": "Thành phần giàu dinh dưỡng và quen thuộc trong ẩm thực Việt Nam, thường được chế biến dạng ốp la lòng đào hoặc chiên vàng thơm ngậy ăn kèm các món cơm, bánh mì.",
        "origin": "Phổ biến trong ẩm thực toàn cầu và là món ăn kèm kinh điển của các món cơm tấm, bánh mì chảo tại Việt Nam.",
        "ingredients": ["Trứng gà/vịt", "Dầu ăn/bơ", "Tiêu đen", "Nước tương hoặc nước mắm", "Hành lá"],
        "taste": "Vị béo ngậy, bùi bùi của lòng đỏ tan chảy hòa quyện cùng lớp lòng trắng viền giòn xém cạnh thơm lừng.",
        "preparation": "Đun nóng chảo dầu hoặc bơ; đập trứng trực tiếp vào chảo với lửa vừa để viền giòn vàng và lòng đỏ lòng đào; rắc chút tiêu và hành hoa.",
        "note": "Ngon nhất khi thưởng thức nóng hổi, lòng đào sánh mịn chấm cùng nước mắm chua ngọt hoặc nước tương tỏi ớt.",
        "en": {
            "food_name": "Fried Egg (Trứng ốp la)",
            "description": "Nutritious and comforting staple in Vietnamese cuisine, commonly prepared sunny-side up with a velvety runny yolk to accompany broken rice or baguettes.",
            "origin": "Ubiquitous breakfast and topping tradition across Vietnam, especially in Saigon's broken rice stalls.",
            "ingredients": ["Fresh eggs", "Cooking oil or butter", "Black pepper", "Soy sauce or fish sauce", "Scallions"],
            "taste": "Rich, silky running yolk paired with crispy golden edges and savory aroma.",
            "preparation": "Heat oil in a skillet; crack egg directly into the pan over medium heat until edges crisp while keeping yolk liquid; finish with black pepper.",
            "note": "Best enjoyed steaming hot, mixing the velvety runny yolk into warm rice or dipping with bread."
        }
    },
    "trứng ốp la": {
        "food_name": "Trứng ốp la",
        "description": "Món trứng chiên lòng đào thơm lừng với viền lòng trắng giòn rụm và lòng đỏ sánh mịn béo ngậy, món ăn kèm không thể thiếu của cơm tấm và bánh mì.",
        "origin": "Du nhập từ Pháp (oeuf au plat) và trở thành nét ẩm thực đường phố đặc trưng thân thuộc của người Việt.",
        "ingredients": ["Trứng gà", "Bơ hoặc dầu ăn", "Hạt tiêu xay", "Nước tương", "Mỡ hành"],
        "taste": "Lòng đỏ béo ngậy bùi thơm, lòng trắng mềm ngọt với viền xém giòn tan đậm đà.",
        "preparation": "Làm nóng chảo với ít bơ; chiên trứng nhanh tay ở nhiệt độ thích hợp để đạt độ lòng đào hoàn hảo; rưới thêm mỡ hành hoặc tiêu.",
        "note": "Rưới một thìa mỡ hành thơm phức và nước mắm ớt lên trên lòng đào khi ăn cùng cơm tấm để tăng hương vị.",
        "en": {
            "food_name": "Sunny-Side Up Egg (Trứng ốp la)",
            "description": "Classic sunny-side up fried egg featuring crispy caramelized edges and a rich runny center, an indispensable topping for Vietnamese broken rice.",
            "origin": "French culinary influence seamlessly adapted into everyday Vietnamese meals.",
            "ingredients": ["Fresh eggs", "Butter or cooking oil", "Ground black pepper", "Soy sauce", "Scallion oil"],
            "taste": "Silky, creamy, umami-rich yolk with delightfully crisp borders.",
            "preparation": "Fry gently in buttered skillet until white sets with crispy borders while yolk stays soft; top with scallion oil.",
            "note": "Drizzle fragrant scallion oil and seasoned fish sauce over the egg for the authentic flavor."
        }
    },
    "thịt nướng": {
        "food_name": "Thịt nướng",
        "description": "Thịt heo tẩm ướp đậm đà gia vị sả, tỏi, mật ong và nước mắm truyền thống, nướng xém cạnh trên than hoa đỏ rực dậy mùi thơm quyến rũ.",
        "origin": "Đặc trưng của ẩm thực Nam Bộ và miền Trung, xuất hiện chủ đạo trong cơm tấm, bún thịt nướng, bánh ướt.",
        "ingredients": ["Thịt ba chỉ hoặc nạc vai", "Sả băm", "Hành tỏi băm", "Mật ong", "Nước mắm", "Dầu hào", "Tiêu", "Mè rang"],
        "taste": "Đậm đà mặn ngọt hài hòa, thơm nức mùi sả và khói than hoa, miếng thịt mềm mọng nước không bị khô.",
        "preparation": "Ướp thịt thái mỏng cùng hỗn hợp sốt sả mật ong ít nhất 2 giờ; kẹp vỉ nướng trên than hoa chín vàng đều 2 mặt.",
        "note": "Thịt nướng ngon nhất khi nướng than hoa vừa chín tới, viền hơi xém cạnh bóng bẩy nước sốt.",
        "en": {
            "food_name": "Charcoal-Grilled Pork (Thịt nướng)",
            "description": "Marinated pork slices grilled over glowing charcoal embers with lemongrass, garlic, honey, and fish sauce.",
            "origin": "A culinary hallmark across Southern and Central Vietnam, celebrated in broken rice and noodle bowls.",
            "ingredients": ["Pork shoulder or belly", "Minced lemongrass", "Garlic & shallots", "Honey", "Fish sauce", "Oyster sauce", "Black pepper", "Sesame seeds"],
            "taste": "Smoky, sweet-savory perfection with caramelized edges and juicy, tender texture.",
            "preparation": "Marinate thin pork slices with lemongrass-honey sauce; grill over hot charcoal turning frequently until golden brown.",
            "note": "Best served straight off the grill with smoky aroma and glossy glaze."
        }
    }
}

# Khởi tạo bộ nhớ đệm
_FOOD_INFO_CACHE: Dict[str, FoodInfoResponse] = {}
for _k, _v in _PRESEEDED_KNOWLEDGE.items():
    _FOOD_INFO_CACHE[_k] = FoodInfoResponse(
        food_name=_v["food_name"],
        description=_v["description"],
        origin=_v["origin"],
        ingredients=_v["ingredients"],
        taste=_v["taste"],
        preparation=_v["preparation"],
        note=_v.get("note"),
        model_used="cache-instant",
        en=_v.get("en"),
    )


class FoodLLM:
    """
    Class chịu trách nhiệm gọi Google Gemini LLM
    với cơ chế xoay mô hình tự động (Model Fallback / Rotation) khi gặp lỗi,
    kết hợp In-Memory Cache giúp phản hồi tức thì và không bị nghẽn mạng.
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

        # Giảm timeout xuống 6.0s để fail-fast các mô hình không tồn tại, tránh người dùng phải chờ 60s
        timeout = float(os.getenv("GEMINI_TIMEOUT", "6.0"))
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
        )

        self.current_model_idx = 0

    def _promote_working_model(self, model_name: str):
        """Đưa mô hình vừa thành công lên đầu danh sách để các request sau dùng ngay lập tức."""
        if model_name in self.models:
            self.models.remove(model_name)
            self.models.insert(0, model_name)
            self.current_model_idx = 0

    def _demote_failed_model(self, model_name: str):
        """Đẩy mô hình bị lỗi về cuối danh sách để không làm chậm các lượt gọi tiếp theo."""
        if model_name in self.models and len(self.models) > 1:
            self.models.remove(model_name)
            self.models.append(model_name)

    def get_active_model(self) -> str:
        if 0 <= self.current_model_idx < len(self.models):
            return self.models[self.current_model_idx]
        return self.models[0]

    def get_food_info(
        self,
        food_name: str,
    ) -> FoodInfoResponse:
        """
        Nhận tên món ăn và tra cứu thông tin chi tiết (ưu tiên cache trước).
        """
        food_name = food_name.strip()
        if not food_name:
            raise ValueError("Tên món ăn không được để trống.")

        cache_key = food_name.lower()
        if cache_key in _FOOD_INFO_CACHE:
            return _FOOD_INFO_CACHE[cache_key]

        prompt = f"""
Bạn là chuyên gia ẩm thực Việt Nam song ngữ (tiếng Việt và tiếng Anh).

Hãy cung cấp thông tin và công thức chế biến về món ăn sau:
{food_name}

Trả lời bằng JSON hợp lệ duy nhất với đúng cấu trúc song ngữ sau (không kèm văn bản khác ngoài JSON):
{{
    "food_name": "{food_name}",
    "description": "Mô tả tiếng Việt...",
    "origin": "Nguồn gốc, văn hóa tiếng Việt...",
    "ingredients": ["Nguyên liệu 1", "Nguyên liệu 2"],
    "taste": "Hương vị tiếng Việt...",
    "preparation": "Cách chế biến tiếng Việt...",
    "note": "Lưu ý tiếng Việt...",
    "en": {{
        "food_name": "English name of the dish",
        "description": "English description...",
        "origin": "English origin & cultural background...",
        "ingredients": ["English ingredient 1", "English ingredient 2"],
        "taste": "English flavor profile...",
        "preparation": "English preparation and cooking instructions...",
        "note": "English culinary tips/notes..."
    }}
}}

Yêu cầu:
- Phần tiếng Việt viết chuẩn ngữ pháp, súc tích.
- Phần "en" viết bằng tiếng Anh chuẩn, tự nhiên.
- ingredients và en.ingredients phải là danh sách các nguyên liệu chính.
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

                res = FoodInfoResponse(
                    food_name=data.get("food_name", food_name),
                    description=data.get("description", "Chưa có thông tin."),
                    origin=data.get("origin", "Chưa có thông tin."),
                    ingredients=data.get("ingredients", []),
                    taste=data.get("taste", "Chưa có thông tin."),
                    preparation=data.get("preparation", "Chưa có thông tin."),
                    note=data.get("note"),
                    model_used=model_name,
                    en=data.get("en"),
                )

                self._promote_working_model(model_name)
                _FOOD_INFO_CACHE[cache_key] = res
                print(f"[Gemini] Thanh cong voi model: {model_name}")
                return res

            except Exception as exc:
                err_str = str(exc)
                errors.append(f"[{model_name}]: {err_str[:100]}")
                self._demote_failed_model(model_name)
                print(f"[Gemini Rotate] Model '{model_name}' loi ({err_str[:50]}...). Chuyen model...")
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
        Nhận danh sách nhiều món ăn và tra cứu chi tiết.
        Tự động tận dụng Cache để trả về tức thì các món đã biết.
        """
        unique_names: List[str] = []
        seen = set()
        for name in food_names:
            clean = name.strip()
            if clean and clean.lower() not in seen:
                seen.add(clean.lower())
                unique_names.append(clean)

        if not unique_names:
            return []

        # Kiểm tra xem những món nào đã có trong Cache
        cached_results: List[FoodInfoResponse] = []
        uncached_names: List[str] = []
        for name in unique_names:
            key = name.lower()
            if key in _FOOD_INFO_CACHE:
                cached_results.append(_FOOD_INFO_CACHE[key])
            else:
                uncached_names.append(name)

        # Nếu tất cả món đều đã có trong cache: Trả về ngay lập tức (0ms)!
        if not uncached_names:
            return cached_results

        # Nếu chỉ có 1 món chưa có trong cache: Gọi get_food_info
        if len(uncached_names) == 1:
            try:
                new_info = self.get_food_info(uncached_names[0])
                cached_results.append(new_info)
                return cached_results
            except Exception:
                return cached_results

        # Xây dựng prompt cho các món chưa có trong cache
        list_str = "\n".join([f"{i+1}. {name}" for i, name in enumerate(uncached_names)])
        prompt = f"""
Bạn là chuyên gia ẩm thực Việt Nam song ngữ (tiếng Việt và tiếng Anh).

Hãy cung cấp thông tin chi tiết và công thức chế biến cho từng món ăn trong danh sách sau:
{list_str}

Trả lời bằng một mảng JSON (JSON array) hợp lệ duy nhất, mỗi phần tử tương ứng với một món ăn theo đúng cấu trúc song ngữ sau:
[
  {{
    "food_name": "Tên món ăn tiếng Việt",
    "description": "Mô tả tiếng Việt ngắn gọn...",
    "origin": "Nguồn gốc, văn hóa tiếng Việt...",
    "ingredients": ["Nguyên liệu tiếng Việt 1", "Nguyên liệu tiếng Việt 2"],
    "taste": "Hương vị tiếng Việt...",
    "preparation": "Cách chế biến tiếng Việt...",
    "note": "Lưu ý tiếng Việt...",
    "en": {{
      "food_name": "English name (Vietnamese name)",
      "description": "English concise description...",
      "origin": "English origin & culture...",
      "ingredients": ["English ingredient 1", "English ingredient 2"],
      "taste": "English flavor profile...",
      "preparation": "English preparation and cooking...",
      "note": "English notes/tips..."
    }}
  }}
]

Yêu cầu:
- Bắt buộc trả về đầy đủ thông tin cho tất cả các món ăn trong danh sách trên.
- Phần tiếng Việt viết chuẩn ngữ pháp, súc tích.
- Phần "en" viết bằng tiếng Anh chuẩn, tự nhiên.
- ingredients và en.ingredients phải là danh sách các nguyên liệu chính.
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

                new_results: List[FoodInfoResponse] = []
                for item in raw_data:
                    if isinstance(item, dict):
                        f_info = FoodInfoResponse(
                            food_name=item.get("food_name", "Món ăn"),
                            description=item.get("description", "Chưa có thông tin."),
                            origin=item.get("origin", "Chưa có thông tin."),
                            ingredients=item.get("ingredients", []),
                            taste=item.get("taste", "Chưa có thông tin."),
                            preparation=item.get("preparation", "Chưa có thông tin."),
                            note=item.get("note"),
                            model_used=model_name,
                            en=item.get("en"),
                        )
                        new_results.append(f_info)
                        _FOOD_INFO_CACHE[f_info.food_name.lower()] = f_info

                if new_results:
                    self._promote_working_model(model_name)
                    print(f"[Gemini] Thanh cong lay thong tin {len(new_results)} mon voi model: {model_name}")
                    return cached_results + new_results

            except Exception as exc:
                err_str = str(exc)
                errors.append(f"[{model_name}]: {err_str[:100]}")
                self._demote_failed_model(model_name)
                continue

        # Fallback tuần tự từng món nếu batch thất bại
        for name in uncached_names:
            try:
                info = self.get_food_info(name)
                cached_results.append(info)
            except Exception as e:
                print(f"[Gemini Fallback] Khong the lay thong tin cho '{name}': {e}")

        if cached_results:
            return cached_results

        error_summary = " | ".join(errors[-3:])
        raise RuntimeError(
            f"Không thể lấy thông tin món ăn từ các mô hình Gemini. Lỗi: {error_summary}"
        )
