from app.rag_chain import hybrid_search, run_rag
import re

# Câu chào hỏi/vô nghĩa → trả lời cố định không tốn token
GREETING_PATTERNS = [
    r"^(hi|hello|hey|chào|xin chào|alo|yo|hola)[\s!.]*$",
    r"^(ok|okay|oke|được|thanks|cảm ơn|thank you|ty)[\s!.]*$",
    r"^(test|testing|\d+|abc|xyz|asdf)[\s!.]*$",
    r"^.{1,3}$",  # quá ngắn < 3 ký tự
]

GREETING_RESPONSE = """Xin chào! 👋 Tôi là trợ lý AI của **TNU-AIQA**.

Tôi có thể giúp bạn tra cứu thông tin về:
- Thông tư về kiểm định chất lượng CTĐT
- Thông tư về kiểm định chất lượng CSGD
- Hệ thống bảo đảm chất lượng bên trong TNU
- Tiêu chuẩn, tiêu chí AUN-QA 
- ... 

Bạn muốn hỏi gì về kiểm định chất lượng giáo dục?"""

NO_INFO_MSG = (
    "Tôi không tìm thấy thông tin này trong tài liệu của TNU-AIQA. "
    "Vui lòng liên hệ Phòng Quản lý Chất lượng để được hỗ trợ."
)

# Câu hỏi không liên quan đến chủ đề
OFF_TOPIC_KEYWORDS = [
    "thời tiết", "bóng đá", "phim", "nhạc", "game", "nấu ăn",
    "học phí", "điểm thi", "thời khóa biểu", "lịch học",
    "weather", "football", "movie", "music", "food", "recipe"
]

OFF_TOPIC_RESPONSE = (
    "Xin lỗi, tôi chỉ hỗ trợ các câu hỏi liên quan đến "
    "**kiểm định chất lượng giáo dục** tại TNU. "
    "Bạn có thể hỏi về tiêu chuẩn AUN-QA, Thông tư 04, "
    "Thông tư 20 hoặc hệ thống IQA của TNU."
)


def is_greeting(question: str) -> bool:
    """Phát hiện câu chào hỏi hoặc vô nghĩa."""
    q = question.strip().lower()
    return any(re.match(p, q) for p in GREETING_PATTERNS)


def is_off_topic(question: str) -> bool:
    """Phát hiện câu hỏi không liên quan."""
    q = question.lower()
    return any(kw in q for kw in OFF_TOPIC_KEYWORDS)


def is_valid_question(question: str) -> bool:
    """Kiểm tra câu hỏi có đủ ý nghĩa không."""
    q = question.strip()
    # Quá ngắn
    if len(q) < 5:
        return False
    # Toàn ký tự đặc biệt
    if not re.search(r"[a-zA-ZÀ-ỹ]", q):
        return False
    return True


def run_agent(question: str) -> dict:
    question = question.strip()

    # Bước 1: Kiểm tra câu hỏi hợp lệ
    if not is_valid_question(question):
        return {
            "answer": GREETING_RESPONSE,
            "sources": [],
            "model_used": "none (filtered)"
        }

    # Bước 2: Phát hiện chào hỏi
    if is_greeting(question):
        return {
            "answer": GREETING_RESPONSE,
            "sources": [],
            "model_used": "none (greeting)"
        }

    # Bước 3: Phát hiện off-topic
    if is_off_topic(question):
        return {
            "answer": OFF_TOPIC_RESPONSE,
            "sources": [],
            "model_used": "none (off-topic)"
        }

    # Bước 4: Search context
    context, sources = hybrid_search(question)

    if not context.strip() or len(context.strip()) < 50:
        return {
            "answer": NO_INFO_MSG,
            "sources": [],
            "model_used": "none"
        }

    # Bước 5: Gọi LLM — chỉ khi thực sự cần
    answer, model_used = run_rag(question)

    return {
        "answer": answer,
        "sources": sources,
        "model_used": model_used
    }