import streamlit as st
import requests
import json

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="TNU-AIQA Chatbot",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 TNU-AIQA - Trợ lý Kiểm định Chất lượng")
st.caption("Hỗ trợ tra cứu thông tin về bảo đảm và kiểm định chất lượng giáo dục TNU")

# Sidebar
with st.sidebar:
    st.header("⚙️ Cấu hình")
    api_url = st.text_input("API URL", value=API_URL)
    
    st.divider()
    st.header("🔍 Test nhanh")
    quick_tests = [
        "Thông tư 04 có bao nhiêu tiêu chuẩn?",
        "Tiêu chuẩn 1 có bao nhiêu tiêu chí?",
        "Thông tư 20/2026 có hiệu lực từ ngày nào?",
        "AUN-QA là gì?",
        "Hệ thống IQA của TNU gồm những cấu phần nào?",
        "Học phí năm 2026 là bao nhiêu?",
    ]
    for q in quick_tests:
        if st.button(q, use_container_width=True):
            st.session_state.quick_question = q

    st.divider()
    if st.button("🗑️ Xóa lịch sử", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Kiểm tra API health
    st.divider()
    st.header("📡 Trạng thái API")
    try:
        r = requests.get(f"{api_url}/", timeout=3)
        if r.status_code == 200:
            st.success("✅ API đang chạy")
        else:
            st.error("❌ API lỗi")
    except:
        st.error("❌ Không kết nối được API")

# Khởi tạo chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            if msg["sources"]:
                with st.expander("📚 Nguồn tham khảo"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.caption(f"{i}. {src}")
            if "model_used" in msg:
                st.caption(f"🤖 Model: `{msg['model_used']}`")

# Xử lý quick question từ sidebar
if "quick_question" in st.session_state:
    question = st.session_state.quick_question
    del st.session_state.quick_question
else:
    question = None

# Chat input
prompt = st.chat_input("Nhập câu hỏi về kiểm định chất lượng...") or question

if prompt:
    # Hiển thị câu hỏi
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gọi API
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm..."):
            try:
                response = requests.post(
                    f"{api_url}/chat",
                    json={"message": prompt, "session_id": "streamlit"},
                    timeout=30
                )
                data = response.json()
                answer = data.get("answer", "Không có phản hồi")
                sources = data.get("sources", [])
                model_used = data.get("model_used", "unknown")

                st.markdown(answer)

                if sources:
                    with st.expander("📚 Nguồn tham khảo"):
                        for i, src in enumerate(sources, 1):
                            st.caption(f"{i}. {src}")

                st.caption(f"🤖 Model: `{model_used}`")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "model_used": model_used
                })

            except requests.exceptions.ConnectionError:
                st.error("❌ Không kết nối được API. Hãy chắc chắn uvicorn đang chạy!")
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")