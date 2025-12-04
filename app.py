import os
import streamlit as st
from dotenv import load_dotenv

# --------- LangChain 関連 ----------
try:
    # 新しいバージョンの LangChain
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
except ImportError:
    # 古いバージョンの場合
    from langchain.schema import HumanMessage, AIMessage, SystemMessage

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI

# --------- 環境変数の読み込み (.env) ----------
load_dotenv()   # OPENAI_API_KEY を読み込む
# ローカル(.env) か Cloud(st.secrets) のどちらかから OPENAI_API_KEY を読む
os.environ["OPENAI_API_KEY"] = os.getenv(
    "OPENAI_API_KEY",
    st.secrets.get("OPENAI_API_KEY", "")
)

# --------- LLM 呼び出し用の関数 ----------
def call_llm(user_text: str, expert_type: str) -> str:
    """
    引数:
        user_text: 入力テキスト
        expert_type: ラジオボタンの選択値（専門家の種類）
    戻り値:
        LLM からの回答テキスト
    """

    # 専門家の種類に応じてシステムメッセージを変更
    if expert_type == "キャリアの専門家":
        system_content = "あなたはキャリア相談の専門家です。優しく、わかりやすく助言してください。"
    elif expert_type == "ワークライフバランスの専門家":
        system_content = "あなたはワークライフバランスの専門家です。生活リズムや働き方の観点でアドバイスしてください。"
    else:
        system_content = "あなたは丁寧に回答するアシスタントです。"

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_text)
    ]

    # モデル作成（gpt-4o-mini）
    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # LLM 呼び出し
    response = chat.invoke(messages)
    return response.content


# ===== Streamlit アプリ部分 =====

st.set_page_config(page_title="LLM相談アプリ", layout="wide")

st.title("💬 LLM相談アプリ（キャリア／ワークライフバランス）")

st.markdown("""
### アプリの概要
このアプリは、OpenAI の LLM（大規模言語モデル）と LangChain を使った相談用アプリです。

1. 相談したい「専門家のタイプ」をラジオボタンで選びます  
2. 下の入力欄に相談内容を入力します  
3. 「LLMに相談する」ボタンを押すと、専門家の視点でアドバイスが表示されます  

---

※このアプリを利用するには、`.env` ファイルに **OPENAI_API_KEY** を設定しておく必要があります。  
""")

# ---- APIキーが存在するかチェック ----
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ `OPENAI_API_KEY` が環境変数として設定されていません。.env ファイルに OPENAI_API_KEY=... を記述して、アプリを再起動してください。")
else:
    st.success("✅ OpenAI APIキーが正常に読み込まれています。")

# ---- 専門家タイプ ----
expert = st.radio(
    "相談したい専門家の種類を選んでください：",
    ["キャリアの専門家", "ワークライフバランスの専門家"],
    horizontal=True
)

# ---- 入力フォーム ----
user_input = st.text_area("相談内容を入力してください（日本語OK）", height=200)

# ---- 実行ボタン ----
if st.button("LLMに相談する"):
    if user_input.strip() == "":
        st.warning("相談内容を入力してください。")
    else:
        st.write("### 🔍 回答：")
        answer = call_llm(user_input, expert)
        st.success(answer)
