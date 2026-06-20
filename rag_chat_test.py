import os
import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# ===== LLMクライアントの設定 =====
# .envのLLM_PROVIDERでollamaとopenaiを切り替える
provider = os.getenv("LLM_PROVIDER", "ollama")

if provider == "openai":
    llm_model = os.getenv("OPENAI_MODEL", "gpt-5.1")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    # OllamaはOpenAI互換APIを持つのでOpenAIクライアントがそのまま使える
    llm_model = os.getenv("OLLAMA_MODEL", "gemma4:e2b-it-qat")
    client = OpenAI(
        api_key="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    )

# ===== ChromaDB（ベクトルデータベース）の設定 =====
# ベクトルDBはテキストを数値（ベクトル）に変換して保存し、意味的な類似検索ができるDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")


# embeddingモデルの設定（日本語がわかる組み込み用モデルを指定）
# @st.cache_resource でサーバー起動時の1回だけロードし、以降はキャッシュを再利用する
@st.cache_resource
def load_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )


ef = load_embedding_function()

# ドキュメント登録時に自動でembeddingを生成してくれるように、コレクションにembedding_functionを指定して作成
if "collection" not in st.session_state:
    st.session_state.collection = chroma_client.get_or_create_collection(
        name="local_docs", embedding_function=ef
    )


# PDFファイルを読み込む関数
def load_pdf(file):
    # pypdfでPDFを読み込み、全ページのテキストを結合して返す
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ===== テキスト分割 =====
# chunk_size=400
# chunk_overlap=100：前後のチャンクと100文字重複させることで文脈の切れ目をなくす
def split_text(text):
    chunk_size = 400
    overlap = 100
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks


PAGE_TITLE = "Rag Test Chat (PDF)"
st.set_page_config(page_title=PAGE_TITLE)

system_prompt = (
    # "以下のルールに従って回答してください。"
    # "1. 提供された資料に記載がある内容は、資料を根拠に回答する"
    # "2. 資料に記載がない内容は「資料には記載がありません」と一言だけ述べ、一般的な知識で補足できる場合はその旨を添えて回答する"
    # "3. 完全に不明な場合のみ「わかりません」と答える"
    "日本語で回答してください。"
)

# サイドバーに現在使用中のプロバイダ・モデルを表示
st.sidebar.caption(f"LLM: [{provider}] {llm_model}")

# PDFファイルのアップロード
uploaded_files = st.sidebar.file_uploader(
    "PDFファイルをアップロード", type=["pdf"], accept_multiple_files=True
)

# ===== インデックス作成（RAGの「R」= Retrieval の準備） =====
# アップロード済みファイル名を記録しておき、新しいファイルだけインデックスを作成する
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_files]

if new_files:
    with st.sidebar.status("インデックス作成中..."):
        for file in new_files:
            chunks = split_text(load_pdf(file))  # PDFからテキスト抽出・分割
            st.session_state.collection.add(
                documents=chunks,  # テキストを登録（embeddingは自動生成）
                ids=[
                    f"{file.name}_{i}" for i in range(len(chunks))
                ],  # 重複しないIDが必要
            )
            st.session_state.indexed_files.add(file.name)
    st.sidebar.success("インデックス作成完了")

# タイトル
st.title(PAGE_TITLE)

# 会話の履歴を保管
if "messages" not in st.session_state:
    st.session_state.messages = []

# 会話の履歴をリセットするボタン
if st.sidebar.button("会話をリセット"):
    st.session_state.messages = []

# 会話の履歴を表示
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])


prompt = st.chat_input("メッセージを入力")

if prompt:

    # ユーザーのプロンプトを表示
    with st.chat_message("user"):
        st.write(prompt)

    # ===== RAG検索（Retrieval） =====
    # ユーザーの質問をembeddingに変換し、DBから意味的に近いチャンクを取得する
    # query_textsを渡すとChromaDBが自動でembeddingを生成して検索してくれる
    n = min(2, st.session_state.collection.count())
    if n > 0:
        results = st.session_state.collection.query(
            query_texts=[prompt], n_results=n  # 上位2件を取得
        )
    else:
        results = {"documents": []}

    # ===== Augmentation（プロンプト拡張） =====
    # 取得したチャンクをプロンプトに埋め込み、LLMへの指示として渡す
    # これによりLLMは自分の学習データだけでなく、提供したドキュメントを根拠に回答できる
    if results["documents"]:
        context_text = "\n".join(results["documents"][0])
        user_message = f"以下は関連ドキュメントの抜粋です。\n{context_text}\nこの情報を参考に以下の質問に答えてください。\n{prompt}"
    else:
        user_message = (
            prompt  # 関連ドキュメントが見つからなければ素の質問をそのまま渡す
        )

    # 表示用には元の質問をそのまま保存する
    st.session_state.messages.append({"role": "user", "content": prompt})

    # LLMに渡すメッセージは、履歴の最後だけRAG拡張プロンプトに差し替える
    messages = (
        [{"role": "system", "content": system_prompt}]
        + st.session_state.messages[:-1]
        + [{"role": "user", "content": user_message}]
    )

    # ===== Generation（生成） =====
    # 拡張したプロンプトをLLMに渡して回答を生成する（stream=Trueで逐次表示）
    with st.chat_message("assistant"):
        placeholder = st.empty()
        response = ""
        for chunk in client.chat.completions.create(
            model=llm_model, messages=messages, temperature=0.3, stream=True
        ):
            response += chunk.choices[0].delta.content or ""
            placeholder.write(response)

    # 会話の履歴を保存
    st.session_state.messages.append({"role": "assistant", "content": response})
