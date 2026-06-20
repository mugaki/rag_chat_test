# RAG Chat Test(PDF)

PDF をアップロードして、その内容をもとに質問できる RAG チャットアプリです。

## 機能

- PDF のアップロードとベクトルインデックス作成
- ChromaDB による意味的類似検索（RAG）
- Ollama / OpenAI の切り替え対応
- ストリーミング応答

## セットアップ

```cmd
# 依存パッケージのインストール
uv sync

# 環境変数の設定
copy .env.example .env
# .env を編集してプロバイダ・APIキーを設定
```

## 起動

```cmd
uv run streamlit run rag_chat_test.py --server.fileWatcherType none
```

`sentence-transformers` + `transformers` の組み合わせでは、Streamlit のファイル監視が `torchvision` の未導入モジュールを走査して大量の `ModuleNotFoundError` ログを出すことがあります。`--server.fileWatcherType none` で監視を無効化すると回避できます。

## 環境変数

| 変数名 | 説明 | デフォルト |
|---|---|---|
| `LLM_PROVIDER` | 使用するプロバイダ (`ollama` or `openai`) | `ollama` |
| `OLLAMA_BASE_URL` | Ollama の API エンドポイント | `http://localhost:11434/v1` |
| `OLLAMA_MODEL` | Ollama で使用するモデル名 | `gemma4:e2b-it-qat` |
| `OPENAI_API_KEY` | OpenAI の API キー | — |
| `OPENAI_MODEL` | OpenAI で使用するモデル名 | `gpt-5.1` |
