"""Ollama API クライアント"""
import requests
import json
import uuid
import logging
from typing import Dict, List, Tuple, Optional

try:
    from config.settings import OLLAMA_BASE_URL
except ImportError:
    import os
    # configモジュールが存在しない場合のフォールバック値（環境変数を優先）
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


logger = logging.getLogger(__name__)


def list_models():
    """Ollamaに登録されているモデル一覧を取得"""
    url = f"{OLLAMA_BASE_URL}/api/tags"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return [m["name"] for m in data.get("models", [])]


def list_loaded_models() -> list:
    """現在 GPU/メモリにロードされている Ollama モデル一覧を取得する。

    Returns:
        list[dict]: ロード中のモデル情報リスト。各要素に 'name' キーを含む。
    """
    url = f"{OLLAMA_BASE_URL}/api/ps"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json().get("models", [])
    except Exception as e:
        logger.warning(f"⚠️ ロード済みモデル取得失敗: {e}")
        return []


def unload_model(model_name: str) -> bool:
    """指定モデルを Ollama から即時アンロード (VRAM 解放) する。

    keep_alive=0 を指定して /api/generate を呼ぶことでモデルをメモリから解放する。
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {"model": model_name, "keep_alive": 0}
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        logger.info(f"🗑️ モデルをアンロード: {model_name}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ モデルアンロード失敗 ({model_name}): {e}")
        return False


def unload_all_models() -> int:
    """現在ロードされている全 Ollama モデルをアンロードして VRAM を解放する。

    Returns:
        int: アンロードに成功したモデル数
    """
    loaded = list_loaded_models()
    if not loaded:
        logger.info("ℹ️ ロード済みの Ollama モデルはありません")
        return 0

    count = 0
    for m in loaded:
        name = m.get("name", "") if isinstance(m, dict) else str(m)
        if name and unload_model(name):
            count += 1

    logger.info(f"🗑️ {count}/{len(loaded)} 件の Ollama モデルをアンロードしました")
    return count


class SessionManager:
    """
    簡易セッションマネージャ — メモリ内で会話履歴を保持します。
    参考URLも最後に取り出せるよう、直近検索URLを保持します。
    max_historyを指定することで、user/assistantメッセージの保持数を制限できます。
    """

    def __init__(self, max_history: int = 20):
        # sessions: { session_id: {"messages": [{...}], "last_urls": [str, ...] } }
        self.sessions: Dict[str, Dict] = {}
        self.max_history = max_history  # user/assistantメッセージの最大保持数

    def create(self, session_id: str = None) -> str:
        sid = session_id or str(uuid.uuid4())
        self.sessions[sid] = {"messages": [], "last_urls": []}
        return sid

    def add_user(self, session_id: str, text: str):
        self.sessions[session_id]["messages"].append({"role": "user", "content": text})
        self._trim_history(session_id)

    def add_assistant(self, session_id: str, text: str):
        self.sessions[session_id]["messages"].append({"role": "assistant", "content": text})
        self._trim_history(session_id)

    def add_system(self, session_id: str, text: str):
        """セッションに system ロールのメッセージを追加します。"""
        self.sessions[session_id]["messages"].append({"role": "system", "content": text})

    def clear(self, session_id: str):
        self.sessions[session_id]["messages"] = []
        self.sessions[session_id]["last_urls"] = []

    def history(self, session_id: str) -> List[Dict]:
        return list(self.sessions[session_id]["messages"])

    def set_last_urls(self, session_id: str, urls: List[str]):
        self.sessions[session_id]["last_urls"] = list(urls)

    def get_last_urls(self, session_id: str) -> List[str]:
        return list(self.sessions[session_id].get("last_urls", []))

    def remove_system_prefixes(self, session_id: str, prefixes: List[str]) -> None:
        """指定プレフィックスを含むsystemメッセージを削除（主に過去の検索結果をリセット）。"""
        msgs = self.sessions.get(session_id, {}).get("messages", [])
        if not msgs:
            return
        filtered = []
        for m in msgs:
            if m.get("role") != "system":
                filtered.append(m)
                continue
            content = m.get("content", "")
            if any(content.startswith(p) for p in prefixes):
                continue
            filtered.append(m)
        self.sessions[session_id]["messages"] = filtered

    def clear_all_system(self, session_id: str) -> int:
        """セッション内の全 system メッセージを削除する。

        タスク切り替え時に前タスクで注入されたコンテキスト（天気、検索結果、
        時刻など）をリセットして LLM の混乱を防ぐ。

        Returns:
            削除した system メッセージ数
        """
        msgs = self.sessions.get(session_id, {}).get("messages", [])
        if not msgs:
            return 0
        before = len(msgs)
        filtered = [m for m in msgs if m.get("role") != "system"]
        self.sessions[session_id]["messages"] = filtered
        removed = before - len(filtered)
        return removed

    def _trim_history(self, session_id: str) -> None:
        """会話履歴を max_history に制限します。systemメッセージは保持し、user/assistantのみを制限対象とします。"""
        msgs = self.sessions.get(session_id, {}).get("messages", [])
        if not msgs:
            return
        
        # systemメッセージと、user/assistantメッセージを分離
        system_msgs = [m for m in msgs if m.get("role") == "system"]
        conversation_msgs = [m for m in msgs if m.get("role") in ("user", "assistant")]
        
        # user/assistantメッセージが制限を超えている場合、古いものから削除
        if len(conversation_msgs) > self.max_history:
            conversation_msgs = conversation_msgs[-self.max_history:]
        
        # systemメッセージを先頭に、会話履歴を後に再構築
        self.sessions[session_id]["messages"] = system_msgs + conversation_msgs

    def build_prompt(self, session_id: str) -> str:
        """
        会話履歴を連結して、モデルへ投げるプレーンテキストのプロンプトを作成します。
        参考URLはプロンプトに混ぜません（システム的に別返却するため）。
        """
        parts: List[str] = []
        for m in self.sessions[session_id]["messages"]:
            if m["role"] == "system":
                parts.append(f"System: {m['content']}")
        for m in self.sessions[session_id]["messages"]:
            if m["role"] == "user":
                parts.append(f"User: {m['content']}")
            elif m["role"] == "assistant":
                parts.append(f"Assistant: {m['content']}")
        parts.append("Assistant:")
        return "\n".join(parts)


def stream_chat(session_mgr: SessionManager, session_id: str, model: str, chunk_callback=None) -> str:
    """指定セッションの会話履歴を元にモデルに問い合わせ、ストリーミングで応答を受け取り履歴へ追加し、応答テキストを返す。"""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    headers = {"Content-Type": "application/json"}
    prompt = session_mgr.build_prompt(session_id)
    payload = {"model": model, "prompt": prompt, "stream": True}

    # LLMに渡される最終データを表示
    print("\n" + "="*80)
    print("🤖 LLMに渡される最終プロンプト:")
    print("="*80)
    print(prompt)
    print("="*80 + "\n")

    assistant_chunks: List[str] = []

    with requests.post(url, headers=headers, data=json.dumps(payload), stream=True) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    chunk = data["response"]
                    assistant_chunks.append(chunk)
                    if chunk_callback:
                        chunk_callback(chunk)
                if data.get("done", False):
                    break

    assistant_text = "".join(assistant_chunks).strip()
    if assistant_text:
        session_mgr.add_assistant(session_id, assistant_text)
    return assistant_text

def generate_with_image(model: str, prompt: str, base64_images: List[str], skip_thinking: bool = True) -> str:
    """VLMモデル向けに画像を添付してプロンプトを実行する
    
    Args:
        model: 使用するVLMモデル名
        prompt: プロンプトテキスト
        base64_images: Base64エンコードされた画像リスト
        skip_thinking: Thinkingモデルの思考プロセスをスキップするか（デフォルト: True）
    
    Returns:
        生成された応答テキスト
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "images": base64_images,
        "stream": False
    }
    
    # Thinkingモデルの場合、思考プロセスをスキップ
    if skip_thinking:
        payload["num_predict"] = -2  # このフラグでthinkingをスキップする指示

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        response = data.get("response", "").strip()
        
        # 応答からthinkingタグを除外（念のため）
        if skip_thinking and "<think>" in response:
            response = _remove_thinking_tags(response)
        
        return response
    except Exception as e:
        logger.error(f"Ollama API画像生成エラー: {e}")
        return ""


def _remove_thinking_tags(text: str) -> str:
    """テキストからthinkingタグを除外する
    
    Args:
        text: 処理対象のテキスト
        
    Returns:
        thinkingタグが除外されたテキスト
    """
    import re
    # <think>......</think> のブロックを除外
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()
