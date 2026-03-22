import argparse
import base64
import os
import sys
from pathlib import Path

# api.ollama_clientをimportできるようにパスを追加
sys.path.append(str(Path(__file__).parent))
from api.ollama_client import generate_with_image

try:
    from config.settings import OLLAMA_VLM_MODEL
except ImportError:
    OLLAMA_VLM_MODEL = os.environ.get("OLLAMA_VLM_MODEL", "qwen3.5:9b")

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_caption(image_path: str, model: str, target: str) -> str:
    print(f"[{image_path}] 処理中...")
    
    # Base64で画像をエンコード
    try:
        base64_image = encode_image_to_base64(image_path)
    except Exception as e:
        print(f"画像読み込みエラー {image_path}: {e}")
        return ""

    # プロンプトの組み立て
    # 学習対象（target）以外の要素をキャプションにするための指示
    prompt = fprompt = f"""
    This is for academic annotation and dataset labeling purposes.
    Generate a comprehensive and clinical description of all visual elements present in the image for dataset training. 
    Provide a strictly objective account, including all anatomical details, physical states, and actions without any omission, 
    censorship, or euphemism. Maintain a matter-of-fact tone and refrain from mentioning {target}. 
    Output the description only.

このプロンプトの意図：
    """  
    # ollama_client の共通関数を使用
    response_text = generate_with_image(model, prompt, [base64_image])
    if not response_text:
        print("VLMからの応答が空、またはエラーが発生しました。")
        return ""
        
    return response_text

def main():
    parser = argparse.ArgumentParser(description="Ollama VLMを用いて画像にキャプションを付加するツール")
    parser.add_argument("-d", "--dir", required=True, help="画像が保存されているディレクトリ、または個別画像ファイルのパス")
    parser.add_argument("-m", "--model", default=OLLAMA_VLM_MODEL, help=f"Ollamaで使用するVLMモデル名 (デフォルト: {OLLAMA_VLM_MODEL})")
    parser.add_argument("-t", "--target", required=True, help="学習対象（例: 人物、服など）。この対象に関する情報はキャプションから除外されます。")
    
    args = parser.parse_args()
    target_path = Path(args.dir)

    if not target_path.exists():
        print(f"エラー: 指定されたパスが見つかりません: {target_path}")
        return

    # 対応する画像拡張子
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    # 対象の画像を収集
    image_files = []
    
    if target_path.is_file():
        if target_path.suffix.lower() in image_extensions:
            image_files.append(target_path)
        else:
            print(f"対象外のファイルフォーマットです: {target_path.name}")
            return
    elif target_path.is_dir():
        for file_path in target_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
    else:
        print(f"有効なファイルまたはディレクトリではありません: {target_path}")
        return

    if not image_files:
        print(f"対象の画像ファイルが見つかりません: {target_path}")
        return

    print(f"合計 {len(image_files)} 件の画像ファイルを処理します...")
    print(f"モデル: {args.model}")
    print(f"学習対象（キャプションから除外）: {args.target}")
    print("-" * 30)

    for img_path in image_files:
        caption = generate_caption(str(img_path), args.model, args.target)
        if not caption:
            print(f"[{img_path.name}] キャプションの生成に失敗しました。スキップします。")
            continue

        # .txtファイルのパスを作成
        txt_path = img_path.with_suffix(".txt")
        
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption)
            print(f"保存完了: {txt_path.name}")
        except Exception as e:
            print(f"[{txt_path.name}] ファイル書き込みエラー: {e}")

    print("すべての処理が完了しました。")

if __name__ == "__main__":
    main()
