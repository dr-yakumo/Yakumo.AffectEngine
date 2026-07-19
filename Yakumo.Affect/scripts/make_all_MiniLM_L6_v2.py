import os
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model_name = "sentence-transformers/all-MiniLM-L6-v2"

# install.ps1 から呼ばれる場合は MINILM_OUTPUT_DIR 環境変数でパスが渡される
# 手動実行時はスクリプト配置ディレクトリからの相対パス (libs/models/...) がデフォルト
output_dir = os.environ.get(
    "MINILM_OUTPUT_DIR",
    os.path.join(os.path.dirname(__file__), "libs", "models", "all-MiniLM-L6-v2")
)
output_dir = os.path.abspath(output_dir)
os.makedirs(output_dir, exist_ok=True)
print(f"出力先: {output_dir}")

# ONNX に変換
model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
model.save_pretrained(output_dir)

# Tokenizer を保存
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(output_dir)

print("all-MiniLM-L6-v2 のエクスポートが完了しました")