# -*- coding: utf-8 -*-
"""goemo-roberta-base を「リビジョンを固定して」取得し、そのローカルパスを返す。

install.ps1 の Step 4 から呼ばれる。標準出力の最終行がローカルパスになる。

    python scripts/fetch_goemo_model.py 58b6c5b44a7a12093f782442969019c7e2982299

────────────────────────────────────────────────────────────────
なぜこのスクリプトが必要か
────────────────────────────────────────────────────────────────

公表スコアは特定のリビジョンの重みで測定している。上流で main が更新されると、
新しくインストールした環境だけが別のモデルになってしまう。
そのためリビジョンを固定して取得する。

ただし optimum の ONNX exporter にはリビジョンを指定する手段が無い。

  ✗ `--revision`
      存在しない引数。optimum 1.27.0 / 2.1.0 のどちらにも無く、
      `unrecognized arguments` でエラーになる。

  ✗ `--model-kwargs '{"revision": "..."}'`
      引数としては受け付けるが無視される。
      存在しないリビジョン（0000...）を渡しても export が成功するため、
      固定として機能していないことが分かる。

  ✓ `snapshot_download(revision=...)` で先に取得し、そのローカルパスを
      exporter の `--model` に渡す
      存在しないリビジョンでは RevisionNotFoundError (404) になるので、
      リビジョンが効いていることを確認できる。

そのため「取得」と「変換」を分け、取得側でリビジョンを固定している。

リビジョンは install.ps1 の $GOEMO_REVISION で指定する。
値を変えるとモデルの重みが変わり、スコアも公表値と一致しなくなる。
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ID = "SamLowe/roberta-base-go_emotions"

# ONNX へのエクスポートに必要なものだけ取る。
# 重みは safetensors か bin のどちらか一方しか無いリポジトリもあるため両方許可する。
ALLOW_PATTERNS = ["*.json", "*.txt", "*.safetensors", "*.bin", "*.model"]


def main(argv: list[str]) -> int:
    if len(argv) < 2 or not argv[1].strip():
        print("[ERROR] リビジョンが指定されていません。", file=sys.stderr)
        print("        使い方: python fetch_goemo_model.py <revision>", file=sys.stderr)
        return 1

    revision = argv[1].strip()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] huggingface_hub が導入されていません。", file=sys.stderr)
        print("        pip install -r requirements.txt を先に実行してください。", file=sys.stderr)
        return 1

    print(f"[FETCH] {REPO_ID}", file=sys.stderr)
    print(f"[FETCH] revision = {revision}", file=sys.stderr)

    try:
        local_path = snapshot_download(
            repo_id=REPO_ID,
            revision=revision,
            allow_patterns=ALLOW_PATTERNS,
        )
    except Exception as e:
        print(f"[ERROR] 取得に失敗しました: {type(e).__name__}", file=sys.stderr)
        print(f"        {e}", file=sys.stderr)
        print("", file=sys.stderr)
        print("        リビジョンが存在しない場合は RevisionNotFoundError になります。", file=sys.stderr)
        print("        install.ps1 の $GOEMO_REVISION を確認してください。", file=sys.stderr)
        return 1

    # 呼び出し側（install.ps1）はこの最終行をパスとして受け取る。
    # 進捗や警告はすべて stderr に出しているので、stdout はこの1行だけ。
    print(local_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
