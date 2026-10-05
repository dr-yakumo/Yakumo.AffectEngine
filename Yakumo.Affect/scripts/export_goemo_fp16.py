"""goemo-roberta-base の ONNX モデルを fp16 に変換する。

DirectML (GPU) 利用時の高速化を狙ったオプション処理。
`install.ps1 -Gpu` から呼ばれる。CPU 実行なら不要。

実行:
    python scripts/export_goemo_fp16.py
    python scripts/export_goemo_fp16.py --model-dir <path>

モデルの場所は、指定が無ければ以下の順に探す:
    1. --model-dir 引数
    2. 環境変数 YAKUMO_GOEMO_MODEL_DIR
    3. スクリプトからの相対 ../libs/models/goemo-roberta-base
       （リリース構成では scripts/ の隣が libs/ になる）
    4. カレントディレクトリからの libs/models/goemo-roberta-base

!!! 生成した model_fp16.onnx を使うには affect.config の設定が必要:
       [nli_model] ModelPrecision = fp16
       [nli_model] UseDirectML    = true

!!! 速度はほぼ変わらない。
   実測（Radeon RX 6800 / DirectML / 700文・2026-09-07）:

       条件           所要      速度比
       fp32 + CPU    403.9 秒   1.00x
       fp32 + GPU    394.9 秒   1.02x
       fp16 + GPU    395.9 秒   1.02x

   律速は ONNX 推論ではなく翻訳（Python 側 CPU）。
   推論を速くしても、全体の所要時間はほとんど変わらない。
   fp16 の利点は、ファイルサイズとメモリが半分（476MB → 238MB）になること。

数値の健全性は fp32 と突き合わせて確認済み（同日）:
   NaN/Inf なし / argmax 全一致 / ロジットの最大絶対差 0.002
   同じ 700文での Macro-F1 も fp32 と完全一致。
   入出力仕様も fp32 と同一なので、呼び出し側の変更は不要。
"""
from __future__ import annotations

import argparse
import os
import sys

CANDIDATE_SUFFIX = os.path.join("libs", "models", "goemo-roberta-base")


def resolve_model_dir(explicit: str | None) -> str:
    """モデルディレクトリを探す。見つからなければ終了する。"""
    candidates: list[str] = []

    if explicit:
        candidates.append(explicit)

    from_env = os.environ.get("YAKUMO_GOEMO_MODEL_DIR")
    if from_env:
        candidates.append(from_env)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # リリース構成: <install>/scripts/export_goemo_fp16.py → <install>/libs/models/...
    candidates.append(os.path.join(os.path.dirname(script_dir), CANDIDATE_SUFFIX))
    # モデルディレクトリ内に直接置かれている場合（従来の配置）
    candidates.append(script_dir)
    # カレントから
    candidates.append(os.path.join(os.getcwd(), CANDIDATE_SUFFIX))

    for path in candidates:
        if os.path.isfile(os.path.join(path, "model.onnx")):
            return path

    print("[ERROR] model.onnx が見つかりませんでした。探した場所:", file=sys.stderr)
    for path in candidates:
        print(f"          {path}", file=sys.stderr)
    print("        --model-dir で明示するか、先にモデルを取得してください。", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=None,
                        help="goemo-roberta-base のディレクトリ")
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_dir)

    fp32_path = os.path.join(model_dir, "model.onnx")
    fp16_path = os.path.join(model_dir, "model_fp16.onnx")

    print(f"[IN]  {fp32_path}")

    try:
        import onnx
    except ImportError as e:
        print(f"[ERROR] 依存パッケージがありません: {e}", file=sys.stderr)
        print("        pip install onnx onnxruntime", file=sys.stderr)
        return 1

    # !!! onnxconverter_common ではなく onnxruntime 側の実装を使うこと。
    #
    #   onnxconverter_common.float16.convert_float_to_float16 は
    #   「全テンソルを fp16 にしてから境界に Cast を巻く」方式で、
    #   Cast の to 属性が float のまま残り、ロード時に型エラーになる:
    #       Type (tensor(float16)) of output arg (.../attention/self/Cast_output_0)
    #       does not match expected type (tensor(float))
    #   回避しようと op_block_list に Cast を足すと境界が爆発し、
    #   RoBERTa-base で 1時間15分・メモリ 8.9GB でも完了しなかった（2026-09-07 実測）。
    #
    #   onnxruntime.transformers.float16 は既定の block list のままで
    #   整合の取れたグラフを作る。同じモデルで 4.8 秒。
    try:
        from onnxruntime.transformers.float16 import convert_float_to_float16
    except ImportError as e:
        print(f"[ERROR] onnxruntime が必要です: {e}", file=sys.stderr)
        print("        pip install onnxruntime", file=sys.stderr)
        return 1

    model_fp32 = onnx.load(fp32_path)

    # keep_io_types=True   : 入出力は float32 のまま（呼び出し側の変更が不要）
    # disable_shape_infer=True: 形状推論を飛ばす。多時間ハングの既知回避策
    model_fp16 = convert_float_to_float16(
        model_fp32, keep_io_types=True, disable_shape_infer=True)

    onnx.save(model_fp16, fp16_path)

    size_mb = os.path.getsize(fp16_path) / 1024 / 1024
    print(f"[OUT] {fp16_path}")
    print(f"[OK]  ファイルサイズ: {size_mb:.1f} MB")
    print()
    print("使用するには affect.config を以下に設定してください:")
    print("    [nli_model] ModelPrecision = fp16")
    print("    [nli_model] UseDirectML    = true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
