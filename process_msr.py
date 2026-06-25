import argparse
import os
import sys
import time
from datetime import datetime
import cv2
import numpy as np

'''
実行
python process_msr.py ＜入力フォルダ＞
'''

def single_scale_retinex_fast(img, sigma):
    """リサイズを利用して高速にSSRを計算する。"""
    retinex_input = np.log(img + 1.0)
    scale = 4
    h, w = img.shape[:2]
    sw, sh = max(1, w // scale), max(1, h // scale)
    small_img = cv2.resize(img, (sw, sh))
    small_blurred = cv2.GaussianBlur(small_img, (0, 0), sigma / scale)
    blurred = cv2.resize(small_blurred, (w, h))
    retinex_blurred = np.log(blurred + 1.0)
    return retinex_input - retinex_blurred

def multi_scale_retinex(img, sigmas, weights):
    """指定された複数のスケールでMSRを計算する。

    cv2.resize / cv2.GaussianBlur / np.log はいずれも多チャンネル配列を
    チャンネル独立に処理するため、3次元配列をそのまま渡してよい
    （チャンネルごとにループした場合と数値的に同一の結果になる）。
    """
    img = img.astype(np.float64)
    msr_result = np.zeros_like(img)
    for s, w in zip(sigmas, weights):
        msr_result += w * single_scale_retinex_fast(img, s)
    return msr_result

def apply_mapping(msr_res, gain, offset):
    """計算結果を8ビット（0-255）の範囲にマッピングする。"""
    output = (gain * msr_res) + offset
    return np.clip(output, 0, 255).astype(np.uint8)

def save_msr_parameters(output_root, script_name, sigmas, gain, offset):
    """
    使用したパラメータを 【スクリプト名】_params.txt として保存する。
    """
    # 拡張子を除いたスクリプト名を取得
    base_script_name = os.path.splitext(os.path.basename(script_name))[0]
    param_path = os.path.join(output_root, f"{base_script_name}_params.txt")
    
    with open(param_path, "w", encoding="utf-8") as f:
        f.write("=== MSR Processing Parameters ===\n")
        f.write(f"Date:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sigmas:  {sigmas}\n")
        f.write(f"Gain:    {gain}\n")
        f.write(f"Offset:  {offset}\n")
        f.write("=================================\n")

def process_recursive_with_progress(input_root, output_root, sigmas, gain, offset):
    """画像をソートして取得し、進捗を表示しながら処理を行う。"""
    valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    weights = [1.0 / len(sigmas)] * len(sigmas)

    all_tasks = []
    for root, dirs, files in os.walk(input_root):
        for f in files:
            if f.lower().endswith(valid_exts):
                all_tasks.append(os.path.join(root, f))
    
    all_tasks.sort()
    total_count = len(all_tasks)

    if total_count == 0:
        print("処理対象の画像が見つかりませんでした。")
        return

    print(f"合計 {total_count} 枚の画像を処理します。")

    for i, in_path in enumerate(all_tasks):
        start_time = time.time()
        rel_path = os.path.relpath(in_path, input_root)
        out_path = os.path.join(output_root, rel_path)
        
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)

        current_num = i + 1
        print(f"[{current_num}/{total_count}] Processing: {rel_path} ... ", end="", flush=True)

        img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("Failed to read.")
            continue

        # 透過PNGなどのアルファチャンネルは処理対象から外し、最後に復元する
        # （アルファにRetinexを掛けると透過情報が壊れるため）
        alpha = None
        if img.ndim == 3 and img.shape[2] == 4:
            img, alpha = img[:, :, :3], img[:, :, 3]

        msr_res = multi_scale_retinex(img, sigmas, weights)
        final_img = apply_mapping(msr_res, gain, offset)

        if alpha is not None:
            final_img = cv2.merge([final_img, alpha])

        if not cv2.imwrite(out_path, final_img):
            print("Failed to write.")
            continue

        elapsed = time.time() - start_time
        print(f"Done ({elapsed:.2f}s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="進捗表示・ソート対応 MSR プロセッサ")
    parser.add_argument("input_folder", help="入力フォルダのルートパス")
    parser.add_argument("--sigmas", nargs="+", type=float, default=[15, 80, 250])
    parser.add_argument("--gain", type=float, default=600.0)
    parser.add_argument("--offset", type=float, default=80.0)
    
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_folder)
    if not os.path.isdir(input_path):
        print(f"エラー: {input_path} はディレクトリではありません。")
        sys.exit(1)

    # 出力フォルダの決定（入力と同じ階層にサフィックスを付与）
    parent_dir = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    # パラメータ値をフォルダ名に含めることで識別性を向上
    folder_suffix = f"G{int(args.gain)}_O{int(args.offset)}_msr"
    output_path = os.path.join(parent_dir, f"{base_name}_{folder_suffix}")

    # 上書き確認ロジックの追加
    if os.path.exists(output_path):
        confirm = input(f"警告: 出力フォルダ '{output_path}' は既に存在します。上書きしますか？ (y/n): ")
        if confirm.lower() != 'y':
            print("処理を中断しました。")
            sys.exit(0)
    else:
        os.makedirs(output_path)

    # パラメータ保存（ファイル名をルールに準拠）
    save_msr_parameters(output_path, __file__, args.sigmas, args.gain, args.offset)

    # 処理開始
    process_recursive_with_progress(input_path, output_path, args.sigmas, args.gain, args.offset)
    print("\nすべての処理が完了しました。")