# src/components/segment_manager.py
import os
import numpy as np
from pathlib import Path

class SegmentManager:
    def __init__(self, project_root, dt):
        self.project_root = Path(project_root)
        self.dt = dt

    def create_segments(self, input_full, output_full, segment_sec, warmup_sec, temp_dir):
        """
        データを指定秒数ごとに分割し、Warm-up区間を付与して一時ファイルに保存する。
        """
        full_len = len(input_full)
        seg_len = int(segment_sec / self.dt)
        warmup_len = int(warmup_sec / self.dt)
        
        segments_info = []
        # 切り上げで分割数を決定
        num_segments = int(np.ceil(full_len / seg_len))

        print(f"--- Segment Manager ---")
        print(f"Total Segments: {num_segments} (Segment: {segment_sec}s, Warmup: {warmup_sec}s)")

        for i in range(num_segments):
            # 本番区間 (Core) の開始・終了
            core_start = i * seg_len
            core_end = min((i + 1) * seg_len, full_len)
            
            if core_start >= full_len:
                break

            # Warm-upを含めた実際の開始位置 (Start)
            # 0秒未満にならないようにmaxを取る
            actual_start = max(0, core_start - warmup_len)
            
            # データ切り出し
            seg_input = input_full[actual_start:core_end]
            seg_output = output_full[actual_start:core_end]
            
            # トリムすべき秒数（Warm-upの長さ）
            # 最初のセグメントなど、Warm-upが取れない場合は短くなるため計算する
            current_trim_sec = (core_start - actual_start) * self.dt

            # ファイル保存
            seg_id = i + 1
            inp_path = temp_dir / f"{seg_id}_input.txt"
            out_path = temp_dir / f"{seg_id}_output.txt"
            
            np.savetxt(inp_path, seg_input)
            np.savetxt(out_path, seg_output)

            segments_info.append({
                'id': seg_id,
                'input_path': str(inp_path),
                'output_path': str(out_path),
                'trim_sec': current_trim_sec,
                'info': f"Core[{core_start}:{core_end}] (Warmup {current_trim_sec:.2f}s)"
            })
            
        return segments_info