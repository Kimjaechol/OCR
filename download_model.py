#!/usr/bin/env python3
"""
GOT-OCR 모델 다운로드 스크립트
HuggingFace에서 모델 파일을 다운로드합니다.
"""

from huggingface_hub import hf_hub_download, snapshot_download
import os

def download_got_ocr_model():
    """GOT-OCR2.0 모델을 다운로드합니다."""

    model_dir = "weights/GOT-OCR2_0"
    os.makedirs(model_dir, exist_ok=True)

    print("GOT-OCR2.0 모델 다운로드 중...")
    print("이 작업은 몇 분 정도 소요됩니다 (약 1.4GB)")

    try:
        # 전체 모델 다운로드
        snapshot_download(
            repo_id="stepfun-ai/GOT-OCR2_0",
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"\n모델 다운로드 완료: {model_dir}")

        # 다운로드된 파일 확인
        files = os.listdir(model_dir)
        print(f"다운로드된 파일: {files}")

        # safetensors 파일 확인
        safetensors_files = [f for f in files if f.endswith('.safetensors')]
        if safetensors_files:
            for f in safetensors_files:
                path = os.path.join(model_dir, f)
                size = os.path.getsize(path) / (1024**3)
                print(f"  - {f}: {size:.2f} GB")

        return True

    except Exception as e:
        print(f"다운로드 실패: {e}")
        return False

if __name__ == "__main__":
    success = download_got_ocr_model()
    if success:
        print("\n모델 준비 완료! OCR 서비스를 시작할 수 있습니다.")
    else:
        print("\n모델 다운로드에 실패했습니다. 네트워크 연결을 확인하세요.")
