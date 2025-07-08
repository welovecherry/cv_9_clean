#!/bin/bash

echo "🟠 시작: generate_augmentations_v2.py"
python scripts/generate_augmentations_v2.py

echo "🟢 완료: generate_augmentations_v2.py"
echo "🟠 시작: generate_augmentations_v3.py"
python scripts/generate_augmentations_v3.py

echo "🟢 전체 증강 완료"