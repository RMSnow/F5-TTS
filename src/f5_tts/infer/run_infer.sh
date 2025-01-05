#!/bin/bash

# 创建日志目录
mkdir -p logs

# 运行时记录开始时间
echo "Starting jobs at $(date)" > logs/main.log

fake_ref_audio="/storage/zhangxueyao/workspace/F5-TTS/00000309-00000300.wav"

# python infer_cli.py \
#     --model "F5-TTS" \
#     --ref_audio $fake_ref_audio \
#     --ref_text "fake_ref_text" \
#     --gen_text "fake_gen_text" \
#     --start 0 \
#     --end 10

CUDA_VISIBLE_DEVICES=0 python infer_cli.py --start 0 --end 1500 --model "F5-TTS" --ref_audio $fake_ref_audio --ref_text "fake_ref_text" --gen_text "fake_gen_text" > logs/gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python infer_cli.py --start 1500 --end 3000 --model "F5-TTS" --ref_audio $fake_ref_audio --ref_text "fake_ref_text" --gen_text "fake_gen_text" > logs/gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python infer_cli.py --start 3000 --end 4500 --model "F5-TTS" --ref_audio $fake_ref_audio --ref_text "fake_ref_text" --gen_text "fake_gen_text" > logs/gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python infer_cli.py --start 4500 --end 6000 --model "F5-TTS" --ref_audio $fake_ref_audio --ref_text "fake_ref_text" --gen_text "fake_gen_text" > logs/gpu3.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python infer_cli.py --start 6000 --end 7500 --model "F5-TTS" --ref_audio $fake_ref_audio --ref_text "fake_ref_text" --gen_text "fake_gen_text" > logs/gpu4.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python infer_cli.py --start 7500 --end 9000 --model "F5-TTS" --ref_audio $fake_ref_audio --ref_text "fake_ref_text" --gen_text "fake_gen_text" > logs/gpu5.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python infer_cli.py --start 9000 --end 10500 --model "F5-TTS" --ref_audio $fake_ref_audio --ref_text "fake_ref_text" --gen_text "fake_gen_text" > logs/gpu6.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python infer_cli.py --start 10500 --end 12000 --model "F5-TTS" --ref_audio $fake_ref_audio --ref_text "fake_ref_text" --gen_text "fake_gen_text" > logs/gpu7.log 2>&1 &

wait

# 记录结束时间
echo "All jobs completed at $(date)" >> logs/main.log
