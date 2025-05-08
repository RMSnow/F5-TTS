
model_name="f5tts"
save_root="/storage/zhangxueyao/workspace/SpeechGenerationYC_ckpts/ckpts/vevo2/baselines/${model_name}"
task_name="tts"

mkdir -p $save_root

# 创建日志目录
log_dir="${save_root}/log_inference"
mkdir -p $log_dir

# ========= Define Function =========
export PYTHONPATH="/storage/zhangxueyao/workspace/F5-TTS/src"

run_infer() {
    local cuda_id=$1
    local eval_setting=$2
    local evalset_root=$3

    CUDA_VISIBLE_DEVICES=$cuda_id python infer_cli_vevo2_eval.py \
        --save_root $save_root \
        --model_name $model_name \
        --evalset_root $evalset_root \
        --eval_setting $eval_setting \
        --task_name $task_name > "${log_dir}/${task_name}_${eval_setting}.log" 2>&1 &
}

# ===== Run (TTS) =====
# evalset_root="/storage/zhangxueyao/workspace/SpeechGenerationYC/EvalSet/tts"
# run_infer 6 "genshin_tts_en" $evalset_root
# run_infer 7 "genshin_tts_zh" $evalset_root

evalset_root="/storage/zhangxueyao/workspace/SpeechGenerationYC/EvalSet/svs"
run_infer 6 "gtsinger_svs_en" $evalset_root
run_infer 7 "gtsinger_svs_zh" $evalset_root

# 等待所有后台进程完成
wait

echo "所有推理任务已完成，日志文件保存在 ${log_dir} 目录下"