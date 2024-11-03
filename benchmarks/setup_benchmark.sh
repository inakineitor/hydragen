mkdir ~/model_save_dir
export MODEL_SAVE_DIR="~/model_save_dir"

mkdir /workspace/results
export RESULTS_SAVE_DIR="/workspace/results"

export FULL_MODEL_NAME="meta-llama/Llama-3.1-8B"

export NUM_MODEL_SPLITS=4
export MODEL_SPLIT_NAME="split${NUM_MODEL_SPLITS}-llama-3point1-8b"

export MAKE_TP="true"
export BENCHMARK_MODE="vllm" # Options: "e2e", "noattention", "flashattention", "vllm", "sglang"

if [[ "$MAKE_TP" == "true" ]]; then
	python hydragen/make_tp_files.py "${FULL_MODEL_NAME}" "${MODEL_SAVE_DIR}/${MODEL_SPLIT_NAME}" --num-splits "${NUM_MODEL_SPLITS}"
fi

if [[ "$BENCHMARK_MODE" == "e2e" ]]; then
	# ========== End-to-end ==========
	torchrun --standalone --nproc_per_node=$NUM_MODEL_SPLITS scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-8b 32:129:x2 1024,2048,4096,8192,16256 128 --mode hydragen --num-iters 10 --num-warmup 10 --model-name $FULL_MODEL_NAME --tp-dir "${MODEL_SAVE_DIR}/${MODEL_SPLIT_NAME}"
	torchrun --standalone --nproc_per_node=$NUM_MODEL_SPLITS scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-8b 256:2049:x2 1024,2048,4096,8192,16256 128 --mode hydragen --num-iters 3 --num-warmup 3 --model-name $FULL_MODEL_NAME --tp-dir "${MODEL_SAVE_DIR}/${MODEL_SPLIT_NAME}"
	torchrun --standalone --nproc_per_node=$NUM_MODEL_SPLITS scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-8b-c256 32:129:x2 1024,2048,4096,8192,16128 256 --mode hydragen --num-iters 10 --num-warmup 10 --model-name $FULL_MODEL_NAME --tp-dir "${MODEL_SAVE_DIR}/${MODEL_SPLIT_NAME}"
	torchrun --standalone --nproc_per_node=$NUM_MODEL_SPLITS scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-8b-c256 256:1025:x2 1024,2048,4096,8192,16128 256 --mode hydragen --num-iters 3 --num-warmup 3 --model-name $FULL_MODEL_NAME --tp-dir "${MODEL_SAVE_DIR}/${MODEL_SPLIT_NAME}"
elif [[ "$BENCHMARK_MODE" == "noattention" ]]; then
	# ========== No Attention ==========
	torchrun --standalone --nproc_per_node=$NUM_MODEL_SPLITS scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-8b 32:2049:x2 1024 128 --mode noattention --num-iters 10 --num-warmup 10 --model-name $FULL_MODEL_NAME --tp-dir "${MODEL_SAVE_DIR}/${MODEL_SPLIT_NAME}"
	torchrun --standalone --nproc_per_node=$NUM_MODEL_SPLITS scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-8b-c256 32:1025:x2 1024 256 --mode noattention --num-iters 10 --num-warmup 10 --model-name $FULL_MODEL_NAME --tp-dir "${MODEL_SAVE_DIR}/${MODEL_SPLIT_NAME}"
elif [[ "$BENCHMARK_MODE" == "flashattention" ]]; then
  # ========== Flash Attention ==========
  torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-8b 32 1024,2048,4096,8192,16256 128 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name $FULL_MODEL_NAME --tp-dir "${MODEL_SAVE_DIR}/${MODEL_SPLIT_NAME}"
  torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-8b 64 1024,2048,4096,8192 128 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name $FULL_MODEL_NAME --tp-dir "${MODEL_SAVE_DIR}/${MODEL_SPLIT_NAME}"
  torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-8b 128 1024,2048,4096 128 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name $FULL_MODEL_NAME --tp-dir "${MODEL_SAVE_DIR}/${MODEL_SPLIT_NAME}"
  torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-8b 256 1024,2048 128 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name $FULL_MODEL_NAME --tp-dir "${MODEL_SAVE_DIR}/${MODEL_SPLIT_NAME}"
  torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-8b-c256 32 1024,2048,4096,8192,16128 256 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name $FULL_MODEL_NAME --tp-dir "${MODEL_SAVE_DIR}/${MODEL_SPLIT_NAME}"
  torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-8b-c256 64 1024,2048,4096,8192 256 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name $FULL_MODEL_NAME --tp-dir "${MODEL_SAVE_DIR}/${MODEL_SPLIT_NAME}"
  torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-8b-c256 128 1024,2048,4096 256 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name $FULL_MODEL_NAME --tp-dir "${MODEL_SAVE_DIR}/${MODEL_SPLIT_NAME}"
  torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-8b-c256 256 1024 256 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name $FULL_MODEL_NAME --tp-dir "${MODEL_SAVE_DIR}/${MODEL_SPLIT_NAME}"
elif [[ "$BENCHMARK_MODE" == "vllm"]]; then
  # ========== vLLM ==========
  python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-8b 32:65:x2 1024,2048,4096,8192,16256 128 --mode vllm --tp $NUM_MODEL_SPLITS --num-iters 3 --num-warmup 3 --model-name $FULL_MODEL_NAME
  python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-8b 128:2049:x2 1024,2048,4096,8192,16256 128 --mode vllm --tp $NUM_MODEL_SPLITS --num-iters 1 --num-warmup 1 --model-name $FULL_MODEL_NAME
  python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-8b-c256 32:65:x2 1024,2048,4096,8192,16128 256 --mode vllm --tp $NUM_MODEL_SPLITS --num-iters 3 --num-warmup 3 --model-name $FULL_MODEL_NAME
  python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-8b-c256 128:2049:x2 1024,2048,4096,8192,16128 256 --mode vllm --tp $NUM_MODEL_SPLITS --num-iters 1 --num-warmup 1 --model-name $FULL_MODEL_NAME
elif [[ "$BENCHMARK_MODE" == "sglang"]]; then
  # ========== SGLang ==========
  python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-8b 32:65:x2 1024,2048,4096,8192,16256 128 --mode sglang --tp $NUM_MODEL_SPLITS --num-iters 3 --num-warmup 3 --model-name $FULL_MODEL_NAME
  python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-8b 128:2049:x2 1024,2048,4096,8192,16256 128 --mode sglang --tp $NUM_MODEL_SPLITS --num-iters 1 --num-warmup 1 --model-name $FULL_MODEL_NAME
  python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-8b-c256 32:65:x2 1024,2048,4096,8192,16128 256 --mode sglang --tp $NUM_MODEL_SPLITS --num-iters 3 --num-warmup 3 --model-name $FULL_MODEL_NAME
  python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-8b-c256 128:2049:x2 1024,2048,4096,8192,16128 256 --mode sglang --tp $NUM_MODEL_SPLITS --num-iters 1 --num-warmup 1 --model-name $FULL_MODEL_NAME
fi
