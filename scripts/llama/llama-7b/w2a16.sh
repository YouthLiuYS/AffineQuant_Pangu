ADAROUND=${ADAROUND:-0}
ADAROUND_ARGS=""
if [ "$ADAROUND" -eq 1 ]; then
  ADAROUND_ARGS="--adaround"
fi
ADAROUND_LAYER_IDX=${ADAROUND_LAYER_IDX:-}
ADAROUND_LAYER_ARGS=""
if [ -n "$ADAROUND_LAYER_IDX" ]; then
  ADAROUND_LAYER_ARGS="--adaround-layer-idx ${ADAROUND_LAYER_IDX}"
fi


CUDA_VISIBLE_DEVICES=0 python main.py \
--model /path/to/llama-7b-hf --eval_ppl --save_dir ./fake_quant_model/llama-7b-w2a16 \
--epochs 40 --output_dir ./log/llama-7b-w2a16 --act-scales ./act_scales/llama-7b.pt --act-shifts ./act_shifts/llama-7b.pt \
--wbits 2 --abits 16 --lwc --aug_loss --let --use_ln_matrix --sf 1e-2 ${ADAROUND_ARGS} ${ADAROUND_LAYER_ARGS}
