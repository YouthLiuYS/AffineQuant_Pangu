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
--model /path/to/opt-2.7b --eval_ppl --save_dir ./fake_quant_model/opt-2.7b-w3a16g128 \
--epochs 20 --output_dir ./log/opt-2.7b-w3a16g128 --act-scales ./act_scales/opt-2.7b.pt --act-shifts ./act_shifts/opt-2.7b.pt \
--wbits 3 --abits 16 --group_size 128 --lwc --let --use_ln_matrix --sf 1.0 ${ADAROUND_ARGS} ${ADAROUND_LAYER_ARGS}
