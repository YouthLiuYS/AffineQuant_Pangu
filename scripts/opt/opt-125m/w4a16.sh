ADAROUND=${ADAROUND:-1}
ADAROUND_ARGS=""
if [ "$ADAROUND" -eq 1 ]; then
  ADAROUND_ARGS="--adaround"
fi
ADAROUND_LAYER_IDX=${ADAROUND_LAYER_IDX:-0}
ADAROUND_LAYER_ARGS=""
if [ -n "$ADAROUND_LAYER_IDX" ]; then
  ADAROUND_LAYER_ARGS="--adaround-layer-idx ${ADAROUND_LAYER_IDX}"
fi


CUDA_VISIBLE_DEVICES=0 python main.py \
--model ./opt-125m --eval_ppl --save_dir ./fake_quant_model/opt-125m-w4a16 \
--epochs 20 --output_dir ./log/opt-125m-w4a16 --act-scales ./act_scales/opt-125m.pt --act-shifts ./act_shifts/opt-125m.pt \
--wbits 4 --abits 16 --lwc --let --use_ln_matrix --sf 1.0 ${ADAROUND_ARGS} ${ADAROUND_LAYER_ARGS}
