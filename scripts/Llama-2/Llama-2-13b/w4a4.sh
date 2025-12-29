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
--model /path/to/Llama-2-13b-hf --eval_ppl --save_dir ./fake_quant_model/Llama-2-13b-w4a4 \
--epochs 20 --output_dir ./log/Llama-2-13b-w4a4 --act-scales ./act_scales/Llama-2-13b.pt --act-shifts ./act_shifts/Llama-2-13b.pt \
--wbits 4 --abits 4 --lwc --let --use_matrix --sf 0.1 --let_lr 1e-3 --alpha 0.75 \
--tasks hendrycksTest,piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande ${ADAROUND_ARGS} ${ADAROUND_LAYER_ARGS}
