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
--model /PATH/TO/Llama-2-13b-hf --eval_ppl --save_dir ./fake_quant_model/Llama-2-13b-w2a16 \
--epochs 40 --output_dir ./log/Llama-2-13b-w2a16 \
--wbits 2 --abits 16 --lwc --aug_loss --let --tasks hendrycksTest,piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande ${ADAROUND_ARGS} ${ADAROUND_LAYER_ARGS}
