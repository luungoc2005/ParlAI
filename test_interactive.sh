CUDA_VISIBLE_DEVICES=-1 python ./examples/interactive.py \
--beam_length_penalty .65 --temperature 1.2 --topk 40 \
-mf ./checkpoints/transformer-convai2-transfer-3.checkpoint
# -mf ./checkpoints/transformer-convai2-transfer.checkpoint
