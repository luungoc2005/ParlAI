CUDA_VISIBLE_DEVICES=-1 python -m parlai.scripts.interactive \
--inference beam --beam_length_penalty .65 --temperature 1.4 --topk 40 --beam-size 7 \
-mf ./checkpoints/transformer-reddit.checkpoint
# -mf ./checkpoints/transformer-convai2-transfer.checkpoint
