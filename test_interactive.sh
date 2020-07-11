CUDA_VISIBLE_DEVICES=-1 python -m parlai.scripts.interactive \
--inference beam --beam_length_penalty .65 --temperature 2.4 --topk 60 --beam-size 7 \
-mf ./checkpoints/transformer-reddit-4.checkpoint
# -mf ./checkpoints/transformer-convai2-transfer.checkpoint
