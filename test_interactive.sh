CUDA_VISIBLE_DEVICES=-1 python -m parlai.scripts.interactive \
--inference beam --beam_length_penalty .65 --temperature .7 --topk 40 --beam-size 10 \
-mf ./checkpoints/transformer-distilled.checkpoint
# -mf ./checkpoints/transformer-reddit-4.checkpoint
# -mf ./checkpoints/transformer-convai2-transfer.checkpoint
