CUDA_VISIBLE_DEVICES=-1 python -m parlai.scripts.interactive \
--beam_length_penalty .65 --temperature 1.4 --topk 80 \
-mf ./checkpoints/transformer-opensubtitles-2.checkpoint
# -mf ./checkpoints/transformer-convai2-transfer.checkpoint
