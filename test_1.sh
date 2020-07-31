python -m parlai.scripts.train_model -t convai2,convai2_wild_evaluation,mturkwikimovies,moviedialog:Task:1,moviedialog:Task:2,moviedialog:Task:3,moviedialog:Task:4,webquestions,ccpe,wikimovies,twitter,personachat,convai_chitchat,ubuntu,dailydialog,cornell_movie,coqa -dt train -m transformer/generator_septoken \
--init_model ./checkpoints/transformer-opensubtitles-1.checkpoint \
-mf ./checkpoints/transformer-opensubtitles-2 \
--dict-tokenizer bpe --dict-lower True --dict-septoken __sep__ --person-tokens True --add-p1-after-newln True --history-add-global-end-token end \
--n_positions=512 --truncate=512 \
--inference beam --beam-size 7 \
--embedding-size 512 --n_heads 8 --n_layers 6 --ffn_size 2048 --activation gelu --tie-layers \
--dropout 0.1 --attention-dropout 0.1 \
--num-epochs 2 -veps 0.01 -vme 512 -vmt ppl -vmm min -sval True \
-bs 8 --optimizer mem_eff_adam -lr 5e-5 --update_freq 16 \
--warmup_updates 8000 --lr-scheduler-patience 200 --lr-scheduler-decay 0.2 -vp 500 \
-tblog True --fp16 True --fp16-impl mem_efficient --dynamic_batching batchsort --evaltask convai2

# --tie_layers
# --bpe-vocab ./checkpoints/transformer-withseptoken-opensubtitles-2.checkpoint.dict \
# --bpe-merge ./checkpoints/transformer-withseptoken-opensubtitles-2.checkpoint.dict.codecs \
# empathetic_dialogues,taskmaster,convai2,convai2_wild_evaluation,amazon_qa