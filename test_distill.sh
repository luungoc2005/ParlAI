python -m parlai.scripts.train_distill_model \
-t fromfile:parlaiformat --fromfile_datapath ./data/reddit/train.txt -dt train:stream \
-m transformer/generator_distilled \
-mf ./checkpoints/transformer-distilled-2 \
-teacher-model-file zoo:blender/blender_90M/model \
--dict-lower True \
--variant prelayernorm --n_positions=512 --truncate=512 \
--inference greedy --beam-size 1 --skip-generation True \
--share_word_embeddings True \
--embedding-size 312 --n-heads 12 --n-layers 4 --ffn_size 1200 --activation relu --gradient-clip 0.1 \
--learn-positional-embeddings False --dropout 0.1 --attention-dropout 0.1 \
--num-epochs 1 -veps 0.01 -vme 512 -vmt ppl -vmm min -sval True \
-bs 24 --optimizer mem_eff_adam -lr 1e-05 \
--warmup_updates -1 --warmup_rate 0.0001 -vp 500 \
-tblog True --fp16 True --fp16-impl mem_efficient --evaltask convai2

# -t fromfile:parlaiformat --fromfile_datapath ./data/reddit/train.txt -dt train:stream \
#-t blended_skill_talk,wizard_of_wikipedia,convai2:normalized,empathetic_dialogues --multitask-weights 1,3,3,3 \