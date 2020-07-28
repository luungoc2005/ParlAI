python -m parlai.scripts.train_model \
-t blended_skill_talk,wizard_of_wikipedia,convai2:normalized -dt train \
--multitask-weights 1,3,3,3 \
-m transformer/generator \
-mf ./checkpoints/transformer-distilled \
--dict-lower True -lr 1e-06 --optimizer adamax --lr-scheduler reduceonplateau --gradient-clip 0.1 -veps 0.25 --betas 0.9,0.999 --update-freq 1 --attention-dropout 0.0 --relu-dropout 0.0 --skip-generation True -vp 15 -stim 60 -vme 20000 -bs 16 -vmt ppl -vmm min --save-after-valid True \
-tblog True --evaltask convai2
