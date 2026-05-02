
# Part A
python3 -m continuous_connector/cache_clip --split both
python3 -m continuous_connector/train_phase1
python3 -m continuous_connector/train_phase2
python3 -m continuous_connector/train_phase3
python3 -m continuous_connector/eval --alpaca-ppl
python3 -m continuous_connector/eval --checkpoint phase3
python3 -m continuous_connector/eval --majority
python3 -m continuous_connector/eval --text-only
python3 -m continuous_connector/eval --checkpoint phase3 --qualitative
python3 -m continuous_connector/modality_gap --name initial
python3 -m continuous_connector/modality_gap --checkpoint phase1
python3 -m continuous_connector/modality_gap --checkpoint phase2
python3 -m continuous_connector/modality_gap --checkpoint phase3
python3 -m continuous_connector/ablations --kind lambda
python3 -m continuous_connector/cache_clip --split both --include-cls
python3 -m continuous_connector/ablations --kind representation --cache cifar_train_clip_with_cls.pt
python3 -m continuous_connector/ablations --kind norm

# Part B
python3 discrete_vq_vae/synthetic_data.py --force
python3 discrete_vq_vae/train_vqvae.py --ema --name vqvae_best
python3 discrete_vq_vae/train_lm.py
python3 discrete_vq_vae/eval_vqa.py
python3 discrete_vq_vae/eval_vqa.py --text-only
python3 discrete_vq_vae/eval_imggen.py
python3 discrete_vq_vae/ablations.py --kind no-projector