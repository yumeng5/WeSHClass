
dataset=nyt
sup_source=keywords

export CUDA_VISIBLE_DEVICES=5

python main.py --dataset ${dataset} --sup_source ${sup_source} --beta 500 \
	--maxiter "5000,5000" --gamma 0.9 --block_level 1 --pseudo "lstm" --with_eval "All"
