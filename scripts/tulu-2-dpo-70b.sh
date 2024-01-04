CUDA_VISIBLE_DEVICES=0,1,2,3 \
python vllm_infer.py \
    --download_dir /net/nfs/s2-research/llama2/ \
    --data_name "commongen" \
    --model_name allenai/tulu-2-dpo-70b --tensor_parallel_size 4  --dtype bfloat16 \
    --output_folder "model_outputs/" \
    --top_p 1 --temperature 0 --batch_size 8 --max_tokens 128