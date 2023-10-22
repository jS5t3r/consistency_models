python scripts/image_sample.py --training_mode edm --generator determ-indiv --batch_size 8 --sigma_max 80 --sigma_min 0.002 \
    --s_churn 0 --steps 40 --sampler heun --model_path /home/lorenzp/workspace/consistency_models/checkpoints/edm_bedroom256_ema.pt \
    --attention_resolutions 32,16,8 \
    --class_cond False --dropout 0.1 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 \
    --num_samples 50000 --resblock_updown True --use_fp16 True --use_scale_shift_norm False --weight_schedule karras