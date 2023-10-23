mpiexec -n 1 python image_sample_inpainting.py --batch_size 16 \
--training_mode consistency_distillation \
--sampler multistep \
--ts 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40 \
--steps 40 \
--model_path /home/lorenzp/workspace/consistency_models/checkpoints/cd_bedroom256_lpips.pt \
--attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 \
--image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 \
--num_samples 16 \
--resblock_updown True --use_fp16 True --weight_schedule uniform
