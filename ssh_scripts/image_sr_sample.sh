MODEL_FLAGS="--sr_attention_resolutions 8,16,32  --large_size 256  
--small_size 64 --sr_learn_sigma True --sr_class_cond False
--sr_num_channels 192 --sr_num_heads 4 --sr_num_res_blocks 2 
--sr_resblock_updown True --use_fp16 True --sr_use_scale_shift_norm True"

DIFFUSION_FLAGS="--sr_diffusion_steps 1000 --is_strict True --noise_schedule linear 
--all_save_num 24 --devices 5,6 --save_type gif --img_type gif 
--batch_size 1 --is_strict False --save_noise False  "

REF_PATH="None"
DATA_DIR="/data6/rld/data/landscape/train/explosion"

MODEL_PATH="/data6/rld/guided-diffusion/models/image_sr/landscape_64-256_np50_nr0-20_cp50_cr20-8_ema140000.pt"
LOAD_NOISE="None" #outputs/video-sr/fromgt_AIST++_crop/AIST++_64_256_model350000_dpm_solver++.npy"
NUM_GPUS=2
SAMPEL_ALL="dpm_solver++_False_2_singlestep
dpm_solver++_False_2_multistep
dpm_solver++_False_3_singlestep
dpm_solver++_False_3_multistep
dpm_solver++_True_2_singlestep
dpm_solver++_True_2_multistep
dpm_solver++_True_3_singlestep
dpm_solver++_True_3_multistep"

for SAMPLE_FN in $SAMPEL_ALL; 
do

OUT_DIR="outputs/video-sr/fromgt_landscape/$SAMPLE_FN"
mpiexec -n $NUM_GPUS --allow-run-as-root python py_scripts/image_sr_sample.py --load_noise ${LOAD_NOISE} \
$MODEL_FLAGS $DIFFUSION_FLAGS --base_samples ${DATA_DIR} --output_dir ${OUT_DIR} --model_path ${MODEL_PATH} \
--ref_path ${REF_PATH} --sample_fn ${SAMPLE_FN}
done


