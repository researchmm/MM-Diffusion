MODEL_FLAGS="--cross_attention_resolutions 2,4,8  --cross_attention_windows 1,4,8 
--cross_attention_shift True  
--class_cond False --video_attention_resolutions 2,4,8
--audio_attention_resolutions -1
--video_size 16,3,64,64 --audio_size 1,25600 --learn_sigma False --num_channels 128 
--num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True
--use_scale_shift_norm True"

# if classifier_scale(λ) is 0, conditional generation follows the replacement based method
# if classifier_scale(λ) is larger than 0, conditional generation follows the gradient based method
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear
--all_save_num 48 --save_type mp4  --devices 7
--batch_size 2   --is_strict True --sample_fn ddpm --classifier_scale 3.0"

MODEL_PATH="/data10/rld/outputs/MM-Diffusion/models/AIST++.pt"
OUT_DIR="/data10/rld/outputs/MM-Diffusion/video2audio/video2audio"
REF_PATH="/data10/rld/data/AIST++_crop/train"
NUM_GPUS=1

mpiexec -n $NUM_GPUS  python3 py_scripts/video2audio_sample.py  \
$MODEL_FLAGS $DIFFUSION_FLAGS \
--output_dir ${OUT_DIR} --model_path ${MODEL_PATH} --ref_path ${REF_PATH}
