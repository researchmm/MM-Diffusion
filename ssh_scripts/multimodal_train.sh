
#################64 x 64 uncondition###########################################################
MODEL_FLAGS="--cross_attention_resolutions 2,4,8 --cross_attention_windows 1,4,8
--cross_attention_shift True --dropout 0.1 
--video_attention_resolutions 2,4,8
--audio_attention_resolutions -1
--video_size 16,3,64,64 --audio_size 1,25600 --learn_sigma False --num_channels 128
--num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True 
--use_scale_shift_norm True --num_workers 4"

# Modify --devices to your own GPU ID
TRAIN_FLAGS="--lr 0.0001 --batch_size 4 
--devices G8 --log_interval 100 --save_interval 10000 --use_db False " #--schedule_sampler loss-second-moment
DIFFUSION_FLAGS="--noise_schedule linear --diffusion_steps 1000 --save_type mp4 --sample_fn ddpm" 

# Modify the following pathes to your own paths
DATA_DIR="/home/rld/dataset/landscape/train"
OUTPUT_DIR="/home/rld/outputs/MM-Diffusion/debug/"
NUM_GPUS=8

mpiexec -n $NUM_GPUS  python3 py_scripts/multimodal_train.py --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} $MODEL_FLAGS $TRAIN_FLAGS $VIDEO_FLAGS $DIFFUSION_FLAGS 
