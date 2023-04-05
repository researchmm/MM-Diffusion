REF_DIR="/home/v-zixituo/rld/dataset/landscape/train"
SAMPLE_DIR="/home/v-zixituo/rld/outputs/MM-Diffusion/samples/landscape/sr_mp4"
OUTPUT_DIR="/home/v-zixituo/rld/outputs/MM-Diffusion/eval/debug"

mpiexec -n 1 python py_scripts/eval.py --devices 0 --sample_num 2048 --ref_dir ${REF_DIR} --fake_dir ${SAMPLE_DIR} --output_dir ${OUTPUT_DIR}