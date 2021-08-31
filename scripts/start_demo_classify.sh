export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=/u01/manvd1/action-recognition/SlowFast/slowfast:$PYTHONPATH
conda activate action-recognition
test -f "logs/slowfast-demo.log" && rm logs/slowfast-demo.log
screen -L -Logfile logs/slowfast-demo.log -S slowfast-demo -m bash -c \
"python tools/run_net.py \
--cfg demo/Kinetics/SLOWFAST_8x8_R50_ft.yaml \
TEST.CHECKPOINT_FILE_PATH checkpoints/checkpoint_epoch_00054_1e-4.pyth \
TEST.CHECKPOINT_TYPE pytorch \
DEMO.ENABLE True \
DEMO.LABEL_FILE_PATH /u01/khienpv1/manvd1/action-recognition/data/kinetics-400/class_id_mapping.json \
DEMO.INPUT_VIDEO /u01/khienpv1/manvd1/action-recognition/demo \
DEMO.VIS_MODE thres \
DEMO.COMMON_CLASS_THRES 0.5 \
OUTPUT_DIR ../demo/output/thresh && \
 \
python tools/run_net.py \
--cfg demo/Kinetics/SLOWFAST_8x8_R50_ft.yaml \
TEST.CHECKPOINT_FILE_PATH checkpoints/checkpoint_epoch_00054_1e-4.pyth \
TEST.CHECKPOINT_TYPE pytorch \
DEMO.ENABLE True \
DEMO.LABEL_FILE_PATH /u01/khienpv1/manvd1/action-recognition/data/kinetics-400/class_id_mapping.json \
DEMO.INPUT_VIDEO /u01/khienpv1/manvd1/action-recognition/demo \
DEMO.VIS_MODE top-k \
OUTPUT_DIR ../demo/output/top-k
"
