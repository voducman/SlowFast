export CUDA_VISIBLE_DEVICES=0
conda activate action-recognition
export PYTHONPATH=/u01/manvd1/action-recognition/SlowFast/slowfast:$PYTHONPATH
rm logs/slowfast-demo.log
screen -L -Logfile logs/slowfast-demo.log -S slowfast-demo -m bash -c \
"python tools/run_net.py \
--cfg demo/Kinetics/SLOWFAST_8x8_R50.yaml \
TEST.CHECKPOINT_FILE_PATH checkpoints/SLOWFAST_8x8_R50.pkl \
DEMO.ENABLE True \
DEMO.LABEL_FILE_PATH /u01/khienpv1/manvd1/action-recognition/data/kinetics-400/class_id_mapping.json \
DEMO.INPUT_VIDEO /u01/khienpv1/manvd1/action-recognition/demo \
OUTPUT_DIR ../demo/output
"
