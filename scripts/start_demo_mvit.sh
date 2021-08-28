export CUDA_VISIBLE_DEVICES=0
conda activate action-recognition
test -f "logs/mvit-demo.log" && rm logs/mvit-demo.log
screen -L -Logfile logs/mvit-demo.log -S mvit-demo -m bash -c \
"python tools/run_net.py \
--cfg demo/Kinetics/MVIT_B_16x4_CONV.yaml \
TEST.CHECKPOINT_FILE_PATH checkpoints/K400_MVIT_B_16x4_CONV.pyth \
TEST.CHECKPOINT_TYPE caffe2 \
DEMO.ENABLE True \
DEMO.LABEL_FILE_PATH /u01/khienpv1/manvd1/action-recognition/data/kinetics-400/class_id_mapping.json \
DEMO.INPUT_VIDEO /u01/khienpv1/manvd1/action-recognition/demo \
OUTPUT_DIR ../demo/output
"
