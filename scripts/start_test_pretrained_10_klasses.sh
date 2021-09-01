export CUDA_VISIBLE_DEVICES=2
conda activate action-recognition
export PYTHONPATH=/u01/manvd1/action-recognition/SlowFast/slowfast:$PYTHONPATH
test -f "logs/slowfast-test.log" && rm logs/slowfast-test.log
screen -L -Logfile logs/slowfast-test.log -S slowfast-test -m bash -c "python tools/run_net.py \
  --cfg configs/Kinetics/custom/SLOWFAST_8x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR /u01/khienpv1/manvd1/action-recognition/data/kinetics-400/val \
  TEST.CHECKPOINT_FILE_PATH checkpoints/SLOWFAST_8x8_R50.pkl \
  TEST.CHECKPOINT_TYPE caffe2 \
  TRAIN.ENABLE False \
  OUTPUT_DIR ./output/test/pre-trained"
