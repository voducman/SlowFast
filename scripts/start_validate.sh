export CUDA_VISIBLE_DEVICES=0
conda activate action-recognition
export PYTHONPATH=/u01/manvd1/action-recognition/SlowFast/slowfast:$PYTHONPATH

screen -L -Logfile slowfast.log -S slowfast -m bash -c "python tools/run_net.py \
  --cfg configs/Kinetics/custom/SLOWFAST_8x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR "/u01/manvd1/action-recognition/datasets/kinetics/val" \
  TEST.CHECKPOINT_FILE_PATH checkpoints/SLOWFAST_8x8_R50.pkl \
  TEST.CHECKPOINT_TYPE caffe2 \
  TRAIN.ENABLE False"
