\conda activate action-recognition
export PYTHONPATH=/u01/manvd1/action-recognition/SlowFast/slowfast:$PYTHONPATH
rm slowfast-training.log
screen -L -Logfile slowfast-training.log -S slowfast -m bash -c "CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/run_net.py --cfg configs/Kinetics/custom/SLOWFAST_8x8_R50_FINE_TUNE.yaml"
