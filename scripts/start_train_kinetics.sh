export CUDA_VISIBLE_DEVICES=0
conda activate action-recognition
export PYTHONPATH=/u01/manvd1/action-recognition/SlowFast/slowfast:$PYTHONPATH
rm logs/slowfast-training.log
screen -L -Logfile logs/slowfast-training.log -S slowfast-train -m bash -c \
"python tools/run_net.py --cfg configs/Kinetics/custom/SLOWFAST_8x8_R50_FINE_TUNE.yaml"
