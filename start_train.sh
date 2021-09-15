export CUDA_VISIBLE_DEVICES=4,5
conda activate action-recognition
export PYTHONPATH=/u01/khienpv1/manvd1/action-recognition/SlowFast/slowfast:$PYTHONPATH
test -f "logs/slowfast-train.log" && rm logs/slowfast-train.log
screen -L -Logfile logs/slowfast-train.log -S slowfast -m bash -c \
"python tools/run_net.py --cfg configs/Kinetics/custom/SLOWFAST_8x8_R50_FINE_TUNE.yaml"
