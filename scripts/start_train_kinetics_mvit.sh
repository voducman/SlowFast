export CUDA_VISIBLE_DEVICES=4,6
conda activate action-recognition
export PYTHONPATH=/u01/manvd1/action-recognition/SlowFast/slowfast:$PYTHONPATH
test -f "logs/slowfast-train-mvit-10.log" && rm logs/slowfast-train-mvit-10.log
screen -L -Logfile logs/slowfast-train-mvit-10.log -S slowfast-train-mvit-10 -m bash -c \
"python tools/run_net.py --cfg configs/Kinetics/custom/MVIT_B_16x4_CONV.yaml"
