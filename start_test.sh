export CUDA_VISIBLE_DEVICES=7
conda activate action-recognition
export PYTHONPATH=/u01/khienpv1/manvd1/action-recognition/SlowFast/slowfast:$PYTHONPATH
test -f "logs/slowfast-test.log" && rm logs/slowfast-test.log
screen -L -Logfile logs/slowfast-test.log -S slowfast-test -m bash -c \
"python tools/run_net.py --cfg configs/Kinetics/custom/SLOWFAST_8x8_R50_TEST.yaml"
