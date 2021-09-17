export CUDA_VISIBLE_DEVICES=5
conda activate action-recognition
export PYTHONPATH=/u01/manvd1/action-recognition/SlowFast/slowfast:$PYTHONPATH
rm slowfast-demo.log
screen -L -Logfile slowfast-demo.log -S slowfast -m bash -c " python tools/run_net.py --cfg demo/Kinetics/SLOWFAST_8x8_R50.yaml"
