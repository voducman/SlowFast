conda activate action-recognition
export PYTHONPATH=/u01/manvd1/action-recognition/SlowFast/slowfast:$PYTHONPATH
# run tensorboard and explore to netowrks
tensorboard --logdir=/u01/manvd1/action-recognition/SlowFast/output/runs-kinetics --port=10000 --bind_all
