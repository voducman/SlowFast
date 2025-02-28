conda create -n action-recognition python==3.8 -y
conda activate action-recognition

# cuda vs cudnn
conda install -c anaconda cudatoolkit=10.2 -y
conda install -c nvidia cudnn=8.0.4 -y

pip install torch==1.6.0 torchvision==0.7.0
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install simplejson
conda install av -c conda-forge
pip install -U iopath
pip install psutil
pip install opencv-python
pip install torchvision
pip install tensorboard
pip install moviepy
pip install pytorchvideo
pip install sklearn

pip install tensorflow-gpu==2.4.0

# Detectron2
pip install -U torch torchvision cython
pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo

echo "Dependencies were installed successfully."
