DOT:= .
.PHONY: all clean
all: virtualenv
virtualenv:
	./setup_anaconda.sh venv venv 3.10
	. ./activate_python.sh; conda update -y -n base -c defaults conda
	. ./activate_python.sh; conda update -y -n base -c conda-forge conda
	. ./activate_python.sh; conda install -y numpy pandas scipy scikit-learn matplotlib librosa h5py pyyaml tensorboard tabulate pysoundfile seaborn -c conda-forge
	. ./activate_python.sh; conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
	. ./activate_python.sh; pip install umap-learn
clean:
	rm -fr venv miniconda.sh activate_python.sh
	find -iname "*.pyc" -delete


	