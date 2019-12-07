# Computational Mirrors: Blind Inverse Light Transport by Deep Matrix Factorization

<img src='imgs/light_transport.gif' align="center">
Paper Link: https://arxiv.org/abs/1912.02314
Paper Webpage: http://compmirrors.csail.mit.edu

We present the implementation for both matrix factorization (Figure 4) and blind light transport factorization (Figure 6) methods presented in the paper.

## Setup
```bash
git clone https://github.com/prafull7/compmirrors
cd compmirrors
```

Our code runs using Python=3.7 with the following packages:

- torch=1.0.1.post2
- matplotlib
- scipy
- visdom

## Matrix Factorization

The implementation for the file is present in factorization_1d.py. 
It can be run using the command:
```bash
python factorization_1d.py -T ./data/inputs_1d/lightfield.png -L ./data/inputs_1d/tracks_bg.png -o ./outdir_1d 
```

## Blind Light Transport Factorization

The implementation for the one-off training is in factorization_light_transport.py. 
It can be run using the command:
```bash
export FACTORIZE_DATA_DIR=/path/to/where/data/folders/
export FACTORIZE_OUT_DIR=/path/to/output/directory
python factorization_light_transport.py -d ./data/light_transport/ -f FOLDER_NAME -ds DATASET_NAME -s SEQUENCE_NAME -dev DEVICE_NUMBER
```

In the above command:
- FOLDER_NAME: Name of the folder in the dataset folder to run i.e. disc_data.
- DATASET_NAME: Prefix of the dataset within that folder, i.e. mnist.
- SEQUENCE_NAME: Name of the sequence to run. These can be found as the key names in frames.txt file within each folder.
- DEVICE_NAME: GPU id to run the code.
