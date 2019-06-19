# OpenPose-Plus: Pose Estimation in the Wild

</a>
<p align="center">
    <img src="https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/media/dance_foot.gif?raw=true", width="360">
</p>

[![Documentation Status](https://readthedocs.org/projects/openpose-plus/badge/?version=latest)](https://openpose-plus.readthedocs.io/en/latest/?badge=latest)

[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) is the state-of-the-art pose estimation algorithm.
In its Caffe [codebase](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation),
data augmentation, training, and neural networks are most hard-coded. They are difficult
to be customised. In addition,
key performance features such as embedded platform supports and parallel GPU training are missing.
All these limitations makes OpenPose, in these days, hard to 
be deployed in the wild. To resolve this, we develop **OpenPose-Plus**, a high-performance yet flexible pose estimation framework that offers many powerful features:
- Flexible combination of standard training dataset with your own custom labelled data.
- Customisable data augmentation pipeline without compromising performance
- Deployment on embedded platforms using TensorRT
- Switchable neural networks (e.g., changing VGG to MobileNet for minimal memory consumption)
- High-performance training using multiple GPUs

**Note**: This project is under active development. Some TODOs are as follows:
- Pose Proposal Networks, ECCV 2018

## Custom Model Training

Training the model is implemented using TensorFlow. To run `train.py`, you would need to install packages, shown
in [requirements.txt](https://github.com/tensorlayer/openpose-plus/blob/master/requirements.txt), in your virtual environment (**Python 3**):

```bash
pip3 install -r requirements.txt
pip3 install pycocotools
```

`train.py` automatically download MSCOCO 2017 dataset into `dataset/coco17`.
The default model is VGG19 used in the OpenPose paper.
To customize the model, simply changing it in `models.py`.

You can use `train_config.py` to configure the training. `config.DATA.train_data` can be:
* `coco`: training data is COCO dataset only (default)
* `custom`: training data is your dataset specified by `config.DATA.your_xxx`
* `coco_and_custom`: training data is COCO and your dataset

`config.MODEL.name` can be:
* `vgg`: VGG19 version (default), slow
* `vggtiny`: VGG tiny version, faster
* `mobilenet`: MobileNet version, faster

Train your model by running:

```bash
python3 train.py
```

### Additional steps for training on Windows

There are a few extra steps to follow with Windows. Please make sure you have the following prerequisites installed:
* [git](https://git-scm.com/downloads)
* [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
* [wget](https://eternallybored.org/misc/wget/)

Download the wget executable and copy it into one of your folders in System path to use the wget command from anywhere. Use the `path` command in command line to find the folders. Paste the wget.exe in one of the folders given by `path`. An example folder is `C:\Windows`.

pycocotools is not supported by default on Windows. Use the pycocotools build for Windows at [here](https://github.com/philferriere/cocoapi). Instead of `pip install pycocotools`, using:
```bash
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

Visual C++ Build Tools are required by the build. Everything else is the same.


## Training using Multiple GPUs

The pose estimation neural network can take days to train.
To speed up the training, we support multiple GPU training while requiring
minimal changes in your code. We use Horovod to support training on GPUs that can spread across multiple machines.
You need to install the [OpenMPI](https://www.open-mpi.org/) in your machine.
We also provide an example script (`scripts/install-mpi.sh`) to help you go through the installation.
Once OpenMPI is installed, you can install Horovod python library as follows:

```bash
pip3 install horovod
```

We also have a Python binding for the engine. The current binding relies on
the external tf-pose-estimation project. We are working on providing the Python binding for our high-performance
C++ implementation. For now, to enable the binding, please build C++ library for post processing by:

```bash
./scripts/install-pafprocess.sh
```

See [tf-pose](https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess) for details.

## Live Camera Example

You can look at the examples in the `examples` folder to see how to use the inference C++ APIs.
Running `./scripts/live-camera.sh` will give you a quick review of how it works.

<!---
## 5. Inference

In this project, input images are RGB with 0~1.
Runs `train.py`, it will automatically download the default VGG19-based model from [here](https://github.com/tensorlayer/pretrained-models), and use it for inferencing.
The performance of pre-trained model is as follow:


|                  | Speed | AP | xxx |
|------------------|-------|----|-----|
| VGG19            | xx    | xx | xx  |
| Residual Squeeze | xx    | xx | xx  |

- Speed is tested on XXX

- We follow the [data format of official OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md)

To use the pre-trained models

-->


<!--
## 6. Evaluate a model

Runs `eval.py` for inference.

## 6. Customization

- Model : change `models.py`.
- Data augmentation : change `train.py`
- Train with your own data: ....
    1. prepare your data following MSCOCO format, you need to .
    2. concatenate the list of your own data JSON into ...
-->

## License

You can use the project code under a free [Apache 2.0 license](https://github.com/tensorlayer/tensorlayer/blob/master/LICENSE.rst) ONLY IF you:
- Cite the [TensorLayer paper](https://github.com/tensorlayer/tensorlayer#cite) and this project in your research article if you are an **academic user**.
- Acknowledge TensorLayer and this project in your project websites/articles if you are a **commercial user**.
