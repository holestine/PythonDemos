# Python Demos
Collection of Python sample code mostly in the area of AI and Deep Learning.

## Prepare Environment
``` bash
conda create --name demo python=3.12 -y
conda activate demo
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
python -m pip install -U pip
pip install -r requirements.txt
```

## Project Descriptions
### [chatbot](chatbot)
Web scraping and chatgpt are used for document understanding.

### [models](models)
Various model architectures are applied to MNIST, CIFAR10 and CIFAR100.

### [tracking](tracking)
YOLO is used with the Hungarian algorithm to track objects.

### [yolo](yolo)
YOLO fine tuning and SuperVision annotation.

### [speech_recognition](speech_recognition)
Udacity speech recognition project.

### [word2vec](word2vec)
Demonstration of word vectors.

### [fraud](fraud)
Example of regression model to detect credit card fraud.
