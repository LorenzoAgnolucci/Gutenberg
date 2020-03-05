# Gutenberg

## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)


## About The Project

```Gutenberg``` is a pipeline for training a neural network in segmenting and recognising frequent words in early printed books, in particular we focus on Gutenberg’s Bible.
First we describe the creation of a dataset, containing only the Genesis book, using dynamic programming techniques and projection profiles with the aid of a line-by-line transcription.
Then we leverage this dataset to train a Mask R-CNN model in order to generalize word segmentation and detection in pages where transcription is not available.

For more information, read the [paper](paper.pdf) located in the repo root.

### Built With

* [Python](https://www.python.org/)
* [OpenCV](https://opencv.org/)
* [DTAiDistance](https://github.com/wannesm/dtaidistance)



## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

The project provide a ```Pipfile``` file that can be managed with [pipenv](https://github.com/pypa/pipenv).
```pipenv``` installation is **strongly encouraged** in order to avoid dependency/reproducibility problems.

* pipenv
```sh
pip3 install pipenv
```

### Installation
 
1. Clone the repo
```sh
git clone https://gitlab.com/turboillegali/gutenberg
```
2. Install Python dependencies
```sh
pipenv install
```


## Usage

Every file under ```src/``` is executable. If you have ```pipenv``` installed, executing them
so that the python interpreter can find the project dependencies is as easy as running ```pipenv run $file```.

Here's a brief description of each and every file under the ```src/``` directory:

* ```preprocessing.py```: Image preprocessing (e.g. skew correction and cropping).
* ```caput.py```: Caput detection
* ```punctuation.py```: Punctuation detection (e.g. long accents, periods, ...)
* ```lines.py```: Line and column segmentation
* ```words.py```: Word segmentation (requires output from ```lines.py```)
* ```coco_dataset.py```: [COCO](http://cocodataset.org/)-like dataset building. Requires outpput from ```words.py```
* ```coco_dataset_chunks.py```: Variant of ```coco_dataset.py``` where instead of whole pages the images are split in chunks of N lines each (by default N = 7).

The dataset created with the previous steps can be used with the neural network available in the ```WALL_E_Net.ipnyb```
