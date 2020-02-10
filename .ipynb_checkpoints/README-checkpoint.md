# Facial-Keypoint-Detection Project
---

###  1. Overview 


In this project, you’ll combine your knowledge of computer vision techniques and deep learning architectures to build a facial keypoint detection system. Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. The completed code should be able to look at any image, detect faces, and predict the locations of facial keypoints on each face.

![Example](https://github.com/nalbert9/Facial-Keypoint-Detection/blob/master/images/Obamas.png)

### 2. Project Struture

The project will contain four Python notebooks:

**models.py** : Neural Network file.

**Notebook 1** : Loading and Visualizing the Facial Keypoint Data.

**Notebook 2** : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints.

**Notebook 3** : Facial Keypoint Detection Using Haar Cascades and your Trained CNN.

**Notebook 4** : Fun Filters and Keypoint Uses.

### 3. Setup Up Instructions:

Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.

git clone https://github.com/lucastakara/Facial-Keypoint-Detection

Create (and activate) a new Anaconda environment (Python 3.6). Download via [Anaconda](https://www.anaconda.com/)

- **Linux or Mac**:
---
`conda create -n cv-nd python=3.6`

`source activate cv-nd`

- **Windows**:
----
- `conda create --name cv-nd python=3.6`

- `activate cv-nd`

Install PyTorch and torchvision; this should install the latest version of PyTorch;

- `conda install pytorch torchvision cudatoolkit=9.0 -c pytorch`

Install a few required pip packages, which are specified in the requirements text file (including OpenCV).

- `pip install -r requirements.txt`

