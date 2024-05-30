[![Repository License](https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg)](LICENSE)

# Image Matting with A Closed Form Solution 

Image Matting is a crucial process in accurately estimating the foreground object in images and videos. It finds extensive applications in various domains, particularly in film production for creating visual effects.

## Overview
This project implements the "A Closed Form Solution to Natural Image Matting" method proposed by A. Levin, D. Lischinski, and Y. Weiss. 
<!-- The method was presented at the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) in June 2006, New York. -->

<!-- The paper describing the method can be found [here](https://people.csail.mit.edu/alevin/papers/Matting-Levin-Lischinski-Weiss-CVPR06.pdf). -->
A detailed report of our work will be added soon...

<!-- ## Purpose
The purpose of this project is to provide a Python implementation of the image matting technique . By implementing this method, users can accurately estimate foreground objects in images and videos, which can be beneficial for various image and video editing applications. -->

## Online Demo
An online demo of this work is available [here](https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000489). Users can test the image matting algorithm with their own images through this interactive demo. There are some examples available on the website, or simply upload an image and the scribbles to the demo interface and obtain the results of foreground and background estimation.



## Installation & Running
1. Clone the repository
```bash
git clone https://github.com/reproductible-research/image-matting-with-a-closed-form-solution.git
cd image-matting-with-a-closed-form-solution
```
2. Set up a conda environment with Python 3.9
```bash
conda create -n matting python=3.9 -y
conda activate matting
```
3. Install the required packages
```bash
pip install -r requirements.txt
```

4. Run the main script with the input image and scribbles image

```bash
python main.py input_image.png -s scribbles.png 
```

## Example
Here is an example of input image, scribbles image, and the obtained result:


| Original image                           | Scribbled image                           | Output alpha                             | 
|------------------------------------------|-------------------------------------------|------------------------------------------|
| ![Original image](input_image.png)   | ![Scribbled image](scribbles.png) | ![Output alpha](output.png) |



## Reference

- [Anat Levin, Dani Lischinski, and Yair Weiss, A closed-form solution to natural image matting, IEEE Transactions on Pattern Analysis and Machine Intelligence, 30 (2008), pp. 228– 242.](https://people.csail.mit.edu/alevin/papers/Matting-Levin-Lischinski-Weiss-CVPR06.pdf).
