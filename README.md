# EDM-Craters
This repository contains the code from the paper “Automated characterization of WEDM single craters through CNN based object detection” presented in the conference ICPE 2022 (Nara, Japan).

Abstract of the paper:

“The identification and characterization of single craters are critical for the understanding and optimization of the wire electrical discharge machining (WEDM) process. Recent efforts have been made to study the influence that process parameters have on the geometry of the craters. These efforts collect geometrical data from the single craters through microscope imaging and manual labeling, a method that is time-consuming and labor-intensive. In this work, an automated crater identification and characterization approach based on state-of-the-art object detection algorithms is presented. In particular, the model You Only Look Once (YOLO) – a convolutional neural network-based object detection technique – is used to fit tight bounding boxes enclosing the craters of superficial microscope images. In addition, the model Detectron2 is used for instance segmentation of individual crops of the craters. The models are trained on a custom-made database of microscope images of WEDM single craters. The geometrical characteristics of the single craters are extracted from the segmentation masks and tight bounding boxes."

## What can you find in this repository?

In this repository you will be able to find all the code that was used to develop the paper, as well as an application to obtain all the geometrical information of the craters from a superficial microscope image.

## Describing the process

The following figure describe the inference process.

![image](https://user-images.githubusercontent.com/48300381/199795208-efd46a21-6ce9-47b0-854d-74be77620d70.png)


The inference process consists of the following steps:

[TODO]
