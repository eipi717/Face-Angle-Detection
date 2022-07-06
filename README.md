# Face Angle Detection
### Introduction
The repo investigate the face angle, or face shape of different gender.

---
### Implementation
1. Each detected faces in the image will be cropped out
2. It will automatically detect the face angle, by finding the slope of two side of the face (see figure below).
3. The face angle will be calculated using the equation below,
$$ \theta = \arctan (\frac{m_{2} - m_{1}}{1 + m_{1}m_{2}}) \cdot \frac{180}{\pi}  $$
where m1 and m2 represent the slopes

   
### Dataset used
In this repo, [UTKFace] is used for testing, it includes around 20k images

---

### Model used
The landmarks model can be downloaded from [here]

----
### Required packages

Install the required package by the following command
```
pip install -r requirements.txt
```
---
### Additional information
Please change the directories name, if needed.

[UTKFace]: https://susanqq.github.io/UTKFace/
[here]: https://github.com/davisking/dlib-models
