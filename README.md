## rsna-screening-mammography-breast-cancer-detection
## 1st place score achieved.
![rsna_breast_submission](https://github.com/bishnarender/rsna-screening-mammography-breast-cancer-detection/assets/49610834/e897c0ab-6457-4c37-9885-294bf28210ab)

-----

### Start 
For better understanding of project, read the files in the following order:
1. eda_1.ipynb 
2. prepair_classification_dataset.ipynb
3. prepair_roi_det_dataset.ipynb
4. train_yolox.ipynb
5. cv_split.ipynb
6. train_exp.ipynb
7. rsna-breast-submission.ipynb

Instead of original data, certain external datasets have been used.

4-folds splits have been used instead of 5, probably because the number of positives is not sufficient (0.02 %). During inference I did not use the fold_0 due to poor metrics. And obtained the score with ensemble of fold_1, fold_2 and fold_3 weights.

'stage1_process_func' (in prepair_classification_dataset.ipynb) appends the metadata from dcm files to the corresponding images present in original "train.csv" by creating new columns.

During the visual data analysis it was noticed that there is a large variation in the arrangement of the object in the images. In addition, some objects occupy only a small part of the image. ROI cropping was performed since it effectively helped keeping more texture/detail given a fixed resolution. 

[Remek Kinas](https://www.kaggle.com/remekkinas/). annotated about 500 images in a human-in-the-loop technique. Yolox has been trained on this data for ROI. Human-in-the-loop technique involves the process of manually labeling or marking specific objects or regions of interest within the images.

#### YOLOX
-----
YOLOX starts with a backbone network, such as a variant of the popular backbone network, Darknet or CSPDarknet. The backbone network extracts feature maps from the input image at multiple scales using convolutional layers and downsample operations.

![yolo_x](https://github.com/bishnarender/rsna-screening-mammography-breast-cancer-detection/assets/49610834/4733dbb6-18f4-4b26-b717-1453426ddef4)
[Reference](https://medium.com/mlearning-ai/yolox-explanation-how-does-yolox-work-3e5c89f2bf78/)

The diagram is stating that the input into both the YOLOv3 head and the YOLOX head is the 3 outputs from the FPN (darknet) backbone at three different scales — 1024, 512, 256 channels. The difference between the two heads is that YOLOv3 uses a coupled head and YOLOX uses a decoupled head. So, the output of YOLOX is actually 3 tensors each holding different information instead of 1 massive tensor with all information.

The three tensors YOLOX outputs hold the same information as the massive tensor YOLOv3 outputs:
Cls: The class of each bounding box.
Reg: The 4 parts to the bounding box (x, y, h, w).
IoU (Obj): For some reason, the authors use IoU instead of Obj, but this output is just how confident is the network that there’s an object in the bounding box (objectness).

Just like with the original output, each “pixel” in the height and width of the output is a different bounding box prediction. So, there are H*W different predictions.

The outputs listed above are only for a single output in the FPN. Remember there are three outputs from the FPN which are fed into the heads of YOLOv3 and YOLOX. This means there are actually three different outputs from each of the heads instead of 1. So, the output of YOLOv3 is actually (3×H×W×features) and the output of YOLOX is actually 3 of each of the Cls, Reg, and IoU (obj) outputs making 9 totals outputs.

YOLO is the anchor-free algorithm conduced together with advanced detection techniques, i.e a decoupled head and SimOTA. This model achieves higher performance than the YOLOv4/v5. YOLOX also decouples the YOLO detection head into separate feature channels for box coordinate regression and object classification. This leads to improved convergence speed and model accuracy.

A feature pyramid network extracts information from an image at different stages and with different aspects (different widths and heights). To do this with Darknet, we take transition states from the model and use those as several outputs instead of a single output coming from the end of the network.

OTA: Optimal Transport Assignment for Object Detection, solves the issue by considering the label assignment task as a Transport Problem. Simplified OTA or simOTA is the redesigned Optimal Transport Assignment strategy. 


#### What is an anchor?
-----
An anchor refers to a predefined bounding box of a specific size and aspect ratio. During training, the model adjusts the predicted bounding box coordinates relative to the anchor box. Each anchor box is associated with a set of parameters that the model learns to predict. These parameters include offsets for the box's center coordinates, width, height, objectness score, and class probabilities. Each point in the feature map is corresponding to a set of anchor boxes.

![anchoxboxes](https://github.com/bishnarender/rsna-screening-mammography-breast-cancer-detection/assets/49610834/ddbf6836-e25d-4b39-b794-ab697643885d)

During inference, the model generates a set of predicted bounding boxes based on the anchor boxes and their corresponding parameters. These predicted boxes are then filtered and refined using techniques like non-maximum suppression (NMS) to obtain the final detections. The way nonmax suppression removes bounding boxes with a high overlap is by using the IoU score between overlapping bounding boxes.

YOLOX, does not use predefined anchor boxes. Instead of "anchor box", yolox uses "anchor point". Anchor point is slightly different thing. An anchor point is an offset to move the x,y location of a prediction while the "anchor box" is a predefined box that is used as an offset for the h, w parts of a prediction. 

Anchor-free methods try to localize the objects directly without using the boxes as proposals but using "centers" or "key points". 

In key-points proposal, several specified (or self-learning) key points are located throughout the object by keypoint-based detectors. The spatial extent of the object is obtained through these clusters of key points.

![yolox-keypoint](https://github.com/bishnarender/rsna-screening-mammography-breast-cancer-detection/assets/49610834/c44e16be-bf12-47d8-b077-97aaf9371a74)
[Reference](https://learnopencv.com/yolox-object-detector-paper-explanation-and-custom-training/)

Whereas, center-based detectors find positives in the center (or center region), then predict three/four distances from positives to the boundary.

Actual YOLOX architecture:
![yolox](https://github.com/bishnarender/rsna-screening-mammography-breast-cancer-detection/assets/49610834/db637041-5543-4378-b619-d06cfacd122c)

Output [3549,6] denotes you have 3549 bboxes and each has 4 bbox values (x_center, y_center, height, width),  1 confidence score (or objectness score) and 1 probability for our positive-class.

-----

### ROI Extraction

During ROI extraction, images are manually resized instead of relying on "yolox" for resizing. 

![roi_extraction_filtering](https://github.com/bishnarender/rsna-screening-mammography-breast-cancer-detection/assets/49610834/9eb1eae5-44c2-4405-9e37-22f96b2c6cce)

#### What is Binarization or Thresholding?
-----
Image thresholding is a technique used in image processing and computer vision to convert a grayscale or color image into a binary image, where each pixel is assigned either a black (0) or white (255) value based on a predefined threshold. The thresholding process separates objects or regions of interest from the background by comparing pixel intensities to the threshold value.

Pixels with intensities below the threshold are considered part of the background and set to black, while pixels with intensities equal to or above the threshold are considered part of the foreground and set to white.
Thresholding can be used for various purposes, such as image segmentation, object detection, and feature extraction.

Otsu's method determines an optimal global threshold value from the image histogram. Consider an image with only two distinct image values (bimodal image), where the histogram would only consist of two peaks. A good threshold would be in the middle of those two values. 

Here, image is first filtered with a gaussian kernel to remove the noise, then Otsu thresholding is applied.

#### What is windowing in relation to dicom files?
-----
Windowing, also known as grey-level mapping, contrast stretching, histogram modification or contrast enhancement is the process in which the CT image greyscale component of an image is manipulated via the CT numbers; doing this will change the appearance of the picture to highlight particular structures. The brightness of the image is adjusted via the window level. The contrast is adjusted via the window width.



-----

### Training

During training, Upsample (simply copying samples) is performed using a "custom batch sampler" so that each batch has at least one pos case.

File changed in timm (v0.9.2) is only "timm > data > loader.py".

Demo of applied augmentation:
![augmentation](https://github.com/bishnarender/rsna-screening-mammography-breast-cancer-detection/assets/49610834/00096e37-9115-451d-86f6-3fa0edc9be75)

![model](https://github.com/bishnarender/rsna-screening-mammography-breast-cancer-detection/assets/49610834/e837a9b8-4915-414d-819f-f7ca3960c68a)

#### What is Average Recall (AR)?
-----
AR@[.5:.95] corresponds to the average of recalls for IoU thresholds 0.5 to 0.95 with a step size of 0.05. 

#### What is Average Precision (AP)?
-----
General definition for the Average Precision (AP) is finding the area under the precision-recall curve above. The following AP formula to calculate value of AU-PR curve. AP = sum_over_n{(Rn-Rn-1)Pn} where n is the corresponding threshold.

First, recall and precision (at a certain IoU threshold) is calculated for every ground truth available. So, we have as many recalls and precisions as the ground truths are. Then from available recalls instead of using exact precision value P(R) corresponding to certain recall R, a maximum precision value is chosen based on interpolation whose recall value is greater than R. Finally from these "precision, recall" pairs the average precision is calculated.

AP is calculated individually for each class.

TP: if IoU ≥ 0.5, classify the object detections as True Positive.
FP: if IoU < 0.5.
TN: when ground truth is not present and also model failed to detect an object.
FN: when ground truth is present and model failed to detect an object.

AP@[.5] corresponds to the AP for IOU 0.5.
AP@[.5:.95] corresponds to the average of APs for IoU thresholds 0.5 to 0.95 with a step size of 0.05. 

[Reference](https://towardsdatascience.com/on-object-detection-metrics-with-worked-example-216f173ed31e/)

#### What is mean Average Precision (mAP)?
-----
![map_r](https://github.com/bishnarender/rsna-screening-mammography-breast-cancer-detection/assets/49610834/1d1130dc-e48b-4b4c-b5f7-e3947471f81b)
