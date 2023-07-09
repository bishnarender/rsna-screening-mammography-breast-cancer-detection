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

<b>YOLOX</b>
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


<b>What is an anchor?</b>
-----
An anchor refers to a predefined bounding box of a specific size and aspect ratio. During training, the model adjusts the predicted bounding box coordinates relative to the anchor box. Each anchor box is associated with a set of parameters that the model learns to predict. These parameters include offsets for the box's center coordinates, width, height, objectness score, and class probabilities. Each point in the feature map is corresponding to a set of anchor boxes.

![anchoxboxes](https://github.com/bishnarender/rsna-screening-mammography-breast-cancer-detection/assets/49610834/ddbf6836-e25d-4b39-b794-ab697643885d)

During inference, the model generates a set of predicted bounding boxes based on the anchor boxes and their corresponding parameters. These predicted boxes are then filtered and refined using techniques like non-maximum suppression (NMS) to obtain the final detections. The way nonmax suppression removes bounding boxes with a high overlap is by using the IoU score between overlapping bounding boxes.

YOLOX, does not use predefined anchor boxes, it directly predicts bounding box coordinates and sizes without relying on anchor-related parameters. To directly predict a bounding box, YOLOX uses a decoupled head. Anchor-free methods try to localize the objects directly without using the boxes as proposals but using "centers" or "key points". 

In key-points proposal, several specified (or self-learning) key points are located throughout the object by keypoint-based detectors. The spatial extent of the object is obtained through these clusters of key points.

![yolox-keypoint](https://github.com/bishnarender/rsna-screening-mammography-breast-cancer-detection/assets/49610834/c44e16be-bf12-47d8-b077-97aaf9371a74)
[Reference](https://learnopencv.com/yolox-object-detector-paper-explanation-and-custom-training/)

Whereas, center-based detectors find positives in the center (or center region), then predict three/four distances from positives to the boundary.

Instead of "anchor box", yolox uses "anchor point". Anchor point is slightly different thing. An anchor point is an offset to move the x,y location of a prediction while the "anchor box" is a predefined box that is used as an offset for the w, h parts of a prediction. 

Actual YOLOX architecture:

During ROI extraction, images are manually resized instead of relying on "yolox" for resizing. 



