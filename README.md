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

4-folds splits have been used instead of 5, probably because the number of positives is not sufficient (0.02 %). During inference I did not use the fold_0 due to poor metrics. And obtained the score with ensemble of fold_1, fold_2 and fold_3 weights. 

'stage1_process_func' (in prepair_classification_dataset.ipynb) appends the metadata from dcm files to the corresponding images present in original "train.csv" by creating new columns.

During the visual data analysis it was noticed that there is a large variation in the arrangement of the object in the images. In addition, some objects occupy only a small part of the image. ROI cropping was performed since it effectively helped keeping more texture/detail given a fixed resolution. 

[Remek Kinas](https://www.kaggle.com/remekkinas/). annotated about 500 images in a human-in-the-loop technique. Yolox has been trained on this data for ROI. Human-in-the-loop technique involves the process of manually labeling or marking specific objects or regions of interest within the images.

During ROI extraction, images are manually resized instead of relying on "yolox" for resizing. 

<b>Hello</b>


