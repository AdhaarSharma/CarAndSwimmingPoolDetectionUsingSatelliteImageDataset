# RetinaNet Object Detection
This project aims to detects cars and swimming pools in satellite test image dataset. The deep learning model I used was Fizyr/Retinanet. I have hereby attached the results of my project in the google drive. Link given below:
https://drive.google.com/drive/folders/1SZ9rEcBs7u8t0z0eem8_yEVbAeSc_3Tc

#Installation
conda install tensorflow
conda install keras
Go to the your working directory and open Git Bash. Use git clone https://github.com/fizyr/keras-retinanet

#Dataset
https://www.kaggle.com/kbhartiya83/swimming-pool-and-car-detection
Download the kaggle dataset mentioned above in working directory.

#Preprocessing
* Prepare your dataset (both training and test sets) in the CSV format required for training and evaluation with keras-retinanet (build_dataset.py)
python build_dataset.py -l data/training_data/labels -i data/training_data/images -r data/training_data/train.csv -e data/training_data/test.csv -c data/training_data/classes.csv

#Predictions
* Generate predictions (class and bounding box coordinates) on new images and output them to CSV files (image_inference_write.py)
python train.py --freeze-backbone --random-transform --weights resnet50_coco_best_v2.1.0.h5 --weighted-average --batch-size 8 --steps 100 --epochs 30 csv data/training_data/train.csv data/training_data/classes.csv
* Generate and save copies of original test/inference images with bounding boxes drawn around detected objects (image_inference_print.py)
python keras_retinanet/bin/evaluate.py csv data/training_data/test.csv data/training_data/classes.csv resnet50_csv_30.h5 --convert-model

#Results
The Accuracy could have been improved if I used batch size of 64 or 32 but I used a batch size of 8 because of limitations of my GPU. I'd recommend you to try experimenting with batch size and threshold value during model training
