# How to train a custom object detection model with the Tensorflow Object Detection API
(ReadME inspired by [EdjeElectronics](https://github.com/EdjeElectronics))

![Example Output](doc/output.png)

## Introduction

## Steps:

### 1. Installation

You can install the TensorFlow Object Detection API either with Python Package Installer (pip) or Docker, an open-source platform for deploying and managing containerized applications. For running the Tensorflow Object Detection API locally, Docker is recommended. If you aren't familiar with Docker though, it might be easier to install it using pip.

First clone the master branch of the Tensorflow Models repository:

```bash
git clone https://github.com/tensorflow/models.git
```

#### Docker Installation

```bash
# From the root of the git repository
docker build -f research/object_detection/dockerfiles/tf1/Dockerfile -t od .
docker run -it od
```

#### Python Package Installation

```bash
cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf1/setup.py .
python -m pip install .
```

> Note: The *.proto designating all files does not work protobuf version 3.5 and higher. If you are using version 3.5, you have to go through each file individually. To make this easier, I created a python script that loops through a directory and converts all proto files one at a time.

```python
import os
import sys
args = sys.argv
directory = args[1]
protoc_path = args[2]
for file in os.listdir(directory):
    if file.endswith(".proto"):
        os.system(protoc_path+" "+directory+"/"+file+" --python_out=.")
```

```
python use_protobuf.py <path to directory> <path to protoc file>
```

To test the installation run:

```bash
# Test the installation.
python object_detection/builders/model_builder_tf1_test.py
```

If everything installed correctly you should see something like:

```bash
Running tests under Python 3.6.9: /usr/bin/python3
[ RUN      ] ModelBuilderTF1Test.test_create_context_rcnn_from_config_with_params(True)
[       OK ] ModelBuilderTF1Test.test_create_context_rcnn_from_config_with_params(True)
[ RUN      ] ModelBuilderTF1Test.test_create_context_rcnn_from_config_with_params(False)
[       OK ] ModelBuilderTF1Test.test_create_context_rcnn_from_config_with_params(False)
[ RUN      ] ModelBuilderTF1Test.test_create_experimental_model
[       OK ] ModelBuilderTF1Test.test_create_experimental_model
[ RUN      ] ModelBuilderTF1Test.test_create_faster_rcnn_from_config_with_crop_feature(True)
[       OK ] ModelBuilderTF1Test.test_create_faster_rcnn_from_config_with_crop_feature(True)
[ RUN      ] ModelBuilderTF1Test.test_create_faster_rcnn_from_config_with_crop_feature(False)
[       OK ] ModelBuilderTF1Test.test_create_faster_rcnn_from_config_with_crop_feature(False)
[ RUN      ] ModelBuilderTF1Test.test_create_faster_rcnn_model_from_config_with_example_miner
[       OK ] ModelBuilderTF1Test.test_create_faster_rcnn_model_from_config_with_example_miner
[ RUN      ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_faster_rcnn_with_matmul
[       OK ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_faster_rcnn_with_matmul
[ RUN      ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_faster_rcnn_without_matmul
[       OK ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_faster_rcnn_without_matmul
[ RUN      ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_mask_rcnn_with_matmul
[       OK ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_mask_rcnn_with_matmul
[ RUN      ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_mask_rcnn_without_matmul
[       OK ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_mask_rcnn_without_matmul
[ RUN      ] ModelBuilderTF1Test.test_create_rfcn_model_from_config
[       OK ] ModelBuilderTF1Test.test_create_rfcn_model_from_config
[ RUN      ] ModelBuilderTF1Test.test_create_ssd_fpn_model_from_config
[       OK ] ModelBuilderTF1Test.test_create_ssd_fpn_model_from_config
[ RUN      ] ModelBuilderTF1Test.test_create_ssd_models_from_config
[       OK ] ModelBuilderTF1Test.test_create_ssd_models_from_config
[ RUN      ] ModelBuilderTF1Test.test_invalid_faster_rcnn_batchnorm_update
[       OK ] ModelBuilderTF1Test.test_invalid_faster_rcnn_batchnorm_update
[ RUN      ] ModelBuilderTF1Test.test_invalid_first_stage_nms_iou_threshold
[       OK ] ModelBuilderTF1Test.test_invalid_first_stage_nms_iou_threshold
[ RUN      ] ModelBuilderTF1Test.test_invalid_model_config_proto
[       OK ] ModelBuilderTF1Test.test_invalid_model_config_proto
[ RUN      ] ModelBuilderTF1Test.test_invalid_second_stage_batch_size
[       OK ] ModelBuilderTF1Test.test_invalid_second_stage_batch_size
[ RUN      ] ModelBuilderTF1Test.test_session
[  SKIPPED ] ModelBuilderTF1Test.test_session
[ RUN      ] ModelBuilderTF1Test.test_unknown_faster_rcnn_feature_extractor
[       OK ] ModelBuilderTF1Test.test_unknown_faster_rcnn_feature_extractor
[ RUN      ] ModelBuilderTF1Test.test_unknown_meta_architecture
[       OK ] ModelBuilderTF1Test.test_unknown_meta_architecture
[ RUN      ] ModelBuilderTF1Test.test_unknown_ssd_feature_extractor
[       OK ] ModelBuilderTF1Test.test_unknown_ssd_feature_extractor
----------------------------------------------------------------------
Ran 21 tests in 0.163s

OK (skipped=1)
```

### 2. Gathering data

Now that the Tensorflow Object Detection API is ready to go, we need to gather the images needed for training. 

To train a robust model, we need lots of pictures that should vary as much as possible from each other. That means that they should have different lighting conditions, different backgrounds and lots of random objects in them.

You can either take the pictures yourself or you can download them from the internet. For my microcontroller detector, I have four different objects I want to detect (Arduino Nano, ESP8266, Raspberry Pi 3, Heltect ESP32 Lora).

I took about 25 pictures of each individual microcontroller and 25 pictures containing multiple microcontrollers using my smartphone. After taking the pictures make sure to transform them to a resolution suitable for training (I used 800x600).

![](doc/image_gallery.png)

You can use the [resize_images script](resize_images.py) to resize the image to the wanted resolutions.

```bash
python resize_images.py -d images/ -s 800 600
```

After you have all the images move about 80% to the object_detection/images/train directory and the other 20% to the object_detection/images/test directory. Make sure that the images in both directories have a good variety of classes.

### 3. Labeling data

With all the pictures gathered, we come to the next step - labeling the data. Labeling is the process of drawing bounding boxes around the desired objects.

LabelImg is a great tool for creating a object detection data-set.

[LabelImg GitHub](https://github.com/tzutalin/labelImg)

[LabelImg Download](https://www.dropbox.com/s/tq7zfrcwl44vxan/windows_v1.6.0.zip?dl=1)

Download and install LabelImg. Then point it to your images/train and images/test directories, and draw a box around each object in each image.

![label images](doc/label_image.png)

LabelImg supports two formats, PascalVOC and Yolo. For this tutorial make sure to select PascalVOC. LabelImg saves a xml file containing the label data for each image. These files will be used to create a tfrecord file, which can be used to train the model.

### 4. Generating Training data

With the images labeled, we need to create TFRecords that can be served as input data for training of the object detector. In order to create the TFRecords we will use two scripts from [Dat Tran’s raccoon detector](https://github.com/datitran/raccoon_dataset). Namely the xml_to_csv.py and generate_tfrecord.py files.

After downloading both scripts we can first of change the main method in the   xml_to_csv file so we can transform the created xml files to csv correctly.

```python
# Old:
def main():
    image_path = os.path.join(os.getcwd(), 'annotations')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('raccoon_labels.csv', index=None)
    print('Successfully converted xml to csv.')
# New:
def main():
    for folder in ['train', 'test']:
        image_path = os.path.join(os.getcwd(), ('images/' + folder))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(('images/'+folder+'_labels.csv'), index=None)
	print('Successfully converted xml to csv.')
```

Now we can transform our xml files to csvs by opening the command line and typing:

```bash
python xml_to_csv.py
```

These creates two files in the images directory. One called test_labels.csv and another one called train_labels.csv.

Next, open the generate_tfrecord.py file and replace the labelmap inside the class_text_to_int method with your own label map.

Old:
```python
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'basketball':
        return 1
    elif row_label == 'shirt':
        return 2
    elif row_label == 'shoe':
        return 3
    else:
        return None
```

New:
```python
def class_text_to_int(row_label):
    if row_label == 'Raspberry_Pi_3':
        return 1
    elif row_label == 'Arduino_Nano':
        return 2
    elif row_label == 'ESP8266':
        return 3
    elif row_label == 'Heltec_ESP32_Lora':
        return 4
    else:
        return None
```

Now the TFRecords can be generated by typing:

```bash
python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
```

These two commands generate a train.record and a test.record file which can be used to train our object detector.

### 5. Getting ready for training

The last thing we need to do before training is to create a label map and a training configuration file.

#### 5.1 Creating a label map

The label map maps an id to a name. We will put it in a folder called training, which is located in the object_detection directory. The labelmap for my detector can be seen below.

```python
item {
    id: 1
    name: 'Raspberry_Pi_3'
}
item {
    id: 2
    name: 'Arduino_Nano'
}
item {
    id: 3
    name: 'ESP8266'
}
item {
    id: 4
    name: 'Heltec_ESP32_Lora'
}
```

The id number of each item should match the id of specified in the generate_tfrecord.py file.

#### 5.2 Creating the training configuration

Lastly we need to create a training configuration file. As a base model I will use faster_rcnn_inception, which just like a lot of other models can be downloaded from the [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

Because we are using a faster_rcnn_inception model we can choose one of its predefined configurations. We will use faster_rcnn_inception_v2_pets.config located inside the models/research/object_detection/samples/configs folder.

Copy the config file to the training directory- Then open it inside a text editor and make the following changes:

* Line 9: change the number of classes to number of objects you want to detect (4 in my case)

* Line 106: change fine_tune_checkpoint to the path of the model.ckpt file:

    * ```fine_tune_checkpoint: "C:/Users/Gilbert/Downloads/Other/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"```

* Line 123: change input_path to the path of the train.records file:

    * ```input_path: "C:/Users/Gilbert/Downloads/Other/models/research/object_detection/train.record"```

* Line 135: change input_path to the path of the test.records file:

    * ```input_path: "C:/Users/Gilbert/Downloads/Other/models/research/object_detection/test.record"```

* Line 125 and 137: change label_map_path to the path of the label map:

    * ```label_map_path: "C:/Users/Gilbert/Downloads/Other/models/research/object_detection/training/labelmap.pbtxt"```

* Line 130: change num_example to the number of images in your test folder.

### 6. Training the model

To train the model execute the following command in the command line:

```bash
python model_main.py --logtostderr --model_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```

If everything was setup correctly the training should begin shortly and you should see something like the following:

![training the model](doc/training_model.png)

Every few minutes the current loss gets logged to Tensorboard. Open Tensorboard by opening a second command line, navigating to the object_detection folder and typing:

```tensorboard --logdir=training```

This will open a webpage at localhost:6006.

![monitor training](doc/monitor_training.png)

The training scrips saves checkpoints about every five minutes. Train the model until it reaches a satisfying loss then you can terminat the training process by pressing Ctrl+C.

### 7. Exporting the inference graph

Now that we have a trained model we need to generate an inference graph, which can be used to run the model. For doing so we need to first of find out the highest saved step number. For this, we need to navigate to the training directory and look for the model.ckpt file with the biggest index.

Then we can create the inference graph by typing the following command in the command line.

```bash
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```

XXXX represents the highest number.

### 8. Using the model for inference

After training the model it can be used in many ways. For examples on how to use the model check out my other repositories.

* [Inference with Tensorflow 1.x](https://github.com/TannerGilbert/Tutorials/tree/master/Tensorflow%20Object%20Detection)

## Appendix

### Common Errors

The Tensorflow Object Detection API has lots of painful error that can be quite hard to solve. In this appendix section you can find the errors I encountered and how to solve them.

#### 1. ModuleNotFoundError: No module named 'object_detection', ImportError : cannot import name 'string_int_label_map_pb2, No module named nets

These errors occur when the object detection API wasn't installed correctly. Make sure to follow [my installation guide](https://gilberttanner.com/blog/installing-the-tensorflow-object-detection-api) correctly.


#### 2. AttributeError: module 'tensorflow' has no attribute 'app', AttributeError: module 'tensorflow' has no attribute 'contrib'

Training models with Tensorflow 2.0 insn't supported yet. I will update the repository as soon as it is supported.

### Common Questions

#### 1. How do I extract the images inside the bounding boxes?

```python
output_directory = 'some dir'

# get label and coordinates of detected objects
output = []
for index, score in enumerate(output_dict['detection_scores']):
    label = category_index[output_dict['detection_classes'][index]]['name']
    ymin, xmin, ymax, xmax = output_dict['detection_boxes'][index]
    output.append((label, int(xmin * image_width), int(ymin * image_height), int(xmax * image_width), int(ymax * image_height)))

# Save images and labels
for l, x_min, y_min, x_max, y_max in output:
    array = cv2.cvtColor(np.array(image_show), cv2.COLOR_RGB2BGR)
    image = Image.fromarray(array)
    cropped_img = image.crop((x_min, y_min, x_max, y_max))
    file_path = output_directory+'/images/'+str(len(df))+'.jpg'
    cropped_img.save(file_path, "JPEG", icc_profile=cropped_img.info.get('icc_profile'))
    df.loc[len(df)] = [datetime.datetime.now(), file_path]
    df.to_csv(output_directory+'/results.csv', index=None
``` 

#### 2. How do I host a model?

There are multiple ways to host a model. You can create a Restful API with [Tensorflow Serving](https://github.com/tensorflow/serving) or by creating your own websites. You can also integrate the model into a website by transforming your model to [Tensorflow Lite](https://www.tensorflow.org/lite/convert). 

## Contribution

Anyone is welcome to contribute to this repository, however, if you decide to do so I would appreciate it if you take a moment and review the [guidelines](./.github/CONTRIBUTING.md).

## Author
 **Gilbert Tanner**
 
## Support me

<a href="https://www.buymeacoffee.com/gilberttanner" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details