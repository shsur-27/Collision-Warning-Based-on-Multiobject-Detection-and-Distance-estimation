# Collision Warning using YOLOv3-tiny and Distance estimation on tensorflow 

About
-----
> Implementation of collision warning system based on multi-object detection and distance estimation using Tensorflow
 
Convert Weight file
-------------------
> Download the official weights Or you can put your own weight file in weights folder
For YOLOv3-Tiny:

curl https://pjreddie.com/media/files/yolov3-tiny.weights > ./weights/yolov3-tiny.weights

Usage
python3 convert_weights.py [-h] [--tiny]

-h: Show help message and exit.
--tiny: Convert tiny_weights from "./weights/yolov3-tiny.weights". Default is to convert weights from "./weights/yolov3.weights".

Installation
------------
For libraries

Run "pip3 install -r requirements.txt"

If there are any problem to install requirements.txt you have to install all libraries separartely by following commands:

1) for tensorflow =>1.13.1
   "sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu==1.13.1+nv19.3"
 
 where v42 is jetpack version 4.2.2 and nv19.3 is monthly NVIDIA container version of TensorFlow
 
   
2) for other libraries/packages you can simply use "pip3 install package_name" 



Usage
-----

`python3 detect.py [-h] [--tiny] {video,image} iou confidence path`
* `-h`: Show help message and exit.
* `--tiny`: Enable tiny model mode.
* `{video, image}`: Detection mode (for file `path`).
* `iou`: IoU threshold between [0.0, 1.0].
* `confidence`: Confidence threshold between [0.0, 1.0].
* `path`: Path to file we want to do detection on.

Example
-------

we can run detection for either an images:
```
python3 detect.py image 0.5 0.5 ./data/images/example.jpg
```

Or for a video:
```
python3 detect.py video 0.5 0.5 data/videos/train.mp4
```

NOTE: IF YOU ARE USING PYTHON 2 VERSION THEN SIMPLY PUT python INSTEAD OF python3 IN COMMAND LINE
 
References
----------

[object detection implementation](https://github.com/kcosta42/Tensorflow-YOLOv3)

[Video reference](https://www.youtube.com/watch?v=o3Ky_EdHVrA)
