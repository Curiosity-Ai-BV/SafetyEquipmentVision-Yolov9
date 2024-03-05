# SafetyEquipmentVision

Train your own version of YoloV9 for your dataset. Here is an example on how to do that using a Roboflow training set
https://colab.research.google.com/drive/1gQHUBCZ9BWHOW_zS9XX7IwZxrPl0YsSd?usp=sharing
Be sure to set your collab notebook to use the free Nvidia t4 option!
The colab notebook is also available in this repo

Download the model after running your model training and insert into the model directory
(download model)[link]

create environment and install depenencies
```
python -m venv venv
./venv/Scripts/activate
python -m pip install -r requirements.txts
```

Install yolov9
```bash
git clone https://github.com/WongKinYiu/yolov9.git
python -m pip install -r yolov9/requirements.txt
```

app_webcam.py - works identifying directly using your webcam
app_videoV1.py - works identifying helmets and vests in video's
