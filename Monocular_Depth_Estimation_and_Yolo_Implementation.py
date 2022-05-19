
## Initialization of Libraries
import cv2
import os
import numpy as np

## Selection of Camera device to be used
cam = cv2.VideoCapture(0)
confThreshold = 0.6
nmsThreshold = 0.5 #lower the more agressive and less boxes

## Folder Path for class names
folderpath = 'C:\\Users\\Lanz De Guzman\\source\\repos\\Monocular-Depth-Estimation-and-Yolo-Implementation\\models\\coco.names'
classNames = []
with open(folderpath, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# print(classNames)

# Monocular Depth Estimation Models 
path_model = "C:\\Users\\Lanz De Guzman\\source\\repos\\Monocular-Depth-Estimation-and-Yolo-Implementation\\models"
model_name = "\\model-f6b98070.onnx";
modelMDE = cv2.dnn.readNet(path_model + model_name)
modelMDE.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
modelMDE.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

print('Depth Estimation Model Initialization Successful')

"""

## Original Formula 

def depth_to_distance(depth):
    return -1.7 * depth + 2
  """

## Known Values through references and empirical data



## Equation from Seer V2
def depth_to_distance (depth):
 
       ## Referenced and Measured values 
       Pixelheight = 200
       Knownheight = 160 
       FocalLength = 12 
       RefPoint = 0.5035 

       KD = (Pixelheight*FocalLength)/(Knownheight*10)
       CD = (KD*RefPoint)/(depth)
       return (CD)


"""
## Based On Ratio and Proportion
def depth_to_distance (depth):

     ## RealDistance or Computed Distance based on ratio and proportion = KnownHeight(depth)/Pixel Height  
       Pixelheight = 249
       Knownheight = 160 
    
       CD = (Knownheight*depth)/(Pixelheight) 
       return (CD)
"""

"""
def FocalLength(Distance,RealHeight,PixelHeight):
    return  round((PixelHeight*Distance)/RealHeight)

def ComputedDistance(RealHeight,FocalLength,PixelHeight):
    return  (RealHeight*FocalLength)/(PixelHeight*10)
"""

## Yolo files initialization

modelConfiguration = 'C:\\Users\\Lanz De Guzman\\source\\repos\\Monocular-Depth-Estimation-and-Yolo-Implementation\\models\\yolov3.cfg'
modelWeight = 'C:\\Users\\Lanz De Guzman\\source\\repos\\Monocular-Depth-Estimation-and-Yolo-Implementation\\models\\yolov3.weights'
model = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeight)
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

print('Yolo Initialization Successful')

def findObjects(detection,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    center_Point = []

    for output in detection:
          for det in output:
              scores = det[5:]
              classId = np.argmax(scores)
              confidence = scores[classId]
              if confidence > confThreshold:
                  w,h = int(det[2]*wT) , int(det[3]*hT)
                  x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                  bbox.append([x,y,w,h])
                  classIds.append(classId)
                  confs.append(float(confidence))
              

    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    for i in indices:
        i = i
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', 
                    (x,y-15),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,0,255), 2)
        Xcord = (x+(x+w))/2
        Ycord = (y+(y+h))/2
        center_Point = (Xcord,Ycord)
        output_face = monocEstimation(center_Point,img)
        #print (f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%:', h)
        output_face = depth_to_distance(output_face)
        print (f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%:', output_face*100)
        cv2.putText(img,"Depth in cm: " + str(round(output_face,2)*100), (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255),2)
        cv2.circle(img,(int(Xcord),int(Ycord)), 3, (0,0,255), -1)
        
        

def monocEstimation(center_Point,img):

       blobb = cv2.dnn.blobFromImage(img,1/255.,(384,384),(123.675, 116.28, 103.53), True, False)

       modelMDE.setInput(blobb)
       monocOutput = modelMDE.forward()
    
       monocOutput = monocOutput [0,:,:]
       monocOutput = cv2.resize(monocOutput,(imgWidth, imgHeight))

       monocOutput = cv2.normalize(monocOutput, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

       output_face = monocOutput[int(center_Point[0]),int(center_Point[1])]

       cv2.imshow('Depth Map', monocOutput)
       return output_face


while True:
       success, img = cam.read()

       imgHeight, imgWidth, channels = img.shape
       img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

       #convert img to blob -- the only format the network accepts

       blob = cv2.dnn.blobFromImage(img, 1/255, (320,320), [0,0,0],1,crop =False)
       model.setInput(blob)

       layerNames = model.getLayerNames()
       outputNames = [layerNames[i-1] for i in model.getUnconnectedOutLayers()]
       # print(outputNames)

       # Object Detection Using Yolo
       detection = model.forward(outputNames)
       findObjects(detection,img) # Calling of Object Detection Function


       img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

       cv2.imshow('Image',img)
       cv2.waitKey(1)
