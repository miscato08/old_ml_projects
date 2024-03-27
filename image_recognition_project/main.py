# imageai.Prediction no longer exists, replaced by imageai.Classification
from imageai.Classification import ImageClassification
import os

# create a path
exec_path = os.getcwd()

prediction = ImageClassification()
# SqueezeNet model also no longer exists, now the fastest is MobileNetV2
prediction.setModelTypeAsMobileNetV2()
# Set Model Path
prediction.setModelPath(os.path.join(exec_path, "mobilenet_v2-b0353104.pth"))
# Load the model
prediction.loadModel()

# make predictions
predctions, probabilities = prediction.classifyImage(
    # give the path and the results count
    os.path.join(exec_path, "house.jpg"), result_count=5
)
# iterate through the predictions and probabilities and zip it + output
for eachPred, eachProb in zip(predctions, probabilities): # zip predictions and probabilities
    print(f"{eachPred} : {eachProb}") # write it down
