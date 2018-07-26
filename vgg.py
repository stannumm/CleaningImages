from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
import os

model = VGG16()

pathpre = "path"
names = os.listdir(pathpre)

labels=["labels"]
for n in names:
        try :
            image = load_img(pathpre+n, target_size=(224, 224))
            image = img_to_array(image)
            # reshape data for the model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            # prepare the image for the VGG model
            image = preprocess_input(image)
            # predict the probability across all output classes
            yhat = model.predict(image)
            # convert the probabilities to class labels
            label = decode_predictions(yhat)
            # retrieve the most likely result, e.g. highest probability
            label = label[0][0]
            # print the classification
            print('%s (%.2f%%)' % (label[1], label[2]*100) , n)
            if label[1] not in labels:
                os.remove(pathpre+n)
        except Exception as e:
            print(e)
            os.remove(pathpre+n)

