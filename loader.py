from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  

test_dir='./dataset/test/'

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150,150),
        batch_size=1,
        color_mode="rgb",
        shuffle = False,
        class_mode='categorical')


def loadModelAndPredict(modelName):

    print("Loading model " + modelName)
    model = load_model("./" + modelName + ".h5")
    model.summary()
      
    labels=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

    filenames = test_generator.filenames

    nb_samples = len(filenames)

    predictions = model.predict_generator(test_generator, steps=nb_samples)
    classes = test_generator.classes[test_generator.index_array]
    y_pred = np.argmax(predictions, axis=-1)
    print("Accuracy of "+ str(sum(y_pred==classes)) + "%")
    print(classification_report(test_generator.classes, y_pred, target_names=labels))
    cm = confusion_matrix(test_generator.classes, y_pred)


    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax)
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.savefig("./plot/" + modelName + ".png")


loadModelAndPredict("model.finetunning")