from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import train_test_split

test_dir='./dataset/test/'

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150,150, 3),
        batch_size=1,
        shuffle = False,
        class_mode='binary')


def loadModelAndPredict(fileName):
    print("Loading model " + fileName)
    model = load_model(fileName)
    model.summary()

    filenames = test_generator.filenames
    nb_samples = len(filenames)
    y_pred = model.predict_generator(test_generator, steps=nb_samples)


    print(y_pred)
    print(test_generator.classes)

    matrix = confusion_matrix(test_generator.classes, y_pred > 0.5)

    print("Confusion matrix")
    print(matrix)
    
    auc=roc_auc_score(test_generator.classes, y_pred > 0.5)
    precision=precision_score(test_generator.classes, y_pred  > 0.5)
    recall =recall_score(test_generator.classes, y_pred  > 0.5)
    specificity = matrix[1,1]/(matrix[1,0]+matrix[1,1])
    fscore= f1_score(test_generator.classes, y_pred  > 0.5)
    acc=accuracy_score(test_generator.classes, y_pred  > 0.5)

    print("acc: ", acc)
    print("Precision: ", precision)
    print("recall: ", recall)
    print("specificity: ", specificity)
    print("fscore: ", fscore)
    print("AUC: ", auc)

loadModelAndPredict("./flower_model.h5")