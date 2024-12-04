# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
current_directory = os.getcwd()
print(current_directory)

for dirname, _, filenames in os.walk(current_directory):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# # **Import all the required dependancies**

# %%
import numpy as np
print(np.__version__)
import pandas as pd
print(pd.__version__)
import matplotlib
print(matplotlib.__version__)
import cv2
print(cv2.__version__)
import sklearn
print(sklearn.__version__)




# %%
# opencv yüklü değilse
#!pip install opencv-python

import os
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# %% [markdown]
# # **Visualization of the dataset with the labels**

# %%
train_images_path = current_directory + '/train'
annotation_file_path = current_directory + '/train/_annotations.txt'

class_names = {
    0: 'DangerousDriving',
    1: 'SafeDriving',
    2: 'SleepyDriving',
}

#dict
annotations = {}


with open(annotation_file_path, 'r') as f:
    for line in f:
        # strip yazının başındaki ve sonundaki boşlukları atar, split ile string boşluk karakterine göre bölünür.
        parts = line.strip().split()

        
        if len(parts) < 2:
            print(f'skipping due to the incorrect format: {line}')
            continue
            
        image_name = parts[0]
        
        bbox_and_class = parts[1].split(',')
        
        if len(bbox_and_class) <5:
            print(f"Skipping the line due to missing bouding box or class info: {line}")
            continue
        

        class_id = int(bbox_and_class[-1])

        if class_id == 1:
            class_id = 0
        if class_id == 2 or class_id == 3:
            class_id = 1
        if class_id == 4 or class_id == 5:
            class_id = 2
        
        annotations[image_name] = class_id

        

# %%
def load_image_and_label(image_name):

    # Belirtilen görüntü ismi ile görüntünün tam yolu oluşturuluyor.
    img_path = os.path.join(train_images_path, image_name)
    # OpenCV (cv2) kullanılarak görüntü dosyası yükleniyor.
    img = cv2.imread(img_path)
    #  OpenCV'de yüklenen görüntüler varsayılan olarak BGR formatında olur, bu yüzden görüntüyü RGB formatına dönüştürüyor.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if image_name in annotations:
        class_id = annotations[image_name]
        return img, class_names[class_id]
    else:
        print(f'Warning: No label found for {image_name}')
        return img, "No label"

# fotoğrafları listeleyip ilk 9'u alınıyor.
plt.figure(figsize = (10,10))
image_files = os.listdir(train_images_path)[:9]
print(image_files)

#enumerate hem listedeki elemanın index'ini hem de o index'deki elemanı verir
for i, image_file in enumerate(image_files):
    img, label = load_image_and_label(image_file)

    ax = plt.subplot(3,3,i+1)
    plt.imshow(img)
    plt.title(f'label: {label}')
    plt.axis('off')

plt.show()

# %% [markdown]
# # **Data preprocessing**

# %%
# belirtilen bir dosyadan anotasyon verilerini okuyarak, her bir satırdaki bilgileri işleyip bir pandas 
# DataFrame'e dönüştürür. Kodun amacı, görüntü adları ve bu görüntülere ait sınıf ve bounding box (sınırlayıcı kutu) koordinatlarını organize etmek ve
# daha kolay işlenebilir bir formata (DataFrame) dönüştürmektir.
def read_annotations(file_path):
    annotations = []
    with open(file_path,'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            image_name = parts[0]
            try:
                # ikinci kısım virgüle göre parçalara ayrılır, map(int) ile her biri int'e dönüştürülür.
                bbox = list(map(int,parts[1].split(',')))
                if len(bbox) !=5:
                    continue
                if bbox[4] == 1:
                    bbox[4] = 0
                if bbox[4] == 2 or bbox[4] == 3:
                    bbox[4] = 1
                if bbox[4] == 4 or bbox[4] == 5:
                    bbox[4] = 2
                    
                annotations.append([image_name] + bbox)
            except ValueError:
                continue
    return pd.DataFrame(annotations,columns = ['image_name','x_min','y_min','x_max','y_max','class_id'])

# %%
# verilen bir görüntüyü okur, renk formatını dönüştürür ve belirli bir boyuta yeniden boyutlandırarak ön işleme tabi tutar. 
# Bu tür ön işleme, genellikle derin öğrenme modellerine giriş olarak kullanılacak görüntüler için yapılır.
target_size = (224,224)
def preprocess_image(image_path,target_size):
    # OpenCV (cv2) kullanılarak görüntü dosyası yükleniyor.
    image = cv2.imread(image_path)
    #  OpenCV'de yüklenen görüntüler varsayılan olarak BGR formatında olur, bu yüzden görüntüyü RGB formatına dönüştürüyor.
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # görüntüyü target_size boyutlarına yeniden boyutlandırır.
    image = cv2.resize(image,target_size)
    return image    

# %%
from sklearn.preprocessing import OneHotEncoder

def one_hot_encode_labels(labels, num_classes):
    encoder = OneHotEncoder(categories=[range(num_classes)], sparse_output=False, handle_unknown='ignore')
    labels = np.array(labels).reshape(-1, 1)
    one_hot_labels = encoder.fit_transform(labels)
    return one_hot_labels


# %%
train_folder = current_directory + '/train'
test_folder = current_directory + '/test'
valid_folder = current_directory + '/valid'

# Dataframes
train_annotations = read_annotations(train_folder + '/_annotations.txt')
test_annotations = read_annotations(test_folder + '/_annotations.txt')
valid_annotations = read_annotations(valid_folder + '/_annotations.txt')

# %%
print(train_annotations)

# %%
def preprocess_dataset(annotations,folder_path,num_classes):
    images = []
    labels = []
#        print(annotations.head())
#       print(annotations.columns)

    # iterrows DataFrame'deki her bir satırı bir tuple (demet) olarak döndürür. İlk eleman: Satırın index'i (satır numarası). İkinci eleman: O satırı temsil eden bir pandas Series nesnesi.
    # Yani her döngüde pandasın bir satırını okur ve ilk eleman olarak satır numarasını, ikinci eleman olarak o satırdakileri pandas serisi olarak verir.
    for _, row in annotations.iterrows():
        image_path = os.path.join(folder_path,row['image_name'])
        image = preprocess_image(image_path,target_size)
        images.append(image)
        labels.append(row['class_id'])
    # Listeden numpy array'ine dönüştürme
    print(type(labels))
    images = np.array(images)
    one_hot_labels = one_hot_encode_labels(labels, num_classes)
    return images,one_hot_labels

num_classes = 3

train_images, train_labels = preprocess_dataset(train_annotations,train_folder,num_classes)
test_images, test_labels = preprocess_dataset(test_annotations,test_folder,num_classes)
valid_images, valid_labels = preprocess_dataset(valid_annotations,valid_folder,num_classes)

# %%
train_labels

# %%
processed_images = []
folder_name = "./processed_images"
for file_name in os.listdir(folder_name):
    file_path = os.path.join(folder_name, file_name)
    processed_images.append(cv2.imread(file_path))
    

# %%
len(processed_images)

# %%
import random
# Usage
random_images = random.sample(processed_images, 4400)

len(random_images)



# %%
# Usage
random_images_for_test = random.sample(processed_images, 350)
random_images_for_valid = random.sample(processed_images, 650)



# %%
train_images2 = []
labels2 = []

for img in random_images:
    train_images2.append(img)
    labels2.append(2)

train_images3 = np.array(train_images2)

train_images = np.concatenate((train_images, train_images3), axis=0)

labels3 = one_hot_encode_labels(labels2, num_classes)
train_labels = np.concatenate((train_labels, labels3), axis=0)


# %%
test_images2 = []
labels2 = []

for img in random_images_for_test:
    test_images2.append(img)
    labels2.append(2)

test_images3 = np.array(test_images2)

test_images= np.concatenate((test_images, test_images3), axis=0)

labels3 = one_hot_encode_labels(labels2, num_classes)
test_labels = np.concatenate((test_labels, labels3), axis=0)

# %%
valid_images2 = []
labels2 = []

for img in random_images_for_valid:
    valid_images2.append(img)
    labels2.append(2)

valid_images3 = np.array(valid_images2)

valid_images= np.concatenate((valid_images, valid_images3), axis=0)

labels3 = one_hot_encode_labels(labels2, num_classes)
valid_labels = np.concatenate((valid_labels, labels3), axis=0)

# %% [markdown]
# # **Shape of the dataset after preprocessing steps**

# %%
print(f'Train images shape: {train_images.shape}')
print(f'Train labels shape: {train_labels.shape}')
print(f'Test images shape: {test_images.shape}')
print(f'Test labels shape: {test_labels.shape}')
print(f'Valid images shape: {valid_images.shape}')
print(f'Valid labels shape: {valid_labels.shape}')

# %%
print(type(train_images))
print(type(train_labels))

# %%
print(train_images)

# %% [markdown]
# # **Assigning the data to Train and Validation**

# %%
(X_train, y_train), (X_val, y_val) = (train_images, train_labels), (valid_images, valid_labels)
(X_test, y_test) = (test_images, test_labels)

# %% [markdown]
# # **Neural netowork (CNN model)**

# %% [markdown]
# # **Classical CNN model**

# %%
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")


# %%
#pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# %%
import tensorflow as tf

print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")


# %%
import torch
print(torch.version.cuda)

# %%
import tensorflow as tf
print(tf.__version__)

# %%
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
model = Sequential([
    Conv2D(32,(3,3),activation = 'relu',input_shape=(224,224,3)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3), activation = 'relu'),
    MaxPooling2D((2,2)),
    Conv2D(128,(3,3), activation = 'relu'),
    MaxPooling2D((2,2)),
    Conv2D(256,(3,3), activation = 'relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(512, activation = 'relu'),
    Dropout(0.5),
    Dense(3,activation = 'softmax')
]) 

model.summary()

# %%
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
print(tf.keras.__version__)

# %%
# Train the model
history = model.fit(X_train, y_train, epochs=12, batch_size=16, validation_data=(X_val, y_val))

# %%
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()


# %% [markdown]
# # **Plot the first 16 images with actual label and predicted label**

# %%
import matplotlib.pyplot as plt
import numpy as np

# Function to plot 16 images in a grid with predicted and actual labels
def plot_images_grid_with_predictions(images, actual_labels, predicted_labels, class_names, grid_size=(4, 4)):
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 12))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i])
            actual_label = class_names[np.argmax(actual_labels[i])]
            predicted_label = class_names[np.argmax(predicted_labels[i])]
            ax.set_title(f'Actual: {actual_label}\nPredicted: {predicted_label}')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

class_names = ['DangerousDriving', 'SafeDriving', 'SleepyDriving']

# Select 16 random test images
indices = np.random.choice(len(test_images), 16, replace=False)
sample_images = test_images[indices]
sample_actual_labels = test_labels[indices]



# Predict the labels
sample_predicted_labels = model.predict(sample_images)
plot_images_grid_with_predictions(sample_images, sample_actual_labels, sample_predicted_labels, class_names)



# %% [markdown]
# # **Accuracy of the test dataset**

# %%
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_accuracy:.2f}')

# %% [markdown]
# # **Save the model in h5 version**

# %%
model.save('driving_behavior_model')  # This saves the model in TensorFlow's SavedModel format


# %%
import tensorflow as tf

# Tüm GPU cihazlarını listeleyin
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    details = tf.config.experimental.get_device_details(gpu)
    print("GPU Özellikleri:")
    print("Ad:", details.get('device_name'))
#    print("Bellek (MB):", details.get('memory_limit') / 1024 / 1024)
    print("Compute Capability:", details.get('compute_capability'))
    print()


# %%
import tensorflow as tf

# Tüm GPU cihazlarını listeleyin
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("GPU Özellikleri:")
    print("Ad:", gpu.name)
    
    # Bellek büyüme özelliğini kontrol et
    if tf.config.experimental.get_memory_growth(gpu):
        print("Bellek büyüme etkin.")
    else:
        print("Bellek büyüme etkin değil.")
    
    # Bellek bilgisi almak için nvidia-smi kullan
    import subprocess
    gpu_info = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used', '--format=csv'],
                              stdout=subprocess.PIPE, text=True)
    print(gpu_info.stdout)

    print()


# %%
import tensorflow as tf
print("Kullanılabilir GPU Sayısı: ", len(tf.config.list_physical_devices('GPU')))


# %%
import matplotlib.pyplot as plt

def capture_single_image():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Kamera açılamadı!")
        return None

    ret, frame = cap.read()
    if not ret:
        print("Kare alınamadı!")
        return None

    # Kameradan alınan görüntüyü göster
#    cv2.imshow('Kamera', frame)
    """
    # 'Space' tuşuna basıldığında fotoğraf çek
    if cv2.waitKey(1) & 0xFF == ord(' '):
        print("Fotoğraf çekildi!")
        captured_image = frame
        break

    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Fotoğraf çekiminden çıkıldı.")
        captured_image = None
        break
    """
    cap.release()
#    cv2.destroyAllWindows()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.title("Çekilen Görüntü")
    plt.axis("off")  # Eksenleri kapat
    plt.show()

    file_name = "deneme.jpg"
    current_dir = os.getcwd()
    save_path = os.path.join(current_dir, file_name)
    cv2.imwrite(save_path, frame)
    return save_path

# 3. Tahmin yap
def predict_image(path):
    """
    if image is None:
        print("Fotoğraf alınamadı, tahmin yapılamıyor.")
        return
    """    
    """
    # Görüntüyü modele uygun boyutlara getirin
    input_image = cv2.resize(image, (model.input.shape[1], model.input.shape[2]))
    input_image = input_image / 255.0  # Normalizasyon
    input_image = np.expand_dims(input_image, axis=0)  # (1, height, width, channels)
    """
    print(model.input_shape)

    preprocessed_image = preprocess_image(path, target_size)
    print(preprocessed_image.shape)
    plt.imshow(preprocessed_image)
    plt.title("Çekilen Görüntü")
    plt.axis("off")  # Eksenleri kapat
    plt.show()
    gray_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray_image, cmap="gray")
    plt.title("Çekilen Görüntü")
    plt.axis("off")  # Eksenleri kapat
    plt.show()
    gray_image_3channel = cv2.merge([gray_image, gray_image, gray_image])
    print(gray_image_3channel.shape)
    preprocessed_image = np.expand_dims(gray_image_3channel, axis=0)
    plt.imshow(gray_image_3channel, cmap="gray")
    plt.title("Çekilen Görüntü")
    plt.axis("off")  # Eksenleri kapat
    plt.show()
   
    
    print(preprocessed_image.shape)
    # Model tahmini
    predictions = model.predict(preprocessed_image)
    predicted_label = np.argmax(predictions, axis=1)[0]  # En yüksek olasılığa sahip sınıf
    if predicted_label == 0:
        print("Tahmin: DangerousDriving")
    elif predicted_label == 1:
        print("Tahmin: SafeDriving")
    else:
        print("Tahmin: SleepyDriving")

# 4. İş akışı
#path = capture_single_image()  # Fotoğraf çek
path = capture_single_image()  # Fotoğraf çek
predict_image(path)            # Tahmin yap


