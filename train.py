from google.colab import drive
drive.mount('/content/drive', force_remount = True)

import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as models

import sys, traceback

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

le = LabelEncoder()

HEIGHT, WIDTH, CHANNELS = 128, 128, 3
EMBEDDING_SIZE = 64
CLASSIFICATION = False
EMBEDDING = not CLASSIFICATION

_EPOCHS = 10
_BATCH_SIZE = 9
TRANSFER = True

haar_file = '/content/drive/My Drive/registered/haarcascade_frontalface_default.xml'
train_dir = '/content/drive/My Drive/registered'
save_file = '/content/drive/My Drive/registered/pytorch_embedder.pb'

class TripletLossNet(nn.Module):
  def __init__(self):
    super(TripletLossNet, self).__init__()
    self.resnet_model = models.resnet50(pretrained = True)
    self.resnet_model.eval()

    resnet_layers = list(self.resnet_model.children())
    last_layer = resnet_layers[len(resnet_layers) - 1]

    self.dense1 = nn.Linear(in_features = last_layer.out_features, out_features = 1024)
    self.dense1_norm = nn.BatchNorm1d(self.dense1.out_features)
    self.dense1_relu = nn.ReLU()
    
    self.dense2 = nn.Linear(in_features = self.dense1.out_features, out_features = 512)
    self.dense2_norm = nn.BatchNorm1d(self.dense2.out_features)
    self.dense2_relu = nn.ReLU()

    self.dense3 = nn.Linear(in_features = self.dense2.out_features, out_features = 256)
    self.dense3_norm = nn.BatchNorm1d(self.dense3.out_features)
    self.dense3_relu = nn.ReLU()

    self.dense4 = nn.Linear(in_features = self.dense3.out_features, out_features = 128)
    self.dense4_norm = nn.BatchNorm1d(self.dense4.out_features)
    self.dense4_relu = nn.ReLU()

    self.dense5 = nn.Linear(in_features = self.dense4.out_features, out_features = EMBEDDING_SIZE)
    self.dense5_norm = nn.BatchNorm1d(self.dense5.out_features)
    self.dense5_relu = nn.ReLU()

    self.softmax = nn.Softmax(dim=1)

  def forward(self, inputs):
    output = self.resnet_model(inputs)

    output = self.dense1(output)
    output = self.dense1_norm(output)
    output = self.dense1_relu(output)

    output = self.dense2(output)
    output = self.dense2_norm(output)
    output = self.dense2_relu(output)

    output = self.dense3(output)
    output = self.dense3_norm(output)
    output = self.dense3_relu(output)

    output = self.dense4(output)
    output = self.dense4_norm(output)
    output = self.dense4_relu(output)

    output = self.dense5(output)
    output = self.dense5_norm(output)
    output = self.dense5_relu(output)

    return output

def detect_face(img):
    haar = cv2.CascadeClassifier(haar_file)
    faces = haar.detectMultiScale(img, scaleFactor = 1.05, minNeighbors = 5)

    if(len(faces) == 0):
        raise Exception("[WARNING] face not detected ... ")
        return None, None
    else:
        (x,y,w,h) = faces[0]
        face = img[y:y+h, x:x+w]
        return (x,y,w,h), face

def triplet_data(faces, labels, n_triplets = 50):
    new_faces = []
    new_labels = []
    num_classes = len(np.unique(labels))
    
    iterations = n_triplets
    for i in range(int(iterations*0.9)):
        rand_class = np.random.randint(0, num_classes)
        # where = np.where(labels == rand_class)[0]
        # print(where)
        
        p_faces = np.where(labels == rand_class)[0]
        n_faces = np.where(labels != rand_class)[0]
        
        rand_id = p_faces[np.random.randint(0, len(p_faces))]
        anchor = faces[rand_id]
        rand_id = p_faces[np.random.randint(0, len(p_faces))]
        positive = faces[rand_id]
        rand_id = n_faces[np.random.randint(0, len(n_faces))]
        negative = faces[rand_id]
        
        # anchor - positive - negative
        new_faces.append(anchor)
        new_faces.append(positive)
        new_faces.append(negative)
        
        new_labels.append(0)
        new_labels.append(1)
        new_labels.append(-1)

    # fig,ax = plt.subplots(1,3)
    # ax[0].imshow(new_faces[0])
    # ax[1].imshow(new_faces[1])
    # ax[2].imshow(new_faces[2])
    # plt.show()
        
    # return the triplet data for training and original data for testing
    return new_faces, new_labels, faces, labels


faces = list()
labels = list()

num_classes = 0

for (dir, dirs, files) in os.walk(train_dir):
    if(dir != train_dir):
        num_classes += 1
        # vector = generate_random_vector(labels, min_distance = MAXIMUM_DISTANCE/6 - 0.2, dim = 2)
        text_label = dir.split('/')[len(dir.split('/')) - 1]

        print("[INFO] Generated vector for class : " + dir + " ... ")
        if("unknown" in dir) : continue
        for file in files:
            if(file.endswith('jpg') or file.endswith('jpeg')):
                abs_path = dir + '/' + file
                img = cv2.imread(abs_path)

                try:
                    rect, face = detect_face(img)
                    face = cv2.resize(face, (WIDTH, HEIGHT))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                    # vector = generate_random_vector(labels, min_distance = 2, dim = 2)

                    faces.append(face)
                    labels.append(text_label)
                    # text_labels.append(text_label)

                    print("[INFO] face detected in " + abs_path + " ... ")
                except :
                    traceback.print_exc(file=sys.stdout)
                    print("[WARNING] Face not detected in " + abs_path + " ... ")

labels = le.fit_transform(labels)
faces, labels, val_faces, val_labels = triplet_data(faces, labels, n_triplets = 55)

faces = np.array(faces).reshape(len(faces), HEIGHT, WIDTH, CHANNELS)
labels = np.array(labels)
val_faces = np.array(val_faces).reshape(len(val_faces), HEIGHT, WIDTH, CHANNELS)
val_labels = np.array(val_labels)

# IMPORTANT : PyTorch data preprocessing step
faces = torch.Tensor(faces)
faces = faces.reshape(-1, CHANNELS, HEIGHT, WIDTH)
val_faces = torch.Tensor(val_faces)
val_faces = val_faces.reshape(-1, CHANNELS, HEIGHT, WIDTH)

labels = torch.Tensor(labels)
labels = labels.type(torch.LongTensor)
val_labels = torch.Tensor(val_labels)
val_labels = val_labels.type(torch.LongTensor)
# labels = labels.type(torch.FloatTensor)

print("[INFO] Prepare training phase ...")

if(TRANSFER):
  net = torch.load(save_file)
  net.eval()
else:
  net = TripletLossNet()

optimizer = torch.optim.Adam(net.parameters(), lr = 0.00001, amsgrad = True)

loss = None
cosine_loss = torch.nn.CosineEmbeddingLoss(margin = 2.0)
triplet_loss = torch.nn.TripletMarginLoss(margin = 2.0, p = 2.0)

for i in range(_EPOCHS):
    running_loss = 0.0
    outputs = net(faces)

    # margin = MAXIMUM_DISTANCE / num_classes # 1.0
    # print(outputs)

    # loss = criterion(outputs, torch.max(labels, 1)[1]) # labels)
    if(CLASSIFICATION):
      loss = criterion(outputs, labels)
    else:
      for x in range(int(outputs.shape[0]/3)):
        anchor = outputs[x*3].reshape(1, EMBEDDING_SIZE)
        positive = outputs[x*3+1].reshape(1, EMBEDDING_SIZE)
        negative = outputs[x*3+2].reshape(1, EMBEDDING_SIZE)

        # print(anchor)

        if(x == 0):
          loss = triplet_loss(anchor, positive, negative) + cosine_loss(anchor, positive, torch.Tensor([1])) + cosine_loss(anchor, negative, torch.Tensor([-1]))
        else:
          loss += triplet_loss(anchor, positive, negative) + cosine_loss(anchor, positive, torch.Tensor([1])) + cosine_loss(anchor, negative, torch.Tensor([-1]))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    print("EPOCHS " + str(i + 1) + " | " + " RUNNING LOSS = " + str(running_loss))

# Testing the model
pca = PCA(n_components = 2)

embeddings = net(val_faces)
embeddings = embeddings.detach().numpy()
embeddings = pca.fit_transform(embeddings)

data = pd.DataFrame(data = embeddings, columns = ['x1', 'x2'])
data['label'] = val_labels 

for label in np.unique(data['label']):
  cluster = data[data['label'] == label]
  plt.scatter(cluster['x1'], cluster['x2'], label = label)
  
torch.save(net, save_file)

plt.legend()
plt.show()
