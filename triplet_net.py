import torch
import torch.nn as nn
import torchvision.models as models

EMBEDDING_SIZE = 128

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
