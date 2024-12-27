import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
# CNN 모델 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32,64,3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64* 8* 8, 512)

        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512,10)

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))  # 첫 번째 convolution + ReLU + MaxPool
        x = self.pool(self.relu2(self.conv2(x)))  # 두 번째 convolution + ReLU + MaxPool
        x = self.flatten(x)  # Flatten
        x = self.fc1(x)  # Fully connected layer 1
        x = self.relu3(x)  # ReLU
        x = self.fc2(x)  # Fully connected layer 2
        return x
  
    # 데이터셋 정의
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

def load_cifar10_data(data_dir):
    train_images = []
    train_labels = []

    for i in range(1, 6):
        batch_file = f"{data_dir}/data_batch_{i}"
        batch_data = unpickle(batch_file)
        train_images.append(batch_data[b'data'])
        train_labels.extend(batch_data[b'labels'])

    train_images = np.vstack(train_images).reshape(-1, 3, 32, 32)  
    train_labels = np.array(train_labels)


    test_batch = unpickle(f"{data_dir}/test_batch")
    test_images = np.vstack(test_batch[b'data']).reshape(-1, 3, 32, 32)  
    test_labels = np.array(test_batch[b'labels'])

    return train_images, train_labels, test_images, test_labels
#---------------
# 가져와서 호출하기
data_dir = "./cifar-10-batches-py"
train_images, train_labels, test_images, test_labels = load_cifar10_data(data_dir)

train_images_tensor = torch.tensor(train_images, dtype=torch.float32).permute(0, 1, 2, 3) / 255.0  # [N, C, H, W]
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
test_images_tensor = torch.tensor(test_images, dtype=torch.float32).permute(0, 1, 2, 3) / 255.0
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
print(f"train_images_tensor shape: {train_images_tensor.shape}")
print(f"train_labels_tensor shape: {train_labels_tensor.shape}")
print(f"test_images_tensor shape: {test_images_tensor.shape}")
print(f"test_labels_tensor shape: {test_labels_tensor.shape}")

train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # 사용 중인 장치 확인

model = CNN().to(device)

criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=0.001)  

# 훈련 코드
model.train()
for epoch in range(10):
    for index, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if index % 100 == 0:
            print("loss of {} epoch, {} index: {}".format(epoch, index, loss.item()))


model.eval()

correct = 0
total = 0

start_time = time.time()
# 테스트 코드
with torch.no_grad():

    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)  
        
        _, output_index = torch.max(output, 1)
        
        total += target.size(0)
        correct += (output_index == target).sum()
    print(correct, total)
    
    accuracy = 100 * correct / total
    print(f"Accuracy of Test Data: {accuracy.item():.2f}%")

avg_inference_time = (time.time() - start_time)/ 100
print(f"Average Inference Time: {avg_inference_time:.6f} seconds")
