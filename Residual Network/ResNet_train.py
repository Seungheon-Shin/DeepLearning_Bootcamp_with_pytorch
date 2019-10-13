import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR100, MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from ResNet import ResidualNetwork # ResNet.py로 부터 ResidualNetwork 클래스 불러오기
import os

#transforms = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
dataset = MNIST(root='./datasets', train=True, transform=ToTensor(), download=True) # MNIST 데이터셋 준비, (트레이닝셋, 텐서로 데이터 형식 변환, 다운로드)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True) # 파이토치 DataLoader 모듈을 이용하여 데이터셋 넘겨주기, (batch size = 64)
model = ResidualNetwork() # model 정의


criterion = nn.CrossEntropyLoss() # 사용할 Loss 설정, Loss = 크로스엔트로피

optim = torch.optim.Adam(model.parameters(), lr=0.001) # optimizer로 Adam 사용, optimizer 에는 model의 weight값을 확인 할 수 있도록 model.parameters()가 필수로 필요.

'''if os.path.isfile("./weight_dict.pt"):
    model_dict = torch.load('./weight_dict.pt')['mdoel_weight']
    model.load_state_dict(model_dict)
    adam_dict = torch.load('./weight_dict.pt')['adam_weight']
    optim.load_state_dict(adam_dict)'''

list_loss = [] #loss값을 저장할 리스트 선언
list_acc = [] # accuracy값을 저장할 리스트 선언
for epoch in range(1): # epoch 설정 (1)
    for input, label in tqdm(data_loader): # DataLoader 모듈로 부터 input과 label 받기

        output = model(input) # Resnet 모델에 input 적용
        loss = criterion(output, label) # loss값 계산

        optim.zero_grad() # 모델 weight들의 gradient 초기화
        loss.backward() # loss에 따른 gradient 전달
        optim.step() # weight 갱신
        list_loss.append(loss.detach().item()) # loss값 저장

        n_correct_answers = torch.sum(torch.eq(torch.argmax(output, dim=1), label)).item() # 옳게 분류한 갯수 세기
        print("Accuracy: ", n_correct_answers / 64.0 * 100) # accuracy 계산하여 print
        list_acc.append(n_correct_answers / 64.0 * 100) # accuracy 계산하여 저장



weight_dict = {'model_weight': model.state_dict(), 'adam_weight': optim.state_dict()} # 훈련된 모델에 대해 저장할 가중치 정의 (model weight와 adam weight)
torch.save(weight_dict, "./weight_dict.pt") # weight 저장


plt.plot(list_loss) # loss값을 그래프로 plot
plt.plot(list_acc) # accuracy값을 그래프로 plot

plt.xlabel("Iteration") # x축 정의
plt.ylabel("Loss") # y축 정의
plt.show() # 그래프 출력
