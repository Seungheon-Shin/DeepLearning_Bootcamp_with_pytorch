import torch
from tqdm import tqdm
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from CNN_train import CNN #CNN_train으로 부터 CNN 클래스 불러오기
import numpy as np

dataset = MNIST(root='./datasets', download=True, train=False, transform=ToTensor()) #MNIST데이터 다운로드 (ToTensor는 value를 0~1로 scaleing해줌)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False) #pytorch DataLoader 모듈 이용하여 데이터셋 불러오기


model = CNN() # 모델 정의
weight = torch.load('./weight_dict.pt') # 학습 데이터 불러오기

for k, _ in weight.items(): # 학습 데이터 확인
    print(k)

model_weight = weight['model_weight'] # 학습 데이터중 model_weight만 불러오기

for k, _ in model_weight.items(): # model_weight 확인
    print(k)

model.load_state_dict(model_weight) # model weight 갱신

list_acc = [] # accuracy 값 받을 리스트 선언
for input, label in tqdm(data_loader): # 테스트 진행
    output = model(input) # 학습 데이터로 부터 결과값 도출

    n_correct_answers = torch.sum(torch.eq(torch.argmax(output, dim=1), label)).item() # output과 label 맞는 개수 확인
    list_acc.append(n_correct_answers/ 32.0 * 100) # accuracy 리스트에 저장

print("Acc", np.mean(list_acc)) # 최종 accuracy print