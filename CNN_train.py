import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class CNN(nn.Module): # CNN 클래스 정의 및 nn.Module 클래스를 상속
    def __init__(self):
        super(CNN, self).__init__() # nn.Module 클래스에 있는 __init__() 메소드를 실행.
        self.input_layer = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # 16x28x28
        self.layer_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2)  # 32x14x14
        self.layer_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2)  # 64x7x7
        self.layer_3 = nn.AdaptiveAvgPool2d((1, 1))  # 64x1x1
        self.layer_4 = nn.Linear(in_features=64, out_features=10)  # 10

    def forward(self, x):
        x1 = F.relu(self.input_layer(x))  # 16x28x28
        x2 = F.relu(self.layer_1(x1)) # 32x14x14
        x3 = F.relu(self.layer_2(x2))  # 64x7x7
        x4 = self.layer_3(x3)  # Bx64x1x1
        x5 = x4.view(x4.shape[0], 64)  # x4.shape : Bx64x1x1  >> Bx64  *squeeze 1x64x1x1 >> 64
        output = self.layer_4(x5)  # Bx10
        return output

if __name__ == '__main__': # 이 파일을 직접 실행할때만 True값 리턴
    dataset = MNIST(root='./datasets', train=True, transform=ToTensor(), download=True) # MNIST 데이터셋 다운로드
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True) #pytorch DataLoader 모듈 이용하여 데이터셋을 for 구문에서 돌림.

    model = CNN() #모델 정의


    criterion = nn.CrossEntropyLoss() #Loss 설정 (크로스엔트로피)

    optim = torch.optim.Adam(model.parameters(), lr=0.001)  # weight_new = weight_old - weight_gradient * lr

    if os.path.isfile("./weight_dict.pt"): # weight_dict.pt 파일이 있으면 True 리턴
        model_dict = torch.load('./weight_dict.pt')['mdoel_weight'] # 학습 weight 불러오기 ( model )
        model.load_state_dict(model_dict) # 모델 weight 갱신
        adam_dict = torch.load('./weight_dict.pt')['adam_weight'] # 학습 weight 불러오기 ( optimizer )
        optim.load_state_dict(adam_dict) # adam optimizer weight 갱신

    list_loss = [] # Loss값을 받을 리스트 선언
    list_acc = [] # accuracy값을 받을 리스트 선언
    for epoch in range(1): # 학습 epoch값 설정
        for input, label in tqdm(data_loader): # 데이터 불러오기 (batchsize = 32)
            # label 32
            output = model(input)  # 32x10
            loss = criterion(output, label)  # 1

            optim.zero_grad() #optimizer를 이용하여 weight들의 gradient 초기화
            loss.backward() # loss값에 따른 gradient 전달
            optim.step() #optimizer를 통한 weight 갱신
            list_loss.append(loss.detach().item()) #loss값 저장

            n_correct_answers = torch.sum(torch.eq(torch.argmax(output, dim=1), label)).item() # output과 label 맞는 개수 확인
            print("Accuracy: ", n_correct_answers / 32.0 * 100) # accuracy값 print
            list_acc.append(n_correct_answers / 32.0 * 100) # accuracy값 저장



    weight_dict = {'model_weight': model.state_dict(), 'adam_weight': optim.state_dict()} # 저장시킬 가중치 정의
    torch.save(weight_dict, "./weight_dict.pt") #훈련된 weight 저장


    plt.plot(list_loss) #plot할 데이터 (loss)
    plt.plot(list_acc) # plot할 데이터 (accuracy)

    plt.xlabel("Iteration") # x축 정의
    plt.ylabel("Loss") # y축 정의
    plt.show() # 결과 plot하기
