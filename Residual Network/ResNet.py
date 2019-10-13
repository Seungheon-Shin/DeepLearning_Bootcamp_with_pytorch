import torch
import torch.nn as nn

class ResidualBlock(nn.Module): # Residual Block 클래스 생성, nn.Module 클래스 상속
    def __init__(self, n_ch):
        super(ResidualBlock, self).__init__()   #nn.Module의 __init__ method를 반드시 실행 하라.
        self.conv1 = nn.Conv2d(n_ch, n_ch, kernel_size=3, padding=1, bias =False) # n_ch x input_dim x input_dim ( MNIST image : 28x28)
        self.bn1 = nn.BatchNorm2d(n_ch)  #BatchNormalization 하기전에 선행하는 layer의 bias를 False로 해주어야함.
        self.act = nn.ReLU(inplace=True)  #inplace는 act 함수를 통과하기 전의 메모리와 통과한 후의 메모리 공간을 공유하여 메모리를 차지하는 공간이 늘어나지 않게 해줌.
        self.conv2 = nn.Conv2d(n_ch, n_ch, kernel_size=3, padding=1, bias=False) #n_ch x input_dim x input_dim
        self.bn2 = nn.BatchNorm2d(n_ch)


    def forward(self, x):
        y = self.conv1(x) # convolution 연산 적용
        y = self.bn1(y) # use BN
        y = self.act(y) # activation 함수 적용
        y = self.conv2(y) # convolution 연산 적용
        y = self.bn2(y) # use BN

        return self.act(x + y) # input(x)값과 연산 결과 (y)의 합을 activation 함수 적용하여 리턴 ( ResNet의 기본개념 )


class ResidualNetwork(nn.Module): # ResidualNetwork 클래스 정의, nn,Module 상속
    def __init__(self):
        super(ResidualNetwork, self).__init__() # nn.Module의 __init__ method 실행
        '''self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)  # color 3체널을 16체널로 바꾸어줌
        self.bn1 = nn.BatchNorm2d(16)
        self.act = nn.ReLU(True)

        self.rb1 = ResidualBlock(16)
        self.rb2 = ResidualBlock(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.rb3 = ResidualBlock(32)
        self.rb4 = ResidualBlock(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.rb5 = ResidualBlock(64)
        self.rb6 = ResidualBlock(64)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(64,100)'''

        network = [] # forward 함수를 간략히 하고자 nn.Sequential을 이용할 수 있도록 network list 선언
        network += [nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False), # 16x28x28
                    nn.BatchNorm2d(16), #Batch_Normalization 실행
                    nn.ReLU(True), # activation 함수 적용
                    ResidualBlock(16), #residual block 적용
                    ResidualBlock(16), #residual block 적용
                    nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=False), # 32x14x14
                    nn.BatchNorm2d(32), #Batch_Normalization 실행
                    ResidualBlock(32), #residual block 적용
                    ResidualBlock(32), #residual block 적용
                    nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False), # 64x7x7
                    nn.BatchNorm2d(64), #Batch_Normalization 실행
                    ResidualBlock(64), #residual block 적용
                    ResidualBlock(64), #residual block 적용
                    nn.AdaptiveAvgPool2d((1,1)), # 64x1x1
                    View(64), #64
                    nn.Linear(64,10)] # 64x10
        self.network = nn.Sequential(*network)  # *의 역할은 리스트로 정의된 network의 괄호를 벗겨줌

    def forward(self, x): #forward 함수 정의
        return self.network(x) # __init__의 network(nn.Sequential로 실행됨) 리턴
'''        x = self.conv1(x)
        x = self. bn1(x)
        x = self.act(x)

        x = self.rb1(x)
        x = self.rb2(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.rb3(x)
        x = self.rb4(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)

        x = self.rb5(x)
        x = self.rb6(x)

        x = self.avg_pool(x)
        x = self.linear(x)

        return x'''


class View(nn.Module): # View 클래스 정의 nn.Module 상속
    def __init__(self, *shape):  # *는 argument 개수가 몇개 들어올지 알수 없을때 들어온 argumennt들을 튜플의 형태로 저장, **는 키워드 argument 딕셔너리 형태로 저장
        super(View, self).__init__() # nn.Module의 __init__ method 실행
        self.shape = shape #self.shape 정의 (__init__ method의 인자)

    def forward(self, x): # forward 함수 정의
        return x.view(x.shape[0], *self.shape) #64x64x1x1 -->> 64x64 # *를 한번 더 하면 튜플의 괄호를 벗겨서 value만 넘겨줌