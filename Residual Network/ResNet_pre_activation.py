import torch
import torch.nn as nn

class ResidualBlock(nn.Module): # Residual Block 클래스 생성, nn.Module 클래스 상속
    def __init__(self, n_ch, pre_activation=False):
        super(ResidualBlock, self).__init__()   #nn.Module의 __init__ method를 반드시 실행 하라.
        self.conv1 = nn.Conv2d(n_ch, n_ch, kernel_size=3, padding=1, bias =False) # n_ch x input_dim x input_dim ( MNIST image : 28x28)
        self.bn1 = nn.BatchNorm2d(n_ch)  #BatchNormalization 하기전에 선행하는 layer의 bias를 False로 해주어야함.
        self.act = nn.ReLU(inplace=True)  #inplace는 act 함수를 통과하기 전의 메모리와 통과한 후의 메모리 공간을 공유하여 메모리를 차지하는 공간이 늘어나지 않게 해줌.
        self.conv2 = nn.Conv2d(n_ch, n_ch, kernel_size=3, padding=1, bias=False) #n_ch x input_dim x input_dim
        self.bn2 = nn.BatchNorm2d(n_ch)

        self.pre_activation = True if pre_activation else False #한줄에 if문을 쓸때는 else도 꼭 필요함


    def forward(self, x):
        if self.pre_activation:
            y = self.bn1(x)
            y = self.act(y)
            y = self.conv1(y)
            y = self.bn2(y)
            y = self.act(y)
            y = self.conv2(y)
            return x+y
        else:
            y = self.conv1(x) # convolution 연산 적용
            y = self.bn1(y) # use BN
            y = self.act(y) # activation 함수 적용
            y = self.conv2(y) # convolution 연산 적용
            y = self.bn2(y) # use BN

            return self.act(x + y) # input(x)값과 연산 결과 (y)의 합을 activation 함수 적용하여 리턴 ( ResNet의 기본개념 )
'''마지막 bn 적용 후 activation을 걸어주지 않는다. 더해준 다음에 activation을 걸어준다.
relu가 forward안으로 들어가면 결과가 더 좋아졌는데 그 이유는 처음 들어온 신호에 대한 activation을 빼주어서
처음 신호의 정보가 끝까지 변형되지 않고 이어지기 때문'''

class ResidualNetwork(nn.Module): # ResidualNetwork 클래스 정의, nn,Module 상속
    def __init__(self, pre_activation=False):
        super(ResidualNetwork, self).__init__() # nn.Module의 __init__ method 실행
        network = [nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)]# forward 함수를 간략히 하고자 nn.Sequential을 이용할 수 있도록 network list 선언
        if not pre_activation:
            network += [nn.BatchNorm2d(16, pre_activation = pre_activation),
                        nn.ReLU(True)]

        network += [ResidualBlock(16, pre_activation = pre_activation), #residual block 적용
                    ResidualBlock(16, pre_activation = pre_activation), #residual block 적용
                    nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=False), # 32x14x14
                    nn.BatchNorm2d(32), #Batch_Normalization 실행
                    ResidualBlock(32, pre_activation = pre_activation), #residual block 적용
                    ResidualBlock(32, pre_activation = pre_activation), #residual block 적용
                    nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False), # 64x7x7
                    nn.BatchNorm2d(64), #Batch_Normalization 실행
                    ResidualBlock(64, pre_activation = pre_activation), #residual block 적용
                    ResidualBlock(64, pre_activation = pre_activation)] #residual block 적용

        if pre_activation:
            network += [nn.BatchNorm2d(64),
                        nn.ReLU(True)]
        network += [nn.AdaptiveAvgPool2d((1,1)), # 64x1x1
                    View(64), #64
                    nn.Linear(64,10)] # 64x10
        self.network = nn.Sequential(*network)  # *의 역할은 리스트로 정의된 network의 괄호를 벗겨줌
    '''보통 bias 의 초기 weight를 설정해줌.'''

    def forward(self, x): #forward 함수 정의
        return self.network(x) # __init__의 network(nn.Sequential로 실행됨) 리턴



class View(nn.Module): # View 클래스 정의 nn.Module 상속
    def __init__(self, *shape):  # *는 argument 개수가 몇개 들어올지 알수 없을때 들어온 argumennt들을 튜플의 형태로 저장, **는 키워드 argument 딕셔너리 형태로 저장
        super(View, self).__init__() # nn.Module의 __init__ method 실행
        self.shape = shape #self.shape 정의 (__init__ method의 인자)

    def forward(self, x): # forward 함수 정의
        return x.view(x.shape[0], *self.shape) #64x64x1x1 -->> 64x64 # *를 한번 더 하면 튜플의 괄호를 벗겨서 value만 넘겨줌

'''이미지를 바꾸는 모델의 경우 그 전의 모델이 남아 있으면 성능이 안좋아 지는 경우들이 있음'''