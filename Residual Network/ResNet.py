import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, n_ch):
        super(ResidualBlock, self).__init__()   #nn.Module의 __init__ method를 반드시 실행 하라.
        self.conv1 = nn.Conv2d(n_ch, n_ch, kernel_size=3, padding=1, bias =False)
        self.bn1 = nn.BatchNorm2d(n_ch)  #BatchNormalization 하기전에 선행하는 layer의 bias를 False로 해주어야함.
        self.act = nn.ReLU(inplace=True)  #inplace는 act 함수를 통과하기 전의 메모리와 통과한 후의 메모리 공간을 공유하여 메모리를 차지하는 공간이 늘어나지 않게 해줌.
        self.conv2 = nn.Conv2d(n_ch, n_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_ch)


    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.bn2(y)

        return self.act(x + y)


class ResidualNetwork(nn.Module):
    def __init__(self):
        super(ResidualNetwork, self).__init__()
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

        network = []
        network += [nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(True),
                    ResidualBlock(16),
                    ResidualBlock(16),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(32),
                    ResidualBlock(32),
                    ResidualBlock(32),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(64),
                    ResidualBlock(64),
                    ResidualBlock(64),
                    nn.AdaptiveAvgPool2d((1,1)), # 64x64x1x1
                    View(64), #64x64
                    nn.Linear(64,10)] # 64x10
        self.network = nn.Sequential(*network)  # *의 역할은 리스트로 정의된 network의 괄호를 벗겨줌

    def forward(self, x):
        return self.network(x)
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


class View(nn.Module):
    def __init__(self, *shape):  # *는 argument 개수가 몇개 들어올지 알수 없을때 들어온 argumennt들을 튜플의 형태로 저장, **는 키워드 argument 딕셔너리 형태로 저장
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape) #64x64x1x1 -->> 64x64 # *를 한번 더 하면 튜플의 괄호를 벗겨서 value만 넘겨줌
