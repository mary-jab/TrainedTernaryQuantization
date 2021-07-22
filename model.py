import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ConvNet_1D(nn.Module):
    def __init__(self, name, kernel_size=3, num_classes=6, num_timesteps=128, num_features=9):
        super(ConvNet_1D, self).__init__()
        self.name = name
        nch = 256
        pad = kernel_size//2
        self.layer1 = nn.Sequential(
            nn.Conv1d(num_timesteps, nch, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(nch),
            nn.ReLU(),
            nn.Dropout(.15),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(nch, nch // 2,kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(nch // 2),
            nn.ReLU(),
            nn.Dropout(.15),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )
        # self.layer2_2 = nn.Sequential(
        #     nn.Conv1d(nch//2, nch//2, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(nch//2),
        #     nn.ReLU(),
        #     # nn.Dropout(.5),
        #     )
        self.layer3 = nn.Sequential(
            nn.Conv1d(nch // 2, nch // 4, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(nch // 4),
            nn.ReLU(),
            nn.Dropout(.15),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )
        # self.layer3_2 = nn.Sequential(
        #     nn.Conv1d(nch//4, nch // 4, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(nch // 4),
        #     nn.ReLU(),
        #     nn.Dropout(.25),
        # )
        self.layer4 = nn.Sequential(
            nn.Conv1d(nch // 4, nch // 8, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(nch // 8),
            nn.ReLU(),
            nn.Dropout(.15),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )
        # self.layer5 = nn.Sequential(
        #     nn.Conv1d(nch//8, nch // 8, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(nch // 8),
        #     nn.ReLU(),
        #     nn.Dropout(.25),
        #     nn.MaxPool1d(kernel_size=2, stride=1)
        # )
        self.fc1 = nn.Linear(nch // 8 * (num_features - 4), num_classes * 3)
        self.dropout_fc1 = nn.Dropout(p=.5)
        self.fc2 = nn.Linear(num_classes * 3, num_classes * 2)
        self.dropout_fc2 = nn.Dropout(p=.5)
        self.classifier = nn.Linear(num_classes * 2, num_classes)

        # self.fc2 = nn.Linear(num_classes*2, num_classes)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.LogSigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # out = self.layer2_2(out)

        out = self.layer3(out)
        # out = self.layer3_2(out)

        out = self.layer4(out)
        # out = self.layer5(out)

        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout_fc1(out)
        out = self.fc2(out)
        out = self.dropout_fc2(out)
        out = self.classifier(out)
        # out = self.sigmoid(out)
        # out = self.fc2(out)
        # out = self.softmax(out)
        return out


def get_model_full(n_timesteps=128, n_classes=6, num_features=9, name_prefix='', kernel_size=3):
    return ConvNet_1D(num_timesteps=n_timesteps, num_classes=n_classes, name=name_prefix + '_original',
                      num_features=num_features, kernel_size=kernel_size).to(device)


def get_model_to_quantify(n_timesteps=128, n_classes=6, num_features=9, name_prefix='', kernel_size=3):
    return ConvNet_1D(num_timesteps=n_timesteps, num_classes=n_classes, name=name_prefix + '_quantized',
                      num_features=num_features, kernel_size=kernel_size).to(device)

#
# class AutoQuantizedNet(nn.Module):
#     def __init__(self, name, num_classes=10):
#         super(AutoQuantizedNet, self).__init__()
#         self.name = name
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.fc = nn.Linear(7 * 7 * 32, num_classes)
#
#     def forward(self, x):
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.maxpool(out)
#         out = self.relu(self.bn2(self.conv2(out)))
#         out = self.maxpool(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.fc(out)
#         return out
#
#
# model_auto = AutoQuantizedNet(name='autoquantize').to(device)
