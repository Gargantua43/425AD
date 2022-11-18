import torch
import numpy as np
from Dataset_pair import MyDataset
from torch.utils.data import DataLoader


class RankNet(torch.nn.Module):
    def __init__(self):
        super(RankNet, self).__init__()
        # 视频网络
        self.fc_visual = torch.nn.Linear(1280, 128)
        self.attention_visual = torch.nn.MultiheadAttention(128, 1)
        self.conv = torch.nn.Conv1d(in_channels=62, out_channels=64, kernel_size=128)
        # 音频网络
        self.fc_audio = torch.nn.Linear(128, 128)
        self.lstm = torch.nn.LSTM(128, 128, 1, batch_first=True)
        self.attention_audio = torch.nn.MultiheadAttention(128, 1)
        self.fc2_audio = torch.nn.Linear(128, 1)
        # 融合网络
        self.fc_fusion = torch.nn.Linear(90, 1)
        # 添加Dropout
        self.dropout = torch.nn.Dropout(p=0.5)
        # 预测概率公式Sigmoid
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_visual1, x_audio1, x_visual2, x_audio2):
        # 视频特征1输入
        y_visual_fc1 = self.fc_visual(x_visual1)
        y_visual_drop1 = self.dropout(y_visual_fc1)
        y_attention1, y_visual_weight1 = self.attention_visual(y_visual_drop1, y_visual_drop1, y_visual_drop1)
        y_visual_conv1 = self.conv(y_attention1)
        y_visual_reshape1 = y_visual_conv1.view(y_visual_conv1.size(0), -1)
        # 音频特征1输入
        y_audio_fc1 = self.fc_audio(x_audio1)
        y_audio_drop1 = self.dropout(y_audio_fc1)
        y_audio_lstm1, y_weight1 = self.lstm(y_audio_drop1)
        y_audio_attention1, y_audio_weight1 = self.attention_audio(y_audio_lstm1, y_audio_lstm1, y_audio_lstm1)
        y_fc21 = self.fc2_audio(y_audio_attention1)
        y_audio_reshape1 = y_fc21.view(y_fc21.size(0), -1)
        # 向量拼接1
        connect_feature1 = torch.cat([y_visual_reshape1, y_audio_reshape1], 1)
        y_ctr1 = self.fc_fusion(connect_feature1)
        # 视频特征2输入
        y_visual_fc2 = self.fc_visual(x_visual2)
        y_visual_drop2 = self.dropout(y_visual_fc2)
        y_attention2, y_visual_weight2 = self.attention_visual(y_visual_drop2, y_visual_drop2, y_visual_drop2)
        y_visual_conv2 = self.conv(y_attention2)
        y_visual_reshape2 = y_visual_conv2.view(y_visual_conv2.size(0), -1)
        # 音频特征2输入
        y_audio_fc2 = self.fc_audio(x_audio2)
        y_audio_drop2 = self.dropout(y_audio_fc2)
        y_audio_lstm2, y_weight2 = self.lstm(y_audio_drop2)
        y_audio_attention2, y_audio_weight2 = self.attention_audio(y_audio_lstm2, y_audio_lstm2, y_audio_lstm2)
        y_fc2 = self.fc2_audio(y_audio_attention2)
        y_audio_reshape2 = y_fc2.view(y_fc2.size(0), -1)
        # 向量拼接2
        connect_feature = torch.cat([y_visual_reshape2, y_audio_reshape2], 1)
        y_ctr2 = self.fc_fusion(connect_feature)
        # 输出预测概率
        pred = self.sigmoid(y_ctr1 - y_ctr2)

        return pred, y_ctr1, y_ctr2


# my_dataset = MyDataset("./Data_pair_finish.csv")
# train_loader = DataLoader(dataset=my_dataset, batch_size=64)
# # x_visual1 = np.random.random((1, 62, 1280))
# # x_audio1 = np.random.random((1, 26, 128))
# # x_visual2 = np.random.random((1, 62, 1280))
# # x_audio2 = np.random.random((1, 26, 128))
# # model = RankNet()
# # model = model.double()
# # model = model.train()
#
# model_rank = RankNet()
# model_rank = model_rank.double()
# model_rank = model_rank.train()
#
# for v1_train, a1_train, v2_train, a2_train, label1, label2 in iter(train_loader):
#     output = model_rank(v1_train, a1_train, v2_train, a2_train)
#     # print(output[0].shape)
#     # print(output[1].shape)
#     # print(output[2].shape)
#     # print("+++++++++++++")
#     # print(output)
#     print(label1)
#     print(label2)
#     print("+++++++++++")
# # z = model(torch.from_numpy(x_visual1), torch.from_numpy(x_audio1), torch.from_numpy(x_visual2),
# #           torch.from_numpy(x_audio2))
# # print(z[0])
# # print(z[1])
# # print(z[2])
