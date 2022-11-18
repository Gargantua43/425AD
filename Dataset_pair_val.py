import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MyDataset_val(Dataset):
    def __init__(self, pair_csv_path):
        self.Pair_csv_path = pair_csv_path
        self.df_Pair = pd.read_csv(self.Pair_csv_path, encoding='utf-8')

        # self.Audio_csv_path = audio_csv_path
        # self.Visual_csv_path = visual_csv_path
        # self.df_Audio = pd.read_csv(self.Audio_csv_path, encoding='utf-8')
        # self.df_Visual = pd.read_csv(self.Visual_csv_path, encoding='utf-8')

    # data = np.loadtxt(csv_path, delimiter=',')
    # self.len = data.shape[0]
    # self.train_data = torch.from_numpy(data[:, :])
    # print("数据以准备好.....")
    # print(self.train_data)

    def __getitem__(self, index):
        # 读pair文件中Visual1一列
        visual_csv_dir1 = self.df_Pair['filepath_visual_pair1'][index]
        visual_data1 = np.loadtxt(visual_csv_dir1, delimiter=',')
        visual_train1 = torch.from_numpy(visual_data1[:, :])

        # 读pair文件中Audio1一列
        audio_csv_dir1 = self.df_Pair['filepath_audio_pair1'][index]
        audio_data1 = np.loadtxt(audio_csv_dir1, delimiter=',')
        audio_train1 = torch.from_numpy(audio_data1[:, :])

        # 读pair文件中Visual2一列
        visual_csv_dir2 = self.df_Pair['filepath_visual_pair2'][index]
        visual_data2 = np.loadtxt(visual_csv_dir2, delimiter=',')
        visual_train2 = torch.from_numpy(visual_data2[:, :])

        # 读pair文件中Audio2一列
        audio_csv_dir2 = self.df_Pair['filepath_audio_pair2'][index]
        audio_data2 = np.loadtxt(audio_csv_dir2, delimiter=',')
        audio_train2 = torch.from_numpy(audio_data2[:, :])

        # 读pair文件中label1一列
        label_csv_dir1 = self.df_Pair['label_pair1'][index]
        label_data1 = np.loadtxt(label_csv_dir1, delimiter=',')
        label_train1 = torch.from_numpy(label_data1)

        # 读pair文件中label2一列
        label_csv_dir2 = self.df_Pair['label_pair2'][index]
        label_data2 = np.loadtxt(label_csv_dir2, delimiter=',')
        label_train2 = torch.from_numpy(label_data2)

        return visual_train1, audio_train1, visual_train2, audio_train2, label_train1, label_train2
        # return self.train_data[index]

    def __len__(self):
        # audio_len = len(self.df_Audio)
        # visual_len = len(self.df_Visual)
        pair_len = len(self.df_Pair)
        return pair_len


# my_dataset = MyDataset_val("./Data_pair_val_finish.csv")
# train_loader = DataLoader(dataset=my_dataset, batch_size=1)
# print(len(my_dataset))
# for v1_train, a1_train, v2_train, a2_train, label1, label2 in iter(train_loader):
#     print(v1_train.shape)
#     print(a1_train.shape)
#     print(v2_train.shape)
#     print(a2_train.shape)
#     print(label1.shape)
#     print(label2.shape)
#     print('**********')
