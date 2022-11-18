import torch
import numpy as np
import torch.optim as optim
import pandas as pd
import torch.nn as nn

from Fusion_Net_revise import RankNet
from Dataset_pair import MyDataset
from Dataset_pair_val import MyDataset_val
from torch.utils.data import DataLoader


def label(lab1, lab2):
    if lab1 >= lab2:
        lab = 1
    else:
        lab = 0
    return lab


def main():
    best_loss = 0.0
    save_path = './Ranknet.pth'
    epochs = 35
    # 导入训练集
    my_dataset = MyDataset("./Data_pair_finish.csv")
    train_loader = DataLoader(dataset=my_dataset, batch_size=65)

    # 导入验证集
    my_dataset_val = MyDataset_val("./Data_pair_val_finish.csv")
    val_loader = DataLoader(dataset=my_dataset_val, batch_size=1)
    val_num = len(my_dataset_val)

    # 实例化模型
    model_rank = RankNet()
    model_rank = model_rank.double()
    model_rank = model_rank.train()

    # 优化器、损失函数
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adadelta(model_rank.parameters(), lr=0.0001)

    # 训练
    for epoch in range(epochs):
        # train
        train_steps = 0  # 计数,运行到第几个数据对
        val_steps = 0
        running_loss = 0.0
        for v1_train, a1_train, v2_train, a2_train, label1, label2 in iter(train_loader):
            train_steps += 1
            optimizer.zero_grad()
            output = model_rank(v1_train, a1_train, v2_train, a2_train)
            # print(output)
            # 真实概率_修改位置
            real_pre_list = []
            for i in range(65):
                if label1[i] > label2[i]:
                    s = 1
                elif label1[i] < label2[i]:
                    s = -1
                else:
                    s = 0
                real_pre = 0.5 * (1 + s)
                real_pre_list.append(real_pre)
            # real_pre_torch = torch.from_numpy(np.array(real_pre_list))
            target = torch.tensor([real_pre_list])
            # print(target.shape)
            # 将预测概率和真实概率送入损失函数中
            target_squ = torch.squeeze(target)
            loss = criterion(output[0], torch.unsqueeze(target_squ.to(torch.double), dim=1))
            print(loss)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        # print(running_loss)
        # validate
        model_rank = model_rank.eval()
        val_loss = 0.0
        val_list = []
        running_loss_val = 0.0
        with torch.no_grad():
            for v1_val, a1_val, v2_val, a2_val, label1_val, label2_val in iter(val_loader):
                val_steps += 1
                output_val = model_rank(v1_val, a1_val, v2_val, a2_val)

                if label1_val > label2_val:
                    s_val = 1
                elif label1_val < label2_val:
                    s_val = -1
                else:
                    s_val = 0
                real_pre_val = 0.5 * (1 + s_val)
                target_val = torch.tensor([real_pre_val])

                loss_val = criterion(output_val[0], torch.unsqueeze(target_val.to(torch.double), 0))
                running_loss_val += loss_val.item()

        val_list.append(running_loss_val / val_steps)
        col = ['val_loss']
        info_array = np.array(val_list)
        df = pd.DataFrame(info_array, columns=col)
        df.to_csv('./val_list_loss' + str(epoch) + '.csv', encoding='utf-8')
        print('[epoch %d] train_loss: %.3f  val_loss: %.3f' %
              (epoch + 1, running_loss / train_steps, running_loss_val / val_steps))

        if val_loss > best_loss:
            best_loss = val_loss
            torch.save(model_rank.state_dict(), save_path)


if __name__ == '__main__':
    main()
