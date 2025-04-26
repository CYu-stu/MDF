import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('..')
from datasets import dataloaders
from tqdm import tqdm
from torch.nn import NLLLoss


def get_score(acc_list):

    mean = np.mean(acc_list)
    interval = 1.96*np.sqrt(np.var(acc_list)/len(acc_list))

    return mean,interval


def meta_test(data_path,model,way,shot,pre,transform_type,query_shot=16,trial=10000,return_list=False):

    eval_loader = dataloaders.meta_test_dataloader(data_path=data_path,
                                                way=way,
                                                shot=shot,
                                                pre=pre,
                                                transform_type=transform_type,
                                                query_shot=query_shot,
                                                trial=trial)
    
    target = torch.LongTensor([i//query_shot for i in range(query_shot*way)]).cuda()

    acc_list = []

    for i, (inp,_) in tqdm(enumerate(eval_loader)):

        inp = inp.cuda()
        max_index = model.meta_test(inp,way=way,shot=shot,query_shot=query_shot)

        acc = 100*torch.sum(torch.eq(max_index,target)).item()/query_shot/way
        acc_list.append(acc)

    if return_list:
        return np.array(acc_list)
    else:
        mean,interval = get_score(acc_list)
        return mean,interval

# def meta_test(data_path, model, way, shot, pre, transform_type, query_shot=16, trial=10000, return_list=False, writer=None, iter_counter=0):

#     eval_loader = dataloaders.meta_test_dataloader(data_path=data_path,
#                                                    way=way,
#                                                    shot=shot,
#                                                    pre=pre,
#                                                    transform_type=transform_type,
#                                                    query_shot=query_shot,
#                                                    trial=trial)
    
#     target = torch.LongTensor([i // query_shot for i in range(query_shot * way)]).cuda()
#     criterion = NLLLoss().cuda()  # 使用相同的损失函数

#     acc_list = []
#     loss_list = []  # 记录损失值

#     for i, (inp, _) in tqdm(enumerate(eval_loader)):

#         inp = inp.cuda()
#         log_prediction = model(inp, way=way, shot=shot, query_shot=query_shot)

#         # 计算损失
#         loss = criterion(log_prediction, target)
#         loss_value = loss.item()
#         loss_list.append(loss_value)

#         if writer is not None:
#             writer.add_scalar('test_loss', loss_value, iter_counter + i)  # 将每次迭代的损失记录到 TensorBoard

#         # 计算准确率
#         _, max_index = torch.max(log_prediction, 1)
#         acc = 100 * torch.sum(torch.eq(max_index, target)).item() / query_shot / way
#         acc_list.append(acc)

#     if return_list:
#         return np.array(acc_list), np.array(loss_list)
#     else:
#         mean, interval = get_score(acc_list)
#         avg_loss = np.mean(loss_list)

#         # 如果你还想在最后记录平均损失
#         if writer is not None:
#             writer.add_scalar('avg_test_loss', avg_loss, iter_counter)

#         return mean, interval, avg_loss
