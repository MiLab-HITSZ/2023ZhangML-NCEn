# coding=utf-8
import time
# from selenium import webdriver
import threading
import os
from itertools import product
import pynvml
from multiprocessing import Pool
from multiprocessing import Process

data = ['ncen_2007']
# para_norm = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10,
#              0.12, 0.14, 0.16, 0.18, 0.20, 0.22]
# para_cos = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
#             0.02, 0.02, 0.02, 0.02, 0.02, 0.02]

# model = ['resnet20']
# loss_type = ['norm_cos', 'norm_cos1', 'norm_cos2', 'norm_cos3', 'norm_cos4', 'norm_cos5',
#              'norm_cos6', 'norm_cos7', 'norm_cos8', 'norm_cos9', 'norm_cos10', 'norm_cos11']

para_norm = [0.2, 0.3, 0.5, 0.6]
para_cos = [0.2, 0.3, 0.5, 0.6]
# model = ['VGG16']
# model = ['resnet20']
loss_type = ['norm_cos2', 'norm_cos3', 'norm_cos4', 'norm_cos5']

NUM_EXPAND = 1024 * 1024
PID_NUM = 2
MODEL_NUM = 3


def fun(args):
    print(args)
    pynvml.nvmlInit()
    gpuDeviceCount = pynvml.nvmlDeviceGetCount()  # 获取Nvidia GPU块数
    os.system("CUDA_VISIBLE_DEVICES={} python train_{}.py --loss_type={} --para_norm={} --para_cos={}"
                  .format(args[0], args[1], args[2], args[3], args[4]))


def temp_run(args):
    import time, os, sched
    print(args)
    schedule = sched.scheduler(time.time, time.sleep)
    pidNum = 1

    def perform_command(cmd, inc):
        # print(cmd)
        os.system(cmd)
        print()
        print('task')

    def timming_exe(cmd, inc=60):
        schedule.enter(inc, 0, perform_command, (cmd, inc))
        schedule.run()

    pynvml.nvmlInit()
    gpuDeviceCount = pynvml.nvmlDeviceGetCount()  # 获取Nvidia GPU块数
    # timming_exe("ls", 1)
    while True:
        time.sleep(args[0] * 40)
        # if (pidNum <= PID_NUM):
        #     pidNum = pidNum + 2
        for i in range(gpuDeviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # 获取GPU i的handle，后续通过handle来处理
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # 通过handle获取GPU i的信息
            info_pid = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            # gpu_memory_total = info.total #GPU i的总显存
            gpu_memory_used = info.used / NUM_EXPAND  # 转为MB单位
            print()
            print("gpu_memory_used:", gpu_memory_used, "MB", "-", "GPU:", i, "-", "num of PID:", len(info_pid))
            print()
            if gpu_memory_used < 10000 and len(info_pid) < PID_NUM:
                # print(info_pid)
                timming_exe("CUDA_VISIBLE_DEVICES={} python train_{}.py --loss_type={} --para_norm={} --para_cos={}"
                            .format(i, args[1], args[2], args[3], args[4]), 1)
                exit(0)


def dateset1():
    global gpu_index
    gpu_list = list(range(0, 6))
    gpu_num = len(gpu_list)

    # data = ['fashion-mnist', 'cifar10']
    # model = ['resnet20']
    # loss_type = ['norm_cos', 'norm_cos1', 'norm_cos2', 'norm_cos3', 'norm_cos4', 'norm_cos5']
    #
    # para_norm = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75]
    # para_cos = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    # args1 = product(data, loss_type, para_norm, para_cos)
    # args = [list(x) for x in args1]

    # print(args)
    args = [[a] + [para_norm[i]] for i, a in enumerate(loss_type)]
    args = [a + [para_cos[i]] for i, a in enumerate(args)]
    # data_model = product(data, model)
    # args = [(list(x) + a) for x in data_model for a in args]
    args = [(list(data) + a) for a in args]
    gpu_index = [gpu_list[i % gpu_num] for i in range(len(args))]
    #
    args = [[gpu_index[i]] + a for i, a in enumerate(args)]
    # print(args)
    # #
    for i in range(len(args)):
        p = Process(target=temp_run, args=(args[i],))
        p.start()


if __name__ == '__main__':
    dateset1()
