import numpy as np
import datetime as dt
from datetime import date
from datetime import datetime
from matplotlib import pyplot as plt
import cmocean
from HYJfunction import *
import math

from netCDF4 import Dataset
import pandas as pd
from dateutil.relativedelta import relativedelta
import matplotlib.colors
import os
import cartopy
from cartopy import crs as ccrs
import matplotlib.ticker as mticker
import matplotlib.ticker as ticker
from scipy import ndimage
from multiprocessing import Pool, Manager
import cartopy.feature as cfeature
from scipy.ndimage import convolve
from scipy.signal import detrend
import pickle
import xarray as xr
import regionmask
from matplotlib.patches import Polygon
import matplotlib.path as mpath
from matplotlib.lines import Line2D
from multiprocessing import Pool, Manager
from matplotlib.colors import BoundaryNorm, ListedColormap
import seaborn as sns
import imageio
from scipy import stats
from collections import defaultdict
import random
import time

#%% 01 read the blocking and get the persistence list
# for each blocking event, it can be defined as related with AC/CC or not (1/0 on each grid point for the cluster)
# transfer to 2d array (time, lon)
typeid = 1
rgname = "SP"
ss = "ALL"
cyc = "AC"

# 02-2 read in the blocking flag, and transfer to 2d array (time, lon)
blockingEidArr = np.load(f'/scratch/bell/hu1029/LGHW/BlockingClustersEventID_Type{typeid}_{rgname}_{ss}.npy') # the blocking event index array
blockingEidArr = np.flip(blockingEidArr, axis=1) # flip the lat
print('blockingEidArr shape:', blockingEidArr.shape, flush=True)
blockingEidArr = np.where(blockingEidArr != -1, 1, 0) # transfer to 0/1
blockingEidArr_2D = np.any(blockingEidArr, axis=1).astype(int)
print('blockingEidArr_2D shape:', blockingEidArr_2D.shape, flush=True) # (nday*4, 360), 0 or 1
blockingEidArr_6hr = np.repeat(blockingEidArr_2D, 4, axis=0)
print('blockingEidArr_6hr shape after reshaping:', blockingEidArr_6hr.shape, flush=True)

def getDuration(pos_hw):
    #input:
    # pos_hw - the positions of Blocking days
    diff = np.diff(pos_hw)
    group_boundaries = np.where(diff > 1)[0] + 1  
    # get the length of each group
    groups = np.split(pos_hw, group_boundaries)
    group_lengths = [len(group) for group in groups]
    return group_lengths


# fake_blockingEidArr = np.zeros_like(blockingEidArr_6hr)  # create an empty array to store the fake blocking events
def generate_unique_random_lists_excludeList(n, duration_distribution, exclude_list):
    """
    生成包含连续整数的随机列表，每个子列表长度服从给定的 duration 分布，
    且每个整数在所有子列表中只出现一次。
    
    参数：
    n: int，整数范围的最大值（0 到 n）
    duration_distribution: list，代表长度的分布（例如 [1, 2, 3]）
    num_lists: int，需要生成的子列表个数
    
    返回：
    一个包含 num_lists 个子列表的列表，每个子列表内容为连续整数，且所有整数唯一。
    """
    result = []
    used_numbers = set(exclude_list)  # 记录已使用的整数
    num_lists = len(duration_distribution)
    
    for i in range(num_lists):
        while True:  # 直到找到合法子列表为止
            length = duration_distribution[i]
            if length > n - len(used_numbers):
                # 如果剩余数字不够生成该长度的子列表，则跳过
                continue
            
            # 在范围内生成合法的连续整数子列表
            possible_starts = [
                i for i in range(0, n - length + 1) 
                if all(num not in used_numbers for num in range(i, i + length))
            ]
            
            if not possible_starts:
                continue  # 没有可用起点，继续尝试
            
            start_point = random.choice(possible_starts)
            sublist = list(range(start_point, start_point + length))
            result.append(sublist)
            
            # 标记子列表中的数字为已使用
            used_numbers.update(sublist)
            break
    
    return result

# for lon in range(blockingEidArr_6hr.shape[1]):
#     blkdayindex = np.where(blockingEidArr_6hr[:, lon] == 1)[0]  # get the blocking event days for each lon
#     BlockPersis = getDuration(blkdayindex) 
#     shuffleIndex = generate_unique_random_lists_excludeList(62824, BlockPersis, blkdayindex)  # generate the random unique lists for each lon
#     allIndex = [item for sublist in shuffleIndex for item in sublist]
#     fake_blockingEidArr[np.array(allIndex),lon] = 1
    
#     print(f'lon {lon} done', flush=True)
    
def _worker(args):
    lon, col = args            # col 是该经度的一列：shape (T,)
    blkdayindex = np.where(col == 1)[0]
    BlockPersis = getDuration(blkdayindex)
    # 这里 62824 是时间轴长度 T，请用 len(col) 替代更稳妥
    shuffleIndex = generate_unique_random_lists_excludeList(len(col), BlockPersis, blkdayindex)
    allIndex = [i for sub in shuffleIndex for i in sub]
    return lon, np.array(allIndex, dtype=int)

if __name__ == "__main__":  # Windows/macOS 必须有这行
    T, L = blockingEidArr_6hr.shape
    fake_blockingEidArr = np.zeros_like(blockingEidArr_6hr, dtype=blockingEidArr_6hr.dtype)

    # 准备任务：只把每列的数据传给子进程，减少拷贝
    tasks = [(lon, blockingEidArr_6hr[:, lon].copy()) for lon in range(L)]

    # 并行跑
    with Pool(processes=128) as pool:     # 进程数按需改
        results = pool.map(_worker, tasks)

    # 汇总写回
    for lon, idx in results:
        if idx.size:
            fake_blockingEidArr[idx, lon] = 1
        print(f'lon {lon} done', flush=True)

np.save(f'/scratch/bell/hu1029/LGHW/fake_innonBlockdays_blockingEidArr_6hr_Type{typeid}_{rgname}_{ss}.npy', fake_blockingEidArr)
