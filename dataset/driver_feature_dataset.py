import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger
class DriverModelDataset(Dataset):
    def __init__(self, track_path_15, track_path_30, historic_windows_length, noise) -> None:
        self.track_path_15 = track_path_15
        self.track_path_30 = track_path_30
        self.historic_windows_length = historic_windows_length
        self.noise = noise
        self.data = []
        self.target = []        
        self.init_track_file()
 
    def init_track_file(self):
        """
        初始化轨迹文件, 从目录中加载轨迹文件
        """
        logger.info("正在读取数据集 {} ...".format(self.track_path_15))
        tracks: np.ndarray = np.load(self.track_path_15)
        tracks = tracks.transpose([0, 2, 1]).astype(np.float32)
        for i in range(tracks.shape[0]):
            track = tracks[i]
            for j in range(track.shape[0] - self.historic_windows_length - 1):
                data = track[j : j + self.historic_windows_length]
                target = np.array([(track[j + self.historic_windows_length][1] - track[j + self.historic_windows_length - 1][1]) / 0.04], dtype=np.float32)
                self.data.append(torch.FloatTensor(data) )
                self.target.append(torch.FloatTensor(target))
                
        tracks: np.ndarray = np.load(self.track_path_30)
        tracks = tracks.transpose([0, 2, 1]).astype(np.float32)
        for i in range(tracks.shape[0]):
            track = tracks[i]
            for j in range(track.shape[0] - self.historic_windows_length - 1):
                data = track[j : j + self.historic_windows_length]
                target = np.array([(track[j + self.historic_windows_length][1] - track[j + self.historic_windows_length - 1][1]) / 0.04], dtype=np.float32)
                self.data.append(torch.FloatTensor(data) )
                self.target.append(torch.FloatTensor(target))            
        
        logger.info("读取数据集成功，数据集大小: {}, noise: {}".format(self.__len__(), self.noise))
    
        
    def __len__(self):  
        return len(self.data)

    def __getitem__(self, index) :

        d = self.data[index]
        t = self.target[index]
        
        if self.noise > 1e-5:
            d = d + torch.normal(0, self.noise, size=(self.historic_windows_length, 4))
        return d, t

    
    
    
  