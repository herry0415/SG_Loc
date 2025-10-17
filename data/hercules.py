import os
import h5py
import torch
import numpy as np
from torch.utils import data
import struct, sys
import open3d as o3d
import MinkowskiEngine as ME
from utils.pose_util import process_poses, poses_to_matrices
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Hercules(data.Dataset):
    def __init__(self, 
                 data_path,
                 train=True,
                 valid=False,
                 voxel_size=0.3,
                 augment=[],
                 ):

        self.dataset_root = data_path
        sequence_name='Library'
        self.sequence_name = sequence_name #['Library', 'Mountain', 'Sports']
        self.train = train

        self.augment = augment
        self.data_dir = os.path.join(self.dataset_root, sequence_name)
        
         # 根据 sequence_name 和 train/val 设置序列
        seqs = self._get_sequences(sequence_name, train)
    
        ps = {}
        ts = {}
        vo_stats = {}
        self.pcs = []
        for seq in seqs:
            seq_dir = os.path.join(self.data_dir, seq)

            # h5_path = os.path.join(seq_dir, 'radar_poses.h5')
         
        
            # pose_file_path = os.path.join(seq_dir, 'PR_GT/newContinental_gt.txt')
            pose_file_path = os.path.join(seq_dir, 'PR_GT/Aeva_gt.txt')
            ts_raw = np.loadtxt(pose_file_path, dtype=np.int64, usecols=0) # float读取数字丢精度
            ts[seq] = ts_raw
            
            pose_file = np.loadtxt(pose_file_path) #保证pose值不变
            p = poses_to_matrices(pose_file) # (n,4,4) #毫米波雷达坐标系
            ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))     #  (n, 12)
            
            # write to h5 file
            # print('write interpolate pose to ' + h5_path)
            # h5_file = h5py.File(h5_path, 'w')
            # h5_file.create_dataset('valid_timestamps', data=np.asarray(ts[seq], dtype=np.int64))
            # h5_file.create_dataset('poses', data=ps[seq])
            
            # else:
            #     print("load " + seq + ' pose from ' + h5_path)
            #     h5_file = h5py.File(h5_path, 'r')
            #     ts[seq] = h5_file['valid_timestamps'][...]
            #     ps[seq] = h5_file['poses'][...]
            #     print(f'pose len {len(ts[seq])}')
           
            self.pcs.extend(os.path.join(seq_dir, 'LiDAR/np8Aeva', str(t) + '.bin') for t in ts[seq])
            
            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}
            
        # read / save pose normalization information
        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))
        
        pose_stats_filename = os.path.join(self.data_dir, self.sequence_name + '_lidar'+'_pose_stats.txt')
        if self.train:
            self.mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)  # (3,)
            self.std_t = np.std(poses[:, [3, 7, 11]], axis=0)  # (3,)
            np.savetxt(pose_stats_filename, np.vstack((self.mean_t, self.std_t)), fmt='%8.7f')
            print(f'saving pose stats in {pose_stats_filename}')
        else:
            self.mean_t, self.std_t = np.loadtxt(pose_stats_filename)
        
         # convert the pose to translation + log quaternion, align, normalize
         
        self.poses = np.empty((0, 6))
        self.rots = np.empty((0, 3, 3))
        
        for seq in seqs:
            pss, rotation, pss_max, pss_min = process_poses(poses_in=ps[seq], mean_t=self.mean_t, std_t=self.std_t,
                                                            align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                                                            align_s=vo_stats[seq]['s'])
        
            self.poses = np.vstack((self.poses, pss))
            self.rots = np.vstack((self.rots, rotation))
            
        self.voxel_size = voxel_size
        
        if train:
            print("train data num:" + str(len(self.poses)))
        else:
            print("valid data num:" + str(len(self.poses)))
            
    def _get_sequences(self, sequence_name, train):
        mapping = {
            'Library': (['Library_01_Day','Library_02_Night'], ['Library_03_Day']),
            'Mountain': (['Mountain_01_Day','Mountain_02_Night'], ['Mountain_03_Day']),
            'Sports': (['Complex_01_Day','Complex_02_Night'], ['Complex_03_Day'])
        }
        return mapping[sequence_name][0] if train else mapping[sequence_name][1]
    
    def __getitem__(self, index):
        scan_path = self.pcs[index]
        # pts, extra,_ = bin_to_pcd(scan_path, sensor_type='Aeva')
        # scan = np.hstack([pts, extra])[:,:3] #xyz baseline
        scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 8)[:,:3]
        scan = np.ascontiguousarray(scan)
        
        pose = self.poses[index]  # (6,)
        rot = self.rots[index]
        # ground truth
        scan_gt = (rot @ scan.transpose(1, 0)).transpose(1, 0) + pose[:3].reshape(1, 3)
        scan_gt_s8 = np.concatenate((scan, scan_gt), axis=1) #! Nx6 (x1 y1 z1 x2 y2 z2)
        
        coords, feats= ME.utils.sparse_quantize(
            coordinates=scan,
            features=scan,
            quantization_size=self.voxel_size)

        coords_s8, feats_s8 = ME.utils.sparse_quantize(
            coordinates=scan,
            features=scan_gt_s8,
            quantization_size=self.voxel_size*8)

        return (coords, feats, coords_s8, feats_s8, rot, pose)

    def __len__(self):
        return len(self.poses)