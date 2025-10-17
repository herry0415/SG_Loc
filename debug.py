import os
import numpy as np

pose_path = '/home/data/ldq/HeRCULES/Library/Library_01_Day/Radar/multi_frame_w7/1724811869349597076_multi_w7.bin'

pc = np.fromfile(pose_path, dtype=np.float32).reshape(-1, 8)[:,:4]

np.savetxt('./radar.txt', pc)


