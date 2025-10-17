import os
import struct
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

def convert_raw_to_bin(src_fn, dst_fn, sensor_type="Aeva"):
    pts, extra = [], []

    with open(src_fn, 'rb') as f:
        if sensor_type == "Continental":
            rec = 29
            while (chunk := f.read(rec)):
                if len(chunk) < rec: break
                x, y, z, v, r = struct.unpack('<fffff',  chunk[:20])
                RCS           = struct.unpack('<B',      chunk[20:21])[0]
                az, el        = struct.unpack('<ff',     chunk[21:29])
                pts.append([x, y, z])
                extra.append([v, r, RCS, az, el])
            dim = 8

        elif sensor_type == "ContinentalObject":
            rec = 20
            while (chunk := f.read(rec)):
                if len(chunk) < rec: break
                x, y, z, vx, vy = struct.unpack('<fffff', chunk)
                pts.append([x, y, z])
                extra.append([vx, vy])
            dim = 5

        elif sensor_type == "Aeva":
            rec = 29
            while (chunk := f.read(rec)):
                if len(chunk) < rec: break
                x, y, z, refl, vel = struct.unpack('<fffff', chunk[:20])
                t_off   = struct.unpack('<I', chunk[20:24])[0]
                line_id = struct.unpack('<B', chunk[24:25])[0]
                inten   = struct.unpack('<f', chunk[25:29])[0]
                pts.append([x, y, z])
                extra.append([refl, vel, t_off, line_id, inten])
            dim = 8

        else:
            raise ValueError("Unsupported sensor_type")

    # numpy array
    arr = np.hstack([np.asarray(pts, dtype=np.float32),
                     np.asarray(extra, dtype=np.float32)]).astype(np.float32)

    # save
    arr.tofile(dst_fn)
    return dst_fn, arr.shape, dim


def parallel_convert(src_dir, dst_dir, sensor_type="Aeva", workers=8):
    os.makedirs(dst_dir, exist_ok=True)

    src_files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith(".bin")]
    futures = []
    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for src_fn in src_files:
            dst_fn = os.path.join(dst_dir, os.path.basename(src_fn))
            futures.append(executor.submit(convert_raw_to_bin, src_fn, dst_fn, sensor_type))

        for fut in as_completed(futures):
            dst_fn, shape, dim = fut.result()
            results.append((dst_fn, shape, dim))
            print(f"[OK] {dst_fn}, shape={shape}, dim={dim}")

    return results


if __name__ == "__main__":
    src_dir = "/home/data/ldq/HeRCULES/Sports/Complex_03_Day/LiDAR/Aeva/"
    dst_dir = "/home/data/ldq/HeRCULES/Sports/Complex_03_Day/LiDAR/np8Aeva/"
    parallel_convert(src_dir, dst_dir, sensor_type="Aeva", workers=16)
