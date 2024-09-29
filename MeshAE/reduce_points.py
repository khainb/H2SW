import os
import open3d as o3d
import  numpy as np
import tqdm
path = '/work/08387/kbn397/ls6/data/shapenet_psr'

dirs = [d for d in os.listdir(path) if os.path.isdir(path+'/'+d)]
for d in dirs:
    new_path = path+'/'+d
    subdirs= [sd for sd in os.listdir(new_path) if os.path.isdir(new_path+'/'+sd)]
    print(new_path)
    for sd in tqdm.tqdm(subdirs):
        final_path = new_path+'/'+sd+'/pointcloud.npz'
        X = np.load(final_path)

        current_pcd = o3d.geometry.PointCloud()
        current_pcd.points = o3d.utility.Vector3dVector(X['points'])
        current_pcd.normals = o3d.utility.Vector3dVector(X['normals'])
        current_pcd = current_pcd.farthest_point_down_sample(2048)

        X_final = np.concatenate([np.asarray(current_pcd.points),np.asarray(current_pcd.normals)],axis=1)
        X_final[:,-3:] = X_final[:,-3:]/np.sqrt(np.sum(X_final[:,-3:]**2,axis=1,keepdims=True))
        np.save( new_path+'/'+sd+'/pointcloud_reduced.npy',X_final)
