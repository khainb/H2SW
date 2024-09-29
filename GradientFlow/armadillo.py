
import random
import time
from utils import *
import torch
import open3d as o3d
import copy
import numpy as np
for _ in range(1000):
    torch.randn(100)
    np.random.rand(100)
np.random.seed(1)
torch.manual_seed(1)
random.seed(1)

bunny = o3d.data.BunnyMesh()
bunny_mesh = o3d.io.read_triangle_mesh(bunny.path)
sphere_mesh = o3d.geometry.TriangleMesh.create_sphere()
bunny_mesh.compute_vertex_normals()
sphere_mesh.compute_vertex_normals()
bunny_pcd = bunny_mesh.sample_points_poisson_disk(10000)
sphere_pcd = sphere_mesh.sample_points_poisson_disk(10000)

points_bunny = np.asarray(bunny_pcd.points)
r = np.max(np.round(np.sqrt(np.sum(points_bunny ** 2, axis=1)),2))
normals_bunny = np.asarray(bunny_pcd.normals)

points_sphere = np.asarray(sphere_pcd.points)
normals_sphere = np.asarray(sphere_pcd.normals)

current_pcd = copy.deepcopy(sphere_pcd)
# o3d.visualization.draw_geometries([bunny_mesh])


target=np.concatenate([points_bunny,normals_bunny],axis=1)
source=np.concatenate([points_sphere,normals_sphere],axis=1)
device='cuda'
learning_rate = 0.01
N_step=5000
eps=0
seeds=[1,3,5]
Ls=[10,100]
rs=[10]
print_steps = list(range(0,5001,100))
print_steps= [i-1 for i in print_steps]
print_steps[0]=0
Y = torch.from_numpy(target).float().to(device)
N=target.shape[0]
ind1=0
ind2=1
np.random.seed(1)
torch.manual_seed(1)
random.seed(1)
source=torch.from_numpy(source).float()
points=[]
caltimes=[]
distances=[]
start = time.time()




np.random.seed(1)
torch.manual_seed(1)
random.seed(1)

for L in Ls:
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True,device=device)
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        points=[]
        caltimes=[]
        distances=[]
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance,cal_time=compute_true_Wasserstein_joint(X, Y), time.time() - start
                print("W {}:{} ({}s)".format(i + 1, distance,np.round(cal_time,2)))
                points.append(X.cpu().clone().data.numpy())
                caltimes.append(cal_time)
                distances.append(distance)

            optimizer.zero_grad()
            sw= SW(X,Y,L=L,device=device)
            loss= N*sw
            loss.backward()
            optimizer.step()
            X.data[:,-3:] = X.data[:,-3:]/torch.sqrt(torch.sum(X.data[:,-3:]**2,dim=1,keepdim=True))
        points.append(Y.cpu().clone().data.numpy())
        np.save("saved/SW_L{}_{}_{}_points_seed{}.npy".format(L,ind1,ind2,seed),np.stack(points))
        np.savetxt("saved/SW_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
        np.savetxt("saved/SW_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")

np.random.seed(1)
torch.manual_seed(1)
random.seed(1)

for r in rs:
    for L in Ls:
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            X=torch.tensor(source, requires_grad=True,device=device)
            optimizer = torch.optim.SGD([X], lr=learning_rate)
            points=[]
            caltimes=[]
            distances=[]
            start = time.time()
            for i in range(N_step):
                if (i in print_steps):
                    distance,cal_time=compute_true_Wasserstein_joint(X, Y), time.time() - start
                    print("G-W {}:{} ({}s)".format(i + 1, distance,np.round(cal_time,2)))
                    points.append(X.cpu().clone().data.numpy())
                    caltimes.append(cal_time)
                    distances.append(distance)
                    # fig = plt.figure()
                    # ax = fig.add_subplot(projection='3d')
                    # x= X.detach().numpy()
                    # ax.scatter(x[:, 1], x[:, 2], x[:, 0])
                    # plt.show()

                optimizer.zero_grad()
                sw= CSW(X,Y,L=L,device=device,r=r)
                loss= N*sw
                loss.backward()
                optimizer.step()
                X.data[:,-3:] = X.data[:,-3:]/torch.sqrt(torch.sum(X.data[:,-3:]**2,dim=1,keepdim=True))
            points.append(Y.cpu().clone().data.numpy())
            np.save("saved/GSW_L{}_r{}_{}_{}_points_seed{}.npy".format(L,r,ind1,ind2,seed),np.stack(points))
            np.savetxt("saved/GSW_L{}_r{}_{}_{}_distances_seed{}.txt".format(L,r,ind1,ind2,seed), np.array(distances), delimiter=",")
            np.savetxt("saved/GSW_L{}_r{}_{}_{}_times_seed{}.txt".format(L,r,ind1,ind2,seed), np.array(caltimes), delimiter=",")


np.random.seed(1)
torch.manual_seed(1)
random.seed(1)
for r in rs:
    for L in Ls:
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            X=torch.tensor(source, requires_grad=True,device=device)
            optimizer = torch.optim.SGD([X], lr=learning_rate)
            points=[]
            caltimes=[]
            distances=[]
            start = time.time()
            for i in range(N_step):
                if (i in print_steps):
                    distance,cal_time=compute_true_Wasserstein_joint(X, Y), time.time() - start
                    print("W {}:{} ({}s)".format(i + 1, distance,np.round(cal_time,2)))
                    points.append(X.cpu().clone().data.numpy())
                    caltimes.append(cal_time)
                    distances.append(distance)

                optimizer.zero_grad()
                sw= JSW(X,Y,L=L,r=r,device=device)
                loss= N*sw
                loss.backward()
                optimizer.step()
                X.data[:, -3:] = X.data[:, -3:] / torch.sqrt(torch.sum(X.data[:, -3:] ** 2, dim=1, keepdim=True))
            points.append(Y.cpu().clone().data.numpy())
            np.save("saved/H2SW_L{}_r{}_{}_{}_points_seed{}.npy".format(L,r,ind1,ind2,seed),np.stack(points))
            np.savetxt("saved/H2SW_L{}_r{}_{}_{}_distances_seed{}.txt".format(L,r,ind1,ind2,seed), np.array(distances), delimiter=",")
            np.savetxt("saved/H2SW_L{}_r{}_{}_{}_times_seed{}.txt".format(L,r,ind1,ind2,seed), np.array(caltimes), delimiter=",")