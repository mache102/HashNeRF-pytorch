import numpy as np 
# trilinear interpolation given a 3d coordinate and 8 voxel embeddings

def trilinear_interpolation(x, voxel_embeddings):
    if x.shape == (3,):
        # x is a single 3d coordinate
        x = x.reshape(1, 3)
    
    # step 1: interpolate along x-axis
    dx = x[:,0] - np.floor(x[:,0])

    c00 = voxel_embeddings[:,0]*(1-dx) + voxel_embeddings[:,4]*dx
    c01 = voxel_embeddings[:,1]*(1-dx) + voxel_embeddings[:,5]*dx
    c10 = voxel_embeddings[:,2]*(1-dx) + voxel_embeddings[:,6]*dx
    c11 = voxel_embeddings[:,3]*(1-dx) + voxel_embeddings[:,7]*dx

    # step 2: interpolate along y-axis
    dy = x[:,1] - np.floor(x[:,1])

    c0 = c00*(1-dy) + c10*dy
    c1 = c01*(1-dy) + c11*dy

    # step 3: interpolate along z-axis
    dz = x[:,2] - np.floor(x[:,2])

    c = c0*(1-dz) + c1*dz

    return c