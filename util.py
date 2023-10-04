import torch 
import numpy as np

def get_transform_matrix(translation, rotation):
    """
    torch tensor transformation matrix 
    from translation and rotation
    
    translation: (3,) torch tensor
    rotation: (2,) torch tensor
    ^ theta and phi angles
    
    returns: (4, 4) torch tensor
    """
    transform_matrix = torch.zeros((4, 4))
    transform_matrix[3, 3] = 1.0
    transform_matrix[:3, 3] = translation
    
    # Calculate rotation matrix from theta and phi
    theta, phi = rotation
    
    # Construct the rotation matrix
    theta_matx = torch.eye(3)
    theta_matx[0, 0] = torch.cos(theta)
    theta_matx[0, 1] = -torch.sin(theta)
    theta_matx[1, 0] = torch.sin(theta)
    theta_matx[1, 1] = torch.cos(theta)
    theta_matx[2, 2] = 1.0
    
    # Apply the phi rotation (rotation in the XY plane)
    phi_matx = torch.eye(3)
    phi_matx[0, 0] = torch.cos(phi)
    phi_matx[0, 1] = -torch.sin(phi)
    phi_matx[1, 0] = torch.sin(phi)
    phi_matx[1, 1] = torch.cos(phi)
    
    # Combine the two rotations
    rot_matx = torch.mm(theta_matx, phi_matx)
    
    # Copy the 3x3 rotation matrix into the top-left of the 4x4 transform_matrix
    transform_matrix[:3, :3] = rot_matx
    
    return transform_matrix

# kornia's create_meshgrid in numpy 
def create_meshgrid_np(H, W, normalized_coordinates=True):
    if normalized_coordinates:
        xs = np.linspace(-1, 1, W)
        ys = np.linspace(-1, 1, H)
    else:
        xs = np.linspace(0, W-1, W)
        ys = np.linspace(0, H-1, H)
    
    grid = np.stack(np.meshgrid(xs, ys), -1) # H, W, 2
    # transpose is not needed as the resulting grid 
    # is already the same as the one from kornia
    # grid = np.transpose(grid, [1, 0, 2]) # W, H, 2
    return grid