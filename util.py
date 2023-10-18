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


def create_expname(args):
    if args.i_embed==1:
        args.expname += "_hashXYZ"
    elif args.i_embed==0:
        args.expname += "_posXYZ"
    if args.i_embed_views==2:
        args.expname += "_sphereVIEW"
    elif args.i_embed_views==0:
        args.expname += "_posVIEW"
    args.expname += "_fine"+str(args.finest_res) + "_log2T"+str(args.log2_hashmap_size)
    args.expname += "_lr"+str(args.lrate) + "_decay"+str(args.lrate_decay)
    args.expname += "_RAdam"
    if args.sparse_loss_weight > 0:
        args.expname += "_sparse" + str(args.sparse_loss_weight)
    args.expname += "_TV" + str(args.tv_loss_weight)
    #args.expname += datetime.now().strftime('_%H_%M_%d_%m_%Y')

    return args.expname

def all_to_tensor(rays, device):
    """
    Iterate over all rays and convert to torch tensor
    (dataclass)
    """
    for key, value in rays.__dict__.items():
        if value is not None:
            rays.__dict__[key] = torch.Tensor(value).to(device)
    return rays

def shuffle_rays(rays):
    perm_anchor = rays.rgb 
    rand_idx = torch.randperm(perm_anchor.shape[0])

    for key, value in rays.__dict__.items():
        if value is not None:
            rays.__dict__[key] = value[rand_idx]

    return rays


def to_8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)

def psnr(pred_img, gt_img):
    return -10. * np.log10(np.mean(np.square(pred_img - gt_img)))