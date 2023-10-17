import argparse
import matplotlib.pyplot as plt
import numpy as np

from .Equirec2Perspec import Equirectangular 

def crop_pano(image: np.ndarray) -> np.ndarray:

    rows_mask = np.all(image == 0, axis=1)[:, 0]
    image = image[~rows_mask]

    return image[:, :image.shape[0] * 2]

def main():
    # Parse arguments
    args = argparse.Namespace()

    # input_file = './test_panos/pano_0.png'
    # image = Image.open(input_file)
    # image = np.array(image)

    # # Remove duplicate slices
    # new_image = crop_pano(image)
    # print(new_image.shape)
    

    equ = Equirectangular('test_panos/pano_0_cropped.png')    # Load equirectangular image
    
    #
    # FOV unit is degree 
    # theta is z-axis angle(right direction is positive, left direction is negative)
    # phi is y-axis angle(up direction positive, down direction negative)
    # height and width is output image dimension 
    #
    img = equ.GetPerspective(80, 20, 40, 720, 1080) # Specify parameters(FOV, theta, phi, height, width)
    img = img[:, :, ::-1]
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    main()
