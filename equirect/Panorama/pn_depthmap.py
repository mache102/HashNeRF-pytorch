"""
Test to retrieve depthmap from a single panorama tile.
"""
import asyncio
import base64
# import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import struct
import matplotlib.pyplot as plt
import requests
import re
import json

from Panorama.pn_retriever import get_panorama
from Panorama.pn_crop import crop_pano

def get_pano_id(url):
    """
    exactly 22 chars. 
    nested between "!1s" and "!2e"
    """

    pattern = re.compile(r"!1s(.{22})!2e")
    pano_id = pattern.search(url).group(1)
    return pano_id

def get_depth_map(pano_id):

    # x = 0
    # y = 0
    # url = f"https://cbk0.google.com/cbk?output=tile&panoid={pano_id}&zoom={zoom}&x={x}&y={y}&depth_map=true"
    # Set API endpoint URL
    endpoint = 'https://www.google.com/maps/photometa/v1'

    # Set API parameters
    params = {
        'authuser': '0',
        'hl': 'en',
        'gl': 'us',
        'pb': f'!1m4!1smaps_sv.tactile!11m2!2m1!1b1!2m2!1sen!2suk!3m3!1m2!1e2!2s{pano_id}!4m57!1e1!1e2!1e3!1e4!1e5!1e6!1e8!1e12!2m1!1e1!4m1!1i48!5m1!1e1!5m1!1e2!6m1!1e1!6m1!1e2!9m36!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3!1m3!1e3!2b1!3e2!1m3!1e3!2b0!3e3!1m3!1e8!2b0!3e3!1m3!1e1!2b0!3e3!1m3!1e4!2b0!3e3!1m3!1e10!2b1!3e2!1m3!1e10!2b0!3e3'
    }

    # Send GET request to API endpoint and retrieve response
    response = requests.get(endpoint, params=params, proxies=None)
    # response = requests.get(endpoint)

    # Extract image and depth map from response
    # print(response.links)
    response = response.content
    print(len(response))
    
    response = json.loads(response[4:])
    # recursive_pretty_print(response, name="x", max_len = 16) 
    s = response[1][0][5][0][5][1][2]


    # decode string + decompress zip
    depthMapData = parse(s)

    # parse first bytes to describe data
    header = parseHeader(depthMapData)

    # parse bytes into planes of float values
    data = parsePlanes(header, depthMapData)

    # w, h = header["width"], header["height"]
    # indices = data["indices"]
    # indices = np.array(indices).reshape((h, w))


    # planes = data["planes"]
    # print(len(planes))
    # plot the indices 'image'
    # plt.imshow(indices)
    # plt.show()
    # exit()

    # get all dist (d) values in plane dict
    # and sort descending
    # dists = [plane["d"] for plane in planes]
    # dists = np.array(dists)
    # dists = np.sort(dists)[::-1]
    # print(dists)
    # exit()

    # def plot_plane(ax, n, d):
    #     point_on_plane = np.array([0, 0, 0])
    #     ax.scatter(*point_on_plane, color='red', label='Point on Plane', s=100)
        
    #     # Create an arrow to represent the normal vector
    #     ax.quiver(*point_on_plane, *n, color='blue', label='Normal Vector', length=1.0)
        
    #     ax.text(n[0], n[1], n[2], f'n = {n}', fontsize=12)


    # # Create a figure
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot each plane
    # for plane in planes:
    #     n, d = plane["n"], plane["d"]
    #     plot_plane(ax, n, d)

    # # Set labels and display the plot
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()
    # exit()


    # import pdb; pdb.set_trace()

    # compute position and values of pixels
    depthMap = computeDepthMap(header, data["indices"], data["planes"])

    # process float 1D array into int 2D array with 255 values
    depth_map = depthMap["depthMap"]
    # for all values > max_depth, set to np.nan
    # max_depth = 4
    # depth_map[np.where(depth_map > max_depth)[0]] = np.nan
    depth_map = depth_map.reshape((depthMap["height"], depthMap["width"]))

    depth_map = np.fliplr(depth_map)
    return depth_map

def parse(b64_string):
    # fix the 'inccorrect padding' error. The length of the string needs to be divisible by 4.
    b64_string += "=" * ((4 - len(b64_string) % 4) % 4)
    # convert the URL safe format to regular format.
    data = b64_string.replace("-", "+").replace("_", "/")

    data = base64.b64decode(data)  # decode the string
    # data = zlib.decompress(data)  # decompress the data
    return np.array([d for d in data])


def parseHeader(depthMap):
    return {
        "headerSize": depthMap[0],
        "numberOfPlanes": getUInt16(depthMap, 1),
        "width": getUInt16(depthMap, 3),
        "height": getUInt16(depthMap, 5),
        "offset": getUInt16(depthMap, 7),
    }


def get_bin(a):
    ba = bin(a)[2:]
    return "0" * (8 - len(ba)) + ba


def getUInt16(arr, ind):
    a = arr[ind]
    b = arr[ind + 1]
    return int(get_bin(b) + get_bin(a), 2)


def getFloat32(arr, ind):
    return bin_to_float("".join(get_bin(i) for i in arr[ind : ind + 4][::-1]))


def bin_to_float(binary):
    return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]


def parsePlanes(header, depthMap):
    indices = []
    planes = []
    n = [0, 0, 0]

    for i in range(header["width"] * header["height"]):
        indices.append(depthMap[header["offset"] + i])

    for i in range(header["numberOfPlanes"]):
        byteOffset = header["offset"] + header["width"] * header["height"] + i * 4 * 4
        n = [0, 0, 0]
        n[0] = getFloat32(depthMap, byteOffset)
        n[1] = getFloat32(depthMap, byteOffset + 4)
        n[2] = getFloat32(depthMap, byteOffset + 8)
        d = getFloat32(depthMap, byteOffset + 12)
        planes.append({"n": n, "d": d})

    return {"planes": planes, "indices": indices}


def computeDepthMap(header, indices, planes):

    v = [0, 0, 0]
    w = header["width"]
    h = header["height"]

    depthMap = np.empty(w * h)

    sin_theta = np.empty(h)
    cos_theta = np.empty(h)
    sin_phi = np.empty(w)
    cos_phi = np.empty(w)

    for y in range(h):
        theta = (h - y - 0.5) / h * np.pi
        sin_theta[y] = np.sin(theta)
        cos_theta[y] = np.cos(theta)

    for x in range(w):
        phi = (w - x - 0.5) / w * 2 * np.pi + np.pi / 2
        sin_phi[x] = np.sin(phi)
        cos_phi[x] = np.cos(phi)

    for y in range(h):
        for x in range(w):
            planeIdx = indices[y * w + x]

            v[0] = sin_theta[y] * cos_phi[x]
            v[1] = sin_theta[y] * sin_phi[x]
            v[2] = cos_theta[y]

            if planeIdx > 0:
                plane = planes[planeIdx]
                t = np.abs(
                    plane["d"]
                    / (
                        v[0] * plane["n"][0]
                        + v[1] * plane["n"][1]
                        + v[2] * plane["n"][2]
                    )
                )
                depthMap[y * w + (w - x - 1)] = t
            else:
                depthMap[y * w + (w - x - 1)] = np.nan #9999999999999999999.0
    return {"width": w, "height": h, "depthMap": depthMap}


if __name__ == "__main__":
        
    url = "https://www.google.com/maps/@29.6950438,-95.3274211,3a,75y,225.52h,96.09t/data=!3m6!1e1!3m4!1sGo1mMho-l5se_RBtFpfgZA!2e0!7i16384!8i8192?entry=ttu"

    pano_id = get_pano_id(url)
    print(pano_id)
    # pano_id = "FRt0Bxijfyb9vDBvdzlbBQ"# input("Enter pano_id: ")
    zoom = 2
    pano_img, times = asyncio.run(get_panorama(pano_id=pano_id, zoom=zoom, debug=True))
    print(times)

    depth_map = get_depth_map(pano_id)

    pano_img = np.array(pano_img)
    pano_img = crop_pano(pano_img)
    # resize im to same shape as pano_img (upscale)

    row_indices = np.linspace(0, depth_map.shape[0] - 1, pano_img.shape[0]).astype(int)
    col_indices = np.linspace(0, depth_map.shape[1] - 1, pano_img.shape[1]).astype(int)

    # Use the indices to extract the corresponding pixels from 'im'
    depth_map = depth_map[row_indices][:, col_indices]

    # plot pano_img and depth_map sideby side 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.figsize = (20, 10)
    ax1.imshow(pano_img)
    ax2.imshow(depth_map)

    # tight 
    plt.tight_layout()
    plt.show()

    # cmap = plt.get_cmap('viridis')
    # cmap.set_bad(color='gray')
    # plt.imshow(1 - depth_map, cmap=cmap)
    # plt.colorbar()

    exit()
