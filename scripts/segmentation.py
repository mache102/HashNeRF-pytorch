import json
import requests
import matplotlib.pyplot as plt
import numpy as np
import base64

from io import BytesIO
from PIL import Image

API_TOKEN = 'token'
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

fp = "imgs/gmap_rail1_rgb.png"
data = query(fp)

for item in data:
    if item["label"] == 'sky':
        break
mask_str = item['mask']
binary_data = base64.b64decode(mask_str)
with BytesIO(binary_data) as bio:
    image = Image.open(bio)
    mask = np.array(image)

plt.imshow(mask)
plt.show()

# save the mask to img directory