# mache102-nerfs

**[WIP]** NeRF implementations in pure pytorch for research and experimentation

## Models
- **Vanilla NeRF** 
    [Paper](https://arxiv.org/abs/2003.08934), [Reference Code](https://github.com/bmild/nerf)
    The original neural radiance fields model.
    
- **InstantNGP**
    [Paper](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf), [Reference Code](https://github.com/yashbhalgat/HashNeRF-pytorch/)
    Training speedup of NeRFs. **Note**: Implementation here is purely in pytorch (based on the reference code) so the acceleration is due to hash encoding 
    
- **NerfW** 
    [Paper](https://arxiv.org/abs/2008.02268), [Reference Code](https://github.com/kwea123/nerf_pl/tree/nerfw)
    NeRF in the Wild. Synthesis of novel views of complex scenes using only unstructured collections of in-the-wild photographs.

- **OmniNeRF**
    [Paper](https://arxiv.org/abs/2106.10859), [Reference Code](https://github.com/cyhsu14/OmniNeRF/)
    Novel view synthesis for equirectangular (panoramic) images.
## Setup
- Install dependencies: `pip install -r requirements.txt`


## Training

**[WIP]**
```py
python run_nerf.py --config configs/chair.txt --finest_res 512 --log2_hashmap_size 19 --lr 0.01 --lr_decay 10
```