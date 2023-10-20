first of all, cd to this directory

cd equirect


and now, commands for the tasks
1. to test panorama gathering (for a single location):
    python3 -m Panorama.pn_search  

    (by default the images are outputted in imgs/pn_cluster_test/)

2. obtain panorama & generate train + test augmentations:
    python3 generate_data.py  

    takes about 90 seconds to generate the samples (without occlusion func)
    images saved in data/ ...