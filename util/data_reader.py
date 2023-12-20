import os
import pickle
import logging

def read_pkl_data(pkl_path, img_path):
    logging.info('reading pickle file: '+ pkl_path)
    with open(pkl_path, "rb") as fp:
        data = pickle.load(fp)
        fp.close()
    
    root_dir = img_path
    if not os.path.exists(root_dir):
        root_dir = root_dir.replace('train', '').replace('val', '').replace('test', '')
    imgs, phases, steps = [], [], []
    for vid_name in sorted(data.keys()):
        paths = [
                os.path.join(root_dir, vid_name, f"{item['Frame_id']}.jpg")
            for item in data[vid_name]
        ]
        imgs.append(paths)
        phases.append([item['Phase_gt'] for item in data[vid_name]])
        steps.append([item['Step_gt'] for item in data[vid_name]])
    
    return imgs, phases, steps

