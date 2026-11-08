from tqdm import tqdm
import numpy as np
import pickle
import os
import pandas as pd

if __name__ == "__main__":
    csv_path = "/media/dist7t_/xiaochuan/SUGFW/data/Liver/Foreslices/splits/train.csv"
    csv = pd.read_csv(csv_path)
    csv = csv["image_pth"].tolist()
    csv = [int(x[-16:-13]) for x in csv]
    csv = list(set(csv))


    base_dir = "../data/Liver/Foreslices/uncertainty"
    save_path = "../data/Liver/Foreslices/uncertainty/uncertainty_10_avg.pkl"
    all_files = [os.path.join(base_dir, file) for file in os.listdir(base_dir) if file.endswith("_10.pkl")]
    alls = []
    for file in all_files:
        if int(os.path.basename(file)[4:7]) in csv:
            alls.append(file)
    alls.sort()
    uncertainty = np.zeros((512, 512))
    for file in tqdm(alls):
        with open(file, 'rb') as f:
            file = pickle.load(f)
        uncertainty = uncertainty + file
    
    uncertainty /= len(all_files)
    with open(save_path, 'wb') as f:
        pickle.dump(uncertainty, f)

    

        

        
        

    
