import os
import torch
from model import model_to_quantify, device
import numpy as np
import matplotlib.pyplot as plt


def main():
    saved_model_name = 'HAR_quantized.ckpt'
    dir_name = os.path.join(os.path.dirname(__file__), 'weights')
    saved_model = os.path.join(dir_name, saved_model_name)
    model_to_quantify.load_state_dict(torch.load(saved_model, map_location='cpu'))
    for name, weight in model_to_quantify.named_parameters():
        param = weight.detach().numpy()
        i = np.float(sum(param.flatten() != 0)) / np.float(len(param.flatten()))*100
        print(f' layer: {name}, Density: {i}')
        plt.hist(param.flatten(), bins=100)
        plt.show()

        np.histogram(param.flatten(), bins=10, range=None, normed=None, weights=None, density=None)

if __name__ == '__main__':
    main()