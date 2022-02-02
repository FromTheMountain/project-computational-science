import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_validtion():

    all_files = ['cavity_vx', 'cavity_vx_ref', 'cavity_uy', 'cavity_uy_ref']
        
    plt.figure()

    all_y = []
    for filename in all_files:

        data = pd.read_csv('validation/' + str(filename) ,sep='\s+',header=None)

        x = data[0]
        y = data[1]
        all_y.append(y)
        plt.plot(x, y, label=filename)

    model = 100
    plt.title(f"U/V Profile vs Ghia et al. (ref), Re = {model}")
    
    MSE_vx = round(np.square(all_y[0] - all_y[1]).mean(), 5)
    MSE_uy = round(np.square(all_y[2] - all_y[3]).mean(), 5)

    print(MSE_vx, MSE_uy)

    all_files = [f'cavity_vx (MSE={MSE_vx})', 'cavity_vx_ref', f'cavity_uy (MSE={MSE_uy})', 'cavity_uy_ref']

    plt.legend(all_files)
    plt.show()





if __name__ == '__main__':

    plot_validtion()