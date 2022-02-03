# Convert the png files in simulation_frames to a gif file for showing off in
# the repo.

import imageio
import os

files = [f"simulation_frames/{f}" for f in os.listdir("simulation_frames")]

with imageio.get_writer('movie.gif', mode='I', fps=40) as writer:
    for i in range(0, 10000, 16):
        print(f"Frame {i}", end="\r")
        image = imageio.imread(files[i])
        writer.append_data(image)
