import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils import bounding_box


def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument("--plot_dir", type=str, required=True,
                     help="path to directory containing pickle dumps")
    arg.add_argument("--epoch", type=int, required=True,
                     help="epoch of desired plot")
    args = vars(arg.parse_args())
    return args['plot_dir'], args['epoch']


def main(plot_dir, epoch):

    # read in pickle files
    with open(os.path.join(plot_dir, "g_{}.p".format(epoch)), "rb") as f:
        glimpses = pickle.load(f)

    with open(os.path.join(plot_dir, "l_{}.p".format(epoch)), "rb") as f:
        locations = pickle.load(f)

    # grab useful params
    patch_size = int(plot_dir.split('_')[2][0])
    num_glimpses = len(locations)
    num_imgs = glimpses.shape[0]
    img_shape = np.asarray([glimpses[0].shape[1:]])

    # denormalize coordinates
    coords = [0.5 * ((l+1.0) * img_shape) for l in locations]

    fig, axs = plt.subplots(nrows=1, ncols=num_imgs)
    # fig.set_dpi(100)

    # plot base image
    for j, ax in enumerate(axs.flat):
        ax.imshow(glimpses[j], cmap="Greys_r")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def updateData(i):
        color = 'r'
        co = coords[i]
        for j, ax in enumerate(axs.flat):
            for p in ax.patches:
                p.remove()
            c = co[j]
            rect = bounding_box(
                c[0], c[1], patch_size, color
            )
            ax.add_patch(rect)

    # animate
    anim = animation.FuncAnimation(
        fig, updateData, frames=num_glimpses, interval=500, repeat=True
    )

    # save as mp4
    name = plot_dir + 'epoch_{}.mp4'.format(epoch)
    anim.save(name, extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])


if __name__ == "__main__":
    args = parse_arguments()
    main(*args)
