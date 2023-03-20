import numpy as np
import os
from utils import misc
import cv2
import matplotlib.pyplot as plt

path = "./data/modelnet40_c"

# path = 'scanobject_wbg_c'
# path = 'scanobject_nbg_c'
# path = 'shapenetcore_c'
corruptions = [
    'original',
    'uniform',
    'gaussian',
    'background',
    'impulse',
    'upsampling',
    'distortion_rbf',
    'distortion_rbf_inv', 'density',
    'density_inc',
    'shear',
    'rotation', 'cutout',
    'distortion',
    'occlusion',
    'lidar'
]
level = 5


def create_pc_frame(pc, text, roll=15, pitch=-45):
    points = misc.get_ptcloud_img(pc, roll=roll, pitch=pitch)[150:650,150:675,:]
    font = cv2.FONT_HERSHEY_TRIPLEX
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    textX = round((points.shape[1] - textsize[0]) / 2)
    cv2.putText(points, text, (textX, points.shape[0] - 30), font, 1, (0, 0, 0), 1, 2)

    return points


def create_title_image(pc, filename="title_image.png", path="./images/"):
    if not os.path.exists(path):
        os.makedirs(path)

    img1 = []
    img2 = []
    original = create_pc_frame(pc[0].squeeze().detach().cpu().numpy(), "Corrupted")
    masked = create_pc_frame(pc[1].squeeze().detach().cpu().numpy(), "Masked")
    recon_0 = create_pc_frame(pc[2].squeeze().detach().cpu().numpy(), "Reconstruction Grad 0")
    recon_20 = create_pc_frame(pc[3].squeeze().detach().cpu().numpy(), "Reconstruction Grad 20")
    img1.append(original)
    img1.append(masked)
    img2.append(recon_0)
    img2.append(recon_20)

    img1 = np.concatenate(img1, axis=1)
    img2 = np.concatenate(img2, axis=1)
    img = np.concatenate((img1, img2), axis=0)

    cv2.imwrite(path + filename, img)


def create_comp_images(pc, filename="title_image.png", path="./images/"):
    img1 = []
    original = create_pc_frame(pc[0].squeeze().detach().cpu().numpy(), "Ground Truth", roll=0, pitch=0)
    masked = create_pc_frame(pc[1].squeeze().detach().cpu().numpy(), "Input", roll=0, pitch=0)
    recon = create_pc_frame(pc[2].squeeze().detach().cpu().numpy(), "Reconstruction", roll=0, pitch=0)
    img1.append(original)
    img1.append(masked)
    img1.append(recon)
    img1 = np.concatenate(img1, axis=1)
    cv2.imwrite(path + filename, img1)


def create_confidence_plot(data_points):
    for samples in data_points:
        plt.plot(samples)
    plt.ylabel("Confidence")
    plt.xlabel("Gradient Steps")
    plt.xticks(np.arange(0, 22, step=2))
    plt.yticks(np.arange(0.5, 1.05, step=0.1))
    plt.margins(x=0)
    plt.savefig("images/confidence.png", bbox_inches='tight')


def create_video():
    final_frames = []
    pc_to_choose = 1200
    for pitch in range(0, 360, 5):
        img1 = []
        img2 = []
        for new_row_counter, corr in enumerate(corruptions):
            if corr == 'original':
                pc = np.load(os.path.join(path, 'data_' + corr + '.npy'))
            else:
                pc = np.load(os.path.join(path, 'data_' + corr + '_5.npy'))
            pc = [pc[pc_to_choose]]
            for p in pc:
                points = create_pc_frame(p, corr, pitch=pitch)
                if new_row_counter < 8:
                    img1.append(points)
                else:
                    img2.append(points)

        img1 = np.concatenate(img1, axis=1)
        img2 = np.concatenate(img2, axis=1)
        img = np.concatenate((img1, img2), axis=0)
        final_frames.append(img)

    if len(final_frames) == 0:
        print("no frames available")
    else:
        duration = len(final_frames)
        fps = 10
        height = final_frames[0].shape[0]
        width = final_frames[0].shape[1]
        out = cv2.VideoWriter(f'output_{pc_to_choose}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), True)
        for frame in final_frames:
            out.write(frame)
        out.release()

