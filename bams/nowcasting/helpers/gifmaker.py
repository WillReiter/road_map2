import numpy as np
from ium_data.draw_frames import save_frame

def save_gif(single_seq, fname):
    """Save a single gif consisting of image sequence in single_seq to fname."""
    img_seq = [Image.fromarray(img.astype(np.float32) * 255, 'F').convert("L") for img in single_seq]
    img = img_seq[0]
    img.save(fname, save_all=True, append_images=img_seq[1:])


def save_imgs(seq, prefix):
    """Save several gifs.

    Args:
      seq: Shape (num_gifs, IMG_SIZE, IMG_SIZE)
      prefix: prefix-idx.gif will be the final filename.
    """
    if len(seq) < 10:        
        save_frame(seq[4], normalization=False, save_path="{}{}.png".format(prefix, '_30min_GT'))
        #np.save("{}-{}.npy".format(prefix, 4), seq[4])
    else:
        save_frame(seq[4], normalization=False, save_path="{}{}.png".format(prefix, '_30min_F')) 
        save_frame(seq[9], normalization=False, save_path="{}{}.png".format(prefix, '_60min_F'))
        #np.save("{}-{}.npy".format(prefix, 4), seq[4])
        #np.save("{}-{}.npy".format(prefix, 9), seq[9])

def save_imgs_all(seq, prefix):
    """Save several gifs.

    Args:
      seq: Shape (num_gifs, IMG_SIZE, IMG_SIZE)
      prefix: prefix-idx.gif will be the final filename.
    """
    if len(seq) <= 5:
        for i in range(len(seq)):
            save_frame(seq[i], normalization=False, save_path="{}_{}min.png".format(prefix, (i+1)*6))
    else:
        for i in range(len(seq)):
            save_frame(seq[i], normalization=False, save_path="{}_{}min.png".format(prefix, (i+1)*6))

