from skimage.transform import rotate
from skimage.transform import warp
from skimage.transform import ProjectiveTransform
import random
import numpy as np

p = 0.1
intensity = 0.5
seed = 42
shuffle = False

def transform(Xb, yb):

    if yb is not None:
        batch_size = Xb.shape[0]
        image_size = Xb.shape[1]
        
        Xb = image_rotate(Xb, batch_size)
        Xb = apply_projection_transform(Xb, batch_size, image_size)

    return Xb, yb
    
def image_rotate(Xb, batch_size):
    """
    Applies random rotation in a defined degrees range to a random subset of images. 
    Range itself is subject to scaling depending on augmentation intensity.
    """
    for i in np.random.choice(batch_size, int(batch_size * p), replace = False):
        delta = 30. * intensity # scale by self.intensity
        Xb[i] = rotate(Xb[i], random.uniform(-delta, delta), mode = 'edge')
    return Xb   

def apply_projection_transform(Xb, batch_size, image_size):
    """
    Applies projection transform to a random subset of images. Projection margins are randomised in a range
    depending on the size of the image. Range itself is subject to scaling depending on augmentation intensity.
    """
    d = image_size * 0.3 * intensity
    for i in np.random.choice(batch_size, int(batch_size * p), replace = False):        
        tl_top = random.uniform(-d, d)     # Top left corner, top margin
        tl_left = random.uniform(-d, d)    # Top left corner, left margin
        bl_bottom = random.uniform(-d, d)  # Bottom left corner, bottom margin
        bl_left = random.uniform(-d, d)    # Bottom left corner, left margin
        tr_top = random.uniform(-d, d)     # Top right corner, top margin
        tr_right = random.uniform(-d, d)   # Top right corner, right margin
        br_bottom = random.uniform(-d, d)  # Bottom right corner, bottom margin
        br_right = random.uniform(-d, d)   # Bottom right corner, right margin

        transform = ProjectiveTransform()
        transform.estimate(np.array((
                (tl_left, tl_top),
                (bl_left, image_size - bl_bottom),
                (image_size - br_right, image_size - br_bottom),
                (image_size - tr_right, tr_top)
            )), np.array((
                (0, 0),
                (0, image_size),
                (image_size, image_size),
                (image_size, 0)
            )))
        Xb[i] = warp(Xb[i], transform, output_shape=(image_size, image_size), order = 1, mode = 'edge')

    return Xb