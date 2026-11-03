"""Deterministic clean canvas for a controlled local-repair ablation, not a learned inpaint model."""
import cv2
import numpy as np

METHODS=('instance_repair_count','instance_repair_count_init','instance_repair_count_blend','instance_repair_count_clean')
FLAGS={'instance_repair_count_init':(True,False),'instance_repair_count_blend':(False,True),'instance_repair_count_clean':(True,True)}


def clean_canvas(image, edit_mask, foreground, foreground_mask):
    """Erase only permitted pixels with Telea background fill, then paste the selected M2 foreground.

    This uses no extra diffusion samples. It cannot correct a malformed M2 proposal
    and must not enlarge the permitted mask to hide residuals.
    """
    mask=np.asarray(edit_mask,dtype=bool)
    fg_mask=np.asarray(foreground_mask,dtype=bool)
    if image.dtype!=np.uint8 or foreground.dtype!=np.uint8 or image.ndim!=3 or image.shape[-1]!=3 or foreground.shape!=image.shape or mask.shape!=image.shape[:2] or fg_mask.shape!=mask.shape:
        raise ValueError('Clean canvas requires aligned uint8 RGB images and masks')
    if not mask.any() or mask.all() or not (fg_mask & mask).any():
        raise ValueError('Clean canvas needs background context and visible target foreground')
    background=cv2.inpaint(image,mask.astype(np.uint8)*255,3,cv2.INPAINT_TELEA)
    background[~mask]=image[~mask]
    seed=background.copy();target=fg_mask & mask;seed[target]=foreground[target]
    return seed,background
