from PIL import Image
import numpy as np
from datetime import datetime as dt


def save_image(image_in, path):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        return img

    val_block_size = int(image_in.shape[0] ** 0.5)
    preprocesed = preprocess(image_in)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(image_in.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    Image.fromarray(final_image).save(path)


def time_str():
    s = dt.now().isoformat()
    return s[:10] + " " + s[11:19]
