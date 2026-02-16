import numpy as np
import cv2
import math

class RecPreprocess:
    rec_image_shape = (3, 48, 320)

    def run(self, img_raw, dt_boxes):
        crop_coords = [crop.astype(int) for crop in dt_boxes]
        return self.crop_imgs(img_raw, crop_coords)  # (num_boxes, 3, 48, 320)

    def crop_imgs(self, img_raw, crop_coords):
        if not crop_coords:
            c, h, w = self.rec_image_shape
            return np.empty((0, c, h, w), dtype=np.float32)

        norm_img_batch = []
        for crop_coord in crop_coords:
            crop_img = img_raw[
                crop_coord[0][1] : crop_coord[2][1],
                crop_coord[0][0] : crop_coord[1][0],
                ::-1,  # BGR->RGB
            ].copy()

            norm_img = self.resize_norm_img_fixed(crop_img)
            norm_img_batch.append(norm_img[np.newaxis, :])

        return np.concatenate(norm_img_batch, axis=0)

    def resize_norm_img_fixed(self, img):
        """Resize with aspect ratio preserved, but ALWAYS pad/truncate to fixed W.

        This guarantees every request produces (3,48,320), which is required for
        Triton dynamic batching to actually combine multiple requests.
        """
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2], f"Expected {imgC} channels, got {img.shape[2]}"

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((imgC, imgH, imgW), dtype=np.float32)

        ratio = w / float(h)
        resized_w = int(math.ceil(imgH * ratio))
        resized_w = max(1, min(resized_w, imgW))  # clamp into [1, imgW]

        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype(np.float32).transpose((2, 0, 1)) / 255.0
        resized_image = (resized_image - 0.5) / 0.5

        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, :resized_w] = resized_image
        return padding_im
