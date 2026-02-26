import numpy as np
import cv2
import math

class RecPreprocess:
    rec_image_shape = (3, 48, 320)

    def run(self, img_raw, dt_boxes):
        crop_coords = [crop.astype(int) for crop in dt_boxes]

        if len(crop_coords) == 0:
            return np.zeros((0, 3, 48, 320), dtype=np.float32)

        crops = []
        max_wh_ratio = 1.0
        for crop_coord in crop_coords:
            crop_img = img_raw[crop_coord[0][1]:crop_coord[2][1],
                              crop_coord[0][0]:crop_coord[1][0], ::-1].copy()
            h, w = crop_img.shape[0:2]
            wh_ratio = w * 1.0 / h if h > 0 else 1.0
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
            crops.append(crop_img)

        imgC, imgH, imgW_base = self.rec_image_shape
        target_w = int(imgH * max_wh_ratio)
        target_w = max(32, min(target_w, 960))
        target_w = ((target_w + 31) // 32) * 32

        norm_img_batch = []
        for crop_img in crops:
            norm_img = self.resize_norm_img_fixed_width(crop_img, target_w)
            norm_img_batch.append(norm_img)

        norm_img_batch = np.stack(norm_img_batch, axis=0)
        return norm_img_batch

    def resize_norm_img_fixed_width(self, img, target_w):
        imgC, imgH, imgW_base = self.rec_image_shape

        if img.shape[2] != imgC:
            return np.zeros((imgC, imgH, target_w), dtype=np.float32)

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((imgC, imgH, target_w), dtype=np.float32)
        ratio = w / float(h)
        resized_w = int(math.ceil(imgH * ratio))

        if resized_w > target_w:
            resized_w = target_w

        resized_w = max(1, resized_w)

        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255.0
        resized_image -= 0.5
        resized_image /= 0.5

        padding_im = np.zeros((imgC, imgH, target_w), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image

        return padding_im
