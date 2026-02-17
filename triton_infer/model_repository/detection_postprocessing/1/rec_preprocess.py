import numpy as np
import cv2
import math

class RecPreprocess:
    rec_image_shape = (3, 48, 320)

    def run(self, img_raw, dt_boxes):
        """
        Preprocess text crops for recognition.
        All crops are padded to the same width for batching.
        """
        crop_coords = [crop.astype(int) for crop in dt_boxes]

        if len(crop_coords) == 0:
            return np.zeros((0, 3, 48, 320), dtype=np.float32)

        # Extract crops and compute aspect ratios
        crops_with_ratios = []
        max_wh_ratio = 1.0

        for crop_coord in crop_coords:
            crop_img = img_raw[crop_coord[0][1]:crop_coord[2][1],
                              crop_coord[0][0]:crop_coord[1][0], ::-1].copy()
            h, w = crop_img.shape[0:2]
            wh_ratio = w * 1.0 / h if h > 0 else 1.0
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
            crops_with_ratios.append(crop_img)

        # Process all crops with the same target width (based on max ratio)
        norm_img_batch = []
        for crop_img in crops_with_ratios:
            norm_img = self.resize_norm_img(crop_img, max_wh_ratio)
            norm_img_batch.append(norm_img)

        # Stack all normalized images (now all have same shape)
        norm_img_batch = np.stack(norm_img_batch, axis=0)
        return norm_img_batch

    def resize_norm_img(self, img, target_wh_ratio):
        """
        Resize and normalize a single crop image.
        Uses quantized widths for better TensorRT kernel reuse.
        """
        imgC, imgH, imgW_base = self.rec_image_shape

        if img.shape[2] != imgC:
            return np.zeros((imgC, imgH, imgW_base), dtype=np.float32)

        # Calculate target width based on ratio, rounded to multiples of 32 for TensorRT
        target_w = int(imgH * target_wh_ratio)
        target_w = max(32, min(target_w, 960))  # Clamp to reasonable range
        target_w = ((target_w + 31) // 32) * 32  # Round to multiple of 32

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

        # Pad to target width
        padding_im = np.zeros((imgC, imgH, target_w), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image

        return padding_im
