import triton_python_backend_utils as pb_utils
import json
import numpy as np
import cv2
import time
import os
from det_postprocess import DBPostProcess, crop_imgs
from rec_preprocess import RecPreprocess

class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        self.det_postprocess = DBPostProcess(thresh=0.3, box_thresh=0.6, max_candidates=1000, unclip_ratio=1.5,use_dilation=False, score_mode="fast")
        self.rec_preprocessor = RecPreprocess()

        # Must be <= text_recognition.max_batch_size for Triton batching to work reliably.
        # Also prevents pathological pages from exploding VRAM/latency.
        self.max_rec_crops = 256

        self.input_names = []
        for input in model_config['input']:
            self.input_names.append(input['name'])

        self.output_names = []
        self.output_dtype = []
        for output in model_config['output']:
            self.output_names.append(output['name'])
            dtype = pb_utils.triton_string_to_numpy(output["data_type"])
            self.output_dtype.append(dtype)

        self.rec_image_shape = (3, 48, 320)

    def execute(self, requests):
        st=time.time()
        responses = []
        for request in requests:
            det_output = pb_utils.get_input_tensor_by_name(
                request, self.input_names[0]
            )
            ori_img = pb_utils.get_input_tensor_by_name(
                request, self.input_names[1]
            )
            shape_list = pb_utils.get_input_tensor_by_name(
                request, self.input_names[2]
            )

            preds = det_output.as_numpy()
            img_raw = np.squeeze(ori_img.as_numpy(), axis=0)
            shape_list = shape_list.as_numpy()
            
            dt_boxes = self.det_postprocess(preds, shape_list)[0]['points']

            # Cap crops to match recognition batching limits
            if dt_boxes is None:
                dt_boxes = np.zeros((0, 4, 2), dtype=np.float32)
            if dt_boxes.shape[0] > self.max_rec_crops:
                dt_boxes = dt_boxes[: self.max_rec_crops]

            # Clip + filter degenerate boxes before cropping
            H, W = img_raw.shape[:2]
            clean = []
            for b in dt_boxes:
                b = b.astype(np.int32)
                b[:, 0] = np.clip(b[:, 0], 0, W - 1)
                b[:, 1] = np.clip(b[:, 1], 0, H - 1)
                x0, y0 = b[0]
                x1, y1 = b[2]
                if (x1 - x0) < 2 or (y1 - y0) < 2:
                    continue
                # Drop absurd aspect ratios (these are usually detection noise)
                ar = (x1 - x0) / float(y1 - y0)
                if ar > 50.0:
                    continue
                clean.append(b)
                if len(clean) >= self.max_rec_crops:
                    break
            dt_boxes = np.array(clean, dtype=np.int32)

            list_crop_img = self.rec_preprocessor.run(img_raw, dt_boxes)

            if os.environ.get("DEBUG_SHAPES", "0") == "1":
                print(f"[det_post] dt_boxes={len(dt_boxes)} crop_imgs={list_crop_img.shape}", flush=True)

            out_tensor_0 = pb_utils.Tensor(self.output_names[0], dt_boxes.astype(self.output_dtype[0])) 
            out_tensor_1 = pb_utils.Tensor(self.output_names[1], list_crop_img.astype(self.output_dtype[1])) 

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1]
            )
            
            responses.append(inference_response)
        return responses

