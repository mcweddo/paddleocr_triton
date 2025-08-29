import triton_python_backend_utils as pb_utils
import json
import numpy as np
from rec_postprocess import CTCLabelDecodeRobust, load_charset_from_yaml
import os




class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])

        self.input_names = []
        for input in model_config['input']:
            self.input_names.append(input['name'])

        self.output_names = []
        self.output_dtype = []
        for output in model_config['output']:
            self.output_names.append(output['name'])
            dtype = pb_utils.triton_string_to_numpy(output["data_type"])
            self.output_dtype.append(dtype)

        # charracter_dict=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/']
        dir_name = os.path.dirname(os.path.realpath(__file__))
        dict_path = os.path.join(dir_name, "inference.yml")

        charset = load_charset_from_yaml(dict_path)

        self.decoder = CTCLabelDecodeRobust(charset, merge_repeats=True)


    def execute(self, requests):
        responses = []
        for req in requests:
            logits_tensor = pb_utils.get_input_tensor_by_name(req, self.input_names[0])
            if logits_tensor is None:
                return [pb_utils.InferenceResponse(
                    error=pb_utils.TritonError("missing input 'recognition_postprocessing_input'"))]

            logits = logits_tensor.as_numpy()
            if logits.ndim == 3 and logits.shape[0] > 1:
                # Multi-crop in a single request; decode each independently then join.
                texts, scores = [], []
                for i in range(logits.shape[0]):
                    txt, sc = self.decoder.decode(logits[i])
                    if txt:
                        texts.append(txt)
                        scores.append(sc)
                # Simple aggregation: join with spaces; score = median of per-crop medians
                text = " ".join(texts) if texts else ""
                score = float(np.median(np.asarray(scores, dtype=np.float32))) if scores else np.nan
            else:
                text, score = self.decoder.decode(logits)

            out_text = np.array([text.encode("utf-8")], dtype=np.object_).astype(self.output_dtype[0])
            out_score = np.array([score], dtype=np.float32).astype(self.output_dtype[1])

            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(self.output_names[0], out_text),
                pb_utils.Tensor(self.output_names[1], out_score),
            ]))
        return responses


