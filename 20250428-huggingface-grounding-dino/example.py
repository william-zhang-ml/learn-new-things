"""
Code based on the Hugging Face Grounding DINO example.
- Paper: arxiv.org/abs/2303.05499
- Example: huggingface.co/docs/transformers/main/en/model_doc/grounding-dino

My code performs inference using Grounding DINO.
It then draws the bounding boxes on the image and save it.
"""
from typing import List
from PIL import Image, ImageFont
from PIL.ImageDraw import Draw
import requests
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class GroundingDinoEngine:
    """Grounding DINO wrapper. """
    def __init__(
        self,
        box_thres=0.4,
        text_thresh=0.3,
        device: str = 'cuda:0'
    ) -> None:
        self._box_thresh = box_thres
        self._text_thresh = text_thresh
        self._device = device
        self._processor = AutoProcessor.from_pretrained(
            'IDEA-Research/grounding-dino-tiny'
        )
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            'IDEA-Research/grounding-dino-tiny'
        )
        self._model.to(device)

    def __call__(
        self,
        img: Image.Image,
        labels: List[str]
    ) -> None:
        """Single image inference.

        Args:
            img (Image.Image): image of interest
            labels (List[str]): objects to look for
        """
        inp = self._processor(images=img, text=[labels], return_tensors='pt')
        inp.to(self._device)
        with torch.no_grad():
            outputs = self._model(**inp)
        result = self._processor.post_process_grounded_object_detection(
            outputs,
            inp.input_ids,
            box_threshold=self._box_thresh,
            text_threshold=self._text_thresh,
            target_sizes=[img.size[::-1]]
        )[0]
        return result['boxes'], result['scores'], result['labels']


if __name__ == '__main__':
    engine = GroundingDinoEngine()
    image = Image.open(
        requests.get(
            'http://images.cocodataset.org/val2017/000000039769.jpg',
            stream=True,
            timeout=10
        ).raw
    )
    draw = Draw(image)
    font = ImageFont.truetype('arial.ttf', size=20)
    pred = engine(image, ['cat', 'remote control'])
    for box, _, label in zip(*pred):
        box = box.cpu().long().tolist()
        draw.rectangle(box, outline='steelblue', width=2)

        # text can be hard to see so draw it on top of a colored box
        text_box = draw.textbbox([box[0] + 5, box[1]], label, font=font)
        draw.rectangle(
            (
                text_box[0] - 5, text_box[1] - 5,
                text_box[2] + 5, text_box[3] + 5
            ),
            fill='steelblue'
        )
        draw.text((box[0] + 5, box[1]), label, align='center', font=font)
    image.save('example-image.png')
