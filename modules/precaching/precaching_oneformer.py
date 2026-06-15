from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")
model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")
