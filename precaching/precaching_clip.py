import sys
sys.path.append('.')
import clip

# TODO: the model that shall be used is: https://huggingface.co/kauevestena/clip-vit-base-patch32-finetuned-surface-materials

model, preprocess = clip.load("ViT-B/32",
                            #   jit=False
                              ) #Must set jit=False for training

