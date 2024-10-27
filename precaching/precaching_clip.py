import sys
sys.path.append('.')
import clip

model, preprocess = clip.load("ViT-B/32",
                            #   jit=False
                              ) #Must set jit=False for training

