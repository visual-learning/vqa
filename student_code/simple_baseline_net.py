import torch.nn as nn
from external.googlenet.googlenet import googlenet


class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self): # 2.2 TODO: add arguments needed
        super().__init__()
	    ############ 2.2 TODO
        self.image_feature_extractor = None
        self.word_feature_extractor = None
        self.classifier = None
	    ############

    def forward(self, image, question_encoding):
	    ############ 2.2 TODO

	    ############
        raise NotImplementedError()
