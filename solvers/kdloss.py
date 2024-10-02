import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

class VanillaKD(nn.Module):
    def __init__(self, temp=20.0, distil_weight=0.5) -> None:
        super().__init__()
        self.temp = temp
        self.distil_weight = distil_weight
        self.cross_entropy = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        logging.info(f"Using Vanilla KD with: T={self.temp}, Lambda={self.distil_weight}")

    def forward(self, student_output, teacher_output, labels):
        alpha = self.distil_weight
        T = self.temp
        KD_loss = self.kl_loss(F.log_softmax(student_output/T, dim=1),
                                F.softmax(teacher_output/T, dim=1)) * (alpha * T * T) + \
                F.cross_entropy(student_output, labels) * (1. - alpha)

        return KD_loss