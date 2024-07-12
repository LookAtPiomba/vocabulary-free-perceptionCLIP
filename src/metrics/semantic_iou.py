from typing import Union

import torch
from torchmetrics import Metric

#from src import utils


class SentenceIOU(Metric):
    """Metric to evaluate the intersection over union of words between two sentences.

    It takes as input a batch of predicted words with their scores and a batch of target sentences.
    The metric computes the intersection and union of the most probable predicted words (i.e.,
    top-1) and the target words and returns the intersection over union.

    Args:
        task (str): Task to perform. Currently only "multiclass" is supported.
    """

    def __init__(self, *args, task: str = "multiclass", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert task in ["multiclass"]
        self.task = task
        self.add_state("intersection", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor([]), dist_reduce_fx="sum")

    def update(self, values: list[str], targets: Union[list[str], list[list[str]]]) -> None:
        """Update state with data.

        Args:
            values (list[str]): Predicted words.
            targets (list[str] | list[list[str]]): Targets words.
        """
        if isinstance(targets, list) and isinstance(targets[0], list):
            assert len(targets[0]) == 1, "Only one target per sample is supported."
            targets = sum(targets, [])

        intersections = []
        unions = []
        for value, target in zip(values, targets):
            #value = max(value, key=value.get)  # take the word with the highest score
            value = "".join([c for c in value if c.isalnum() or c == " "]).lower()
            target = "".join([c for c in target if c.isalnum() or c == " "]).lower()

            intersections.append(len(set(value.split()) & set(target.split())))
            unions.append(len(set(value.split()) | set(target.split())))

        intersections = torch.tensor(intersections, device=self.device)
        unions = torch.tensor(unions, device=self.device)

        self.intersection = torch.cat([self.intersection, intersections])
        self.union = torch.cat([self.union, unions])

    def compute(self) -> torch.Tensor:
        """Compute the metric."""
        return torch.mean(self.intersection.float() / self.union.float())