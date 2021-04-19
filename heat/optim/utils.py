"""
Utility functions for the heat optimizers
"""

import math
import torch

from typing import Optional, Dict


__all__ = ["DetectMetricPlateau"]


class DetectMetricPlateau(object):
    r"""
    Determine if a  when a metric has stopped improving.
    This scheduler reads a metrics quantity and if no improvement
    is seen for a 'patience' number of epochs, the learning rate is reduced.

    Adapted from `torch.optim.lr_scheduler.ReduceLROnPlateau <https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau>`_.

    Args:
        mode: str, optional
            One of `min`, `max`.
            In `min` mode, the quantity monitored is determined to have plateaued when
            it stops decreasing. In `max` mode, the quantity monitored is determined to
            have plateaued when it stops decreasing.\n
            Default: 'min'.
        patience: int, optional
            Number of epochs to wait before determining if there is a plateau
            For example, if `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only determine if there is a plateau after the
            3rd epoch if the loss still hasn't improved then.\n
            Default: 10.
        threshold: float, optional
            Threshold for measuring the new optimum to only focus on significant changes.\n
            Default: 1e-4.
        threshold_mode: str, optional
            One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode.\n
            Default: 'rel'.
        cooldown: int, optional
            Number of epochs to wait before resuming
            normal operation after lr has been reduced.\n
            Default: 0.
    """

    def __init__(
        self,
        mode: Optional[str] = "min",
        patience: Optional[int] = 10,
        threshold: Optional[float] = 1e-4,
        threshold_mode: Optional[str] = "rel",
        cooldown: Optional[int] = 0,
    ):  # noqa: D107
        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        self.reset()

    def get_state(self) -> Dict:
        """
        Get a dictionary of the class parameters. This is useful for checkpointing.
        """
        return {
            "patience": self.patience,
            "cooldown": self.cooldown,
            "cooldown_counter": self.cooldown_counter,
            "mode": self.mode,
            "threshold": self.threshold,
            "threshold_mode": self.threshold_mode,
            "best": self.best,
            "num_bad_epochs": self.num_bad_epochs,
            "mode_worse": self.mode_worse,
            "last_epoch": self.last_epoch,
        }

    def set_state(self, dic: Dict) -> None:
        """
        Load a dictionary with the status of the class. Typically used in checkpointing.

        Parameters
        ----------
        dic: Dictionary
            contains the values to be set as the class parameters
        """
        self.patience = dic["patience"]
        self.cooldown = dic["cooldown"]
        self.cooldown_counter = dic["cooldown_counter"]
        self.mode = dic["mode"]
        self.threshold = dic["threshold"]
        self.threshold_mode = dic["threshold_mode"]
        self.best = dic["best"]
        self.num_bad_epochs = dic["num_bad_epochs"]
        self.mode_worse = dic["mode_worse"]
        self.last_epoch = dic["last_epoch"]

    def reset(self) -> None:
        """
        Resets num_bad_epochs counter and cooldown counter.
        """
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def test_if_improving(self, metrics: torch.Tensor) -> bool:
        """
        Test if the metric/s is/are improving. If the metrics are better than the adjusted best value, they
        are set as the best for future testing.

        Parameters
        ----------
        metrics: torch.Tensor
            the metrics to test

        Returns
        -------
        True if the metrics are better than the best, False otherwise
        """
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)

        epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            return True
        return False

    @property
    def in_cooldown(self) -> bool:
        """
        Test if the class is in the cool down period
        """
        return self.cooldown_counter > 0

    def is_better(self, a: float, best: float) -> bool:
        """
        Test if the given value is better than the current best value. The best value is adjusted with the threshold

        Parameters
        ----------
        a: float
            the metric value
        best: float
            the current best value for the metric

        Returns
        -------
        boolean indicating if the metric is improving
        """
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            comp = best * rel_epsilon if best >= 0 else best * (1 + self.threshold)
            return a < comp

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode: str, threshold: float, threshold_mode: str) -> None:
        """
        Initialize the is_better function for comparisons later
        """
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        if mode == "min":
            self.mode_worse = math.inf
        else:  # mode == 'max':
            self.mode_worse = -math.inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
