Module heat.optim.utils
=======================
Utility functions for the heat optimizers

Classes
-------

`DetectMetricPlateau(mode: str | None = 'min', patience: int | None = 10, threshold: float | None = 0.0001, threshold_mode: str | None = 'rel', cooldown: int | None = 0)`
:   Determine if a  when a metric has stopped improving.
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

    ### Instance variables

    `in_cooldown: bool`
    :   Test if the class is in the cool down period

    ### Methods

    `get_state(self) ‑> Dict`
    :   Get a dictionary of the class parameters. This is useful for checkpointing.

    `is_better(self, a: float, best: float) ‑> bool`
    :   Test if the given value is better than the current best value. The best value is adjusted with the threshold

        Parameters
        ----------
        a: float
            the metric value
        best: float
            the current best value for the metric

        Returns
        -------
        boolean indicating if the metric is improving

    `reset(self) ‑> None`
    :   Resets num_bad_epochs counter and cooldown counter.

    `set_state(self, dic: Dict) ‑> None`
    :   Load a dictionary with the status of the class. Typically used in checkpointing.

        Parameters
        ----------
        dic: Dictionary
            contains the values to be set as the class parameters

    `test_if_improving(self, metrics: torch.Tensor) ‑> bool`
    :   Test if the metric/s is/are improving. If the metrics are better than the adjusted best value, they
        are set as the best for future testing.

        Parameters
        ----------
        metrics: torch.Tensor
            the metrics to test

        Returns
        -------
        True if the metrics are better than the best, False otherwise
