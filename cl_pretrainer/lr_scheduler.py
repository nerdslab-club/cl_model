import torch


class NoamOpt:
    """
    Copied from https://nlp.seas.harvard.edu/2018/04/03/attention.html#hardware-and-schedule

    A wrapper class for the Adam optimizer (or others) that implements learning rate scheduling.

    """

    def __init__(self, model_size, factor, warmup, optimizer, max_rate):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self._current_rate = 0
        self.max_rate = max_rate

    def get_current_rate(self):
        return self._current_rate

    def get_rate(self):
        return self._rate

    def step(self):
        """
        Update parameters and rate"
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            if rate > self.max_rate:
                self._current_rate = self.max_rate
            else:
                self._current_rate = rate
            p["lr"] = self._current_rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """
        Implement `lrate` above
        """
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )


def get_std_opt(model):
    return NoamOpt(
        model.encoder.hidden_dim,
        2,
        4000,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
        max_rate=0.00002,
    )
