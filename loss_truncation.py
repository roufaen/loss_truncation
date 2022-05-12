import numpy as np


class LossTruncator:

    def __init__(self, percentile=0.5, len=5000, recompute=5000):
        super().__init__()
        self.keepc = 1. - percentile
        self.len = len
        self.recompute = recompute
        self.vals = np.zeros(self.len)
        self.reset()

    def reset(self):
        self.count = 0
        self.last_computed = 0
        self.percentile_val = float('inf')
        self.cur_idx = 0

    def truncate_loss(self, loss):
        non_zero_loss_elems = loss[loss.nonzero(as_tuple=True)]
        self.last_computed += non_zero_loss_elems.numel()
        self.count += non_zero_loss_elems.numel()
        if self.count < len(self.vals):
            self.vals[self.count - non_zero_loss_elems.numel():self.count] = non_zero_loss_elems.detach().cpu().numpy().flatten()
            self.cur_idx += non_zero_loss_elems.numel()
            return loss
        else:
            for item in non_zero_loss_elems:
                self.vals[self.cur_idx] = item
                self.cur_idx += 1
                if self.cur_idx >= len(self.vals):
                    self.cur_idx = 0

        if self.last_computed > self.recompute:
            self.percentile_val = np.percentile(self.vals, self.keepc * 100)
            self.last_computed = 0

        mask = (loss < self.percentile_val).type(loss.dtype)
        return loss * mask
