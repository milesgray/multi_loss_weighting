# Multi-Loss Weighting with Coefficient of Variations

Unofficial Implementation of [Multi-Loss Weighting with Coefficient of Variations](https://arxiv.org/pdf/2009.01717.pdf) -
credit goes to the authors **Rick Groenendijk**, **Sezer Karaoglu**, **Theo Gevers**, and **Thomas Mensink** for the easy to understand paper that
addressed an issue that I have recently been facing as I become more comfortable with creating custom loss functions and learning how different architectures
respond to them. The core idea presented is to keep track of a specific set of metrics for each loss and then use one of them to start dynamically weighing
each loss in real time (after each minibatch update). This is specifically a strategy for multiple losses applied to a single objective - *not* multi-task
learning (which is where the majority of this sort of research is focused on).

## Current Implementation

No reference code was released with the paper, so this code is completely written from scratch as I interpreted the paper. Since I already had started using an
abstraction around 'tracking' my loss values as to centralize the responsibility of updating any related metrics to each loss and to consolidate the logging
functionality that is almost always part of trackng loss values and their relatede metrics. This initial implementation was made for PyTorch and Comet.ml -
but it is only a few lines to change it to work for any other deep learning library. As time permits, I will update this repo to include a generic version
along with implementaions for at least TensorFlow 2.x and MXNet.

## Usage

Initialize a `LossTracker` class for each loss that will be used during training. I like putting things in dictionaries that have keys that line
up with other similar dictionaries, but this part is not neeeded, *technically*.

```python
from multi_loss_weighting import LossTracker
from comet_ml import experiment

max_loss = {
    "G": 100,
    "D": 100,
    "E": 100,
}
loss_weight = {
    "G": 0.1,
    "D": 0.001,
    "E": 1,
}
TRACKER_WARMUP = 30 
EPOC_BATCHES = TOTAL_FILES // BATCH_SIZE
loss_stats = {'G':  LossTracker("G", experiment, 
                                weight=loss_weight["G"], max_loss=max_loss["G"], 
                                block_size=EPOC_BATCHES, warmup=TRACKER_WARMUP),       
              'D':  LossTracker("D", experiment, 
                                weight=loss_weight["D"], max_loss=max_loss["D"], 
                                block_size=EPOC_BATCHES, warmup=TRACKER_WARMUP),
              'E':  LossTracker("E", experiment, 
                                weight=loss_weight["E"], max_loss=max_loss["AE"], 
                                block_size=EPOC_BATCHES, warmup=TRACKER_WARMUP)}
```

Then, during the training loop, just call the `update(loss)` method and pass the loss result in. Since this was made for PyTorch, there is a `do_backwards`
parameter that defaults to `True` and will have the loss object call `loss.backwards()` before passing its value off into the object to be stored. 

```python
real_samp = real_samples.to(DEVICE).requires_grad_()
fake_samples = models["G"](w, scale, alpha).detach()

lossD = losses["D"](models["E"], models["D"], alpha, real_samp, fake_samples, 
                                  gamma=5, use_bce=True)
models["optD"].zero_grad()
loss_stats["D"].update(lossD)
models["optD"].step()
```

## Issues

Unfortunately, I have not yet seen this provide the kind of automatically tuning loss weights that the paper promises. Do not let this scare you off, though!
I have only tested it in a very off the wall set of multi-loss objectives in a generative network. I have not yet tried to reproduce their results as reported 
in the paper.

Please let me know if anyone finds any problems with the implementation or if anyone finds it useful!

