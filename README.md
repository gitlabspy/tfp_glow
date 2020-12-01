## An example of using tfp.bijectors.glow for image generation
Tensorflow Probability Glow example. A conditional version of Glow is provided [here](tfp_glow/bijectors)
Requirements:
- `tfp-nightly`
- `tf-nightly` or `tf-nightly-gpu`

Training:
```
hpams = Hyparams() # hyper parameters
trainer = GlowTrainer(hpams)
trainer.train()
```
Testing:
...
<br>

TODO:
- variational dequantization
- res-flow
