# hyper parameters.
import tensorflow_probability as tfp
tfb = tfp.bijectors

class Hyparams:
    # checkpoint setup
    cpkt_path = './checkpoints/glow'
    max_to_keep = 3

    # model setup
    output_shape = (32, 32, 3)
    num_glow_blocks = 3
    num_steps_per_block = 18
    coupling_bijector_fn = tfb.GlowDefaultNetwork
    exit_bijector_fn= tfb.GlowDefaultExitNetwork
    grab_after_block = None 
    use_actnorm = True

    # training setup
    LR = 1e-4 
    epochs = 100

    # dataset setup
    dataset = 'cifar10'
    shuffle = True
    reshuffle_each_iteration = True
    batch_size = 128

