import tensorflow as tf
import tensorflow_probability as tfp
from model import create_transformed_distribution
tfd = tfp.distributions


class GlowTrainer:
    def __init__(self, hypams):
        self.hypams = hypams
        self.load_dataset()
        self.model_init()
        self.optimizer_setup()
        self.checkpoints_setup()

    def checkpoints_setup(self):
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(0), model=self.model, optimizer=self.optimizer
        )
        ckpt_manager = tf.train.CheckpointManager(ckpt, self.hypams.cpkt_path, max_to_keep=self.hypams.max_to_keep)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            tf.print("Latest checkpoint is restored")
        self.ckpt = ckpt
        self.ckpt_manager = ckpt_manager

    def optimizer_setup(self):
        self.optimizer = tf.keras.optimizers.Adam(tf.Variable(self.hypams.LR, trainable=False))

    def load_dataset(self):
        if self.hypams.dataset == 'cifar10':

            @tf.function
            def augument(img):
                img = tf.cast(img, tf.float32)
                img = img + tf.random.uniform(tf.shape(img))
                img = img / (255.0 + 1.0)
                # img = img / 255.0
                img = tf.clip_by_value(img, 0.0, 1.0)
                img = img * (1 - 0.05) + 0.05 * 0.5 # might be harmful
                img = tf.math.log(img) - tf.math.log(1.0 - img)
                return img

            (train_data, _), (test_data, _)  = tf.keras.datasets.cifar10.load_data()
            self.train_data = tf.data.Dataset.from_tensor_slices(train_data).map(augument)
            self.test_data = tf.data.Dataset.from_tensor_slices(test_data).map(augument)
            if self.hypams.shuffle:
                self.train_data = self.train_data.shuffle(len(train_data))
                self.test_data = self.train_data.shuffle(len(test_data))
            self.train_data = self.train_data.batch(self.hypams.batch_size)
            self.test_data = self.train_data.batch(self.hypams.batch_size)
            if self.hypams.reshuffle_each_iteration:
                self.train_data = self.train_data.shuffle(self.hypams.batch_size, reshuffle_each_iteration=True)
        else:
            raise "Not Yet Implemented!"

    def postprocess(self, img):
        denominator = 1 + tf.math.exp(-img)
        img = 1 / denominator
        img = (img - 0.5 * 0.05) / (1.0 -  0.05) * 255.
        return tf.clip_by_value(tf.cast(img, tf.uint8), 0, 255)

    # ref : https://github.com/MokkeMeguru/TFGENZOO/blob/master/TFGENZOO/flows/quantize.py
    def uniform_quantization_loss(self, img):
        pre_logit_scale = tf.math.log(0.05) - tf.math.log(1.0 - 0.05)
        logdet_jacobian = tf.math.softplus(img) + tf.math.softplus(-img) - tf.math.softplus(pre_logit_scale)
        return tf.reduce_sum(logdet_jacobian, axis=[1,2,3])

    def model_init(self):
        self.model = create_transformed_distribution(
                output_shape=self.hypams.output_shape, 
                num_glow_blocks=self.hypams.num_glow_blocks,
                num_steps_per_block=self.hypams.num_steps_per_block,
                coupling_bijector_fn=self.hypams.coupling_bijector_fn,
                exit_bijector_fn=self.hypams.exit_bijector_fn,
                grab_after_block=self.hypams.grab_after_block,
                use_actnorm=self.hypams.use_actnorm
            )

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            log_prob_loss = - (self.model.log_prob(x) + self.uniform_quantization_loss(x))
        variables = tape.watched_variables()
        grads = tape.gradient(log_prob_loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))
        return tf.reduce_mean(log_prob_loss)

    def sample_images(self, nsample):
        pass

    def train(self):
        for epoch in range(self.hypams.epochs):
            count = 0
            for td in self.train_data:
                loss = self.train_step(td)
                n_bits = 8
                bpd = loss / (tf.math.log(2.0) * tf.cast(tf.math.reduce_prod(list(self.hypams.output_shape)), tf.float32)) + n_bits
                print(
                        'iteration : ', count,
                        ' ,loss : ', loss.numpy(),
                        ' ,bit/dims : ', bpd.numpy(),
                        ' ,epoch : ', epoch,
                        ' **',
                        end = '\r' 
                    )
                count += 1
            ckpt_save_path = self.ckpt_manager.save()
