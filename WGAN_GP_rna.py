# WGAN-GP

import numpy as np
import pandas as pd
import datetime

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, ReLU, Conv1D, Flatten, Reshape
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_DETERMINISTIC_OPS'] = '1'

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("CPU.")

#reproducible
SEED = 123
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# model parameters
noise_dim = 64 # input latent noise dimension for Gaussian Noises
seq_len = 185
vocab_size = 5

# train parameters
BATCH_SIZE = 64
EPOCHS = 12000
save_samples_num = 100
save_samples_iter = 10
save_checkpoints_iter = 1000

# encode table
rna_vocab = {"A":0,
             "C":1,
             "G":2,
             "U":3,
             "*":4}

rev_rna_vocab = {v:k for k,v in rna_vocab.items()}

def one_hot_encode(seq, SEQ_LEN=185):
    mapping = dict(zip("ACGU*", range(5)))  
    seq2 = [mapping[i] for i in seq]
    if len(seq2) < SEQ_LEN:
        extra = [np.eye(5)[4]] * (SEQ_LEN - len(seq2))
        return np.vstack([np.eye(5)[seq2] , extra])
    return np.eye(5)[seq2]

# load data
data_path = './dataset/training/data.csv'
data = pd.read_csv(data_path)

# real samples
data = data.loc[data['Label'] == 1]
apt_data = data['Aptamer'].drop_duplicates().values
apt_data = [x.upper() for x in apt_data]

apt_onehot = np.asarray([one_hot_encode(x) for x in apt_data])


# Define Generators and Discriminator
def resnet_block(x, out_dim): # resnet uses "SAME" padding to keep the dimension
    
    input_tensor = x
    x = Conv1D(filters=out_dim, kernel_size=5, strides=1, padding="SAME")(x)
    x = ReLU()(x)
    x = Conv1D(filters=out_dim, kernel_size=5, strides=1, padding="SAME")(x)
    x = ReLU()(input_tensor+0.3*x)
    
    return x


# generator model
def define_generator(): # -1xnoise_dim -> -1x185x5 (batch, steps, features)
    latent_dim=25
    
    x_input = Input(shape=(noise_dim))
    x = Dense(seq_len * latent_dim)(x_input)
    x = Reshape((seq_len, latent_dim))(x)
    x = resnet_block(x, out_dim=25)
    x = resnet_block(x, out_dim=25)
    x = resnet_block(x, out_dim=25)
    x = resnet_block(x, out_dim=25)
    x = resnet_block(x, out_dim=25)
    x = Conv1D(filters=vocab_size, kernel_size=1, activation='softmax', strides=1, padding="SAME")(x)
    
    model = Model(x_input, x, name="generator")
    
    return model


# critic model
def define_critic(): # 1x185x5 ->1
    
    x_input = Input(shape=(seq_len, vocab_size))
    x = Conv1D(filters=25, kernel_size=1, strides=1, padding="SAME")(x_input)
    x = resnet_block(x, out_dim=25)
    x = resnet_block(x, out_dim=25)
    x = resnet_block(x, out_dim=25)
    x = resnet_block(x, out_dim=25)
    x = resnet_block(x, out_dim=25)
    x = Flatten()(x)
    x = Dense(1)(x) # linear activation

    model = Model(x_input, x, name="discriminator")
    
    return model


# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def discriminator_loss(real_score, fake_score):
    real_loss = tf.reduce_mean(real_score)
    fake_loss = tf.reduce_mean(fake_score)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_score):
    return -tf.reduce_mean(fake_score)


# plot history
def plot_history(history):
    loss_gen = history.history['g_loss']
    loss_dis = history.history['d_loss']
    epochs = range(0, EPOCHS)
    plt.plot(epochs, loss_gen[0:], 'g', label='generator loss')
    plt.plot(epochs, loss_dis[0:], 'b', label='discriminator loss')
    plt.title('Generator and Discriminator Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./results/Loss.png")


# Create the WGAN-GP model
class WGAN(tf.keras.Model):
    def __init__(self,discriminator, generator, discriminator_extra_steps=5, gp_weight=10.0):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = noise_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_sequences, fake_sequences):
        """Calculates the gradient penalty.
        This loss is calculated on an interpolated sequence and added to the discriminator loss.
        """
        # Get the interpolated sequence
        e = tf.random.uniform([batch_size, seq_len, vocab_size], 0.0, 1.0)
        interpolated = e * real_sequences + (1.-e)*fake_sequences
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated sequence.
        grads = gp_tape.gradient(pred,[interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_sequences):
        if isinstance(real_sequences, tuple):
            real_sequences = real_sequences[0]
            
        # Get the batch size
        batch_size = tf.shape(real_sequences)[0]
        
        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary
        
        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator.
        
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_sequences = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_sequences, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_sequences, training=True)
                
                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_score=real_logits, fake_score=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_sequences, fake_sequences)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent (noise) vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_sequences = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_seq_logits = self.discriminator(generated_sequences, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_seq_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        
        return {"d_loss": d_loss, "g_loss": g_loss}
    
    
# Create a Keras callback that periodically saves generated sequences
class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self):
        self.latent_dim = noise_dim
        self.save_samples_num = save_samples_num
        self.save_samples_iter = save_samples_iter
        self.save_checkpoints_iter = save_checkpoints_iter
    
    def on_epoch_end(self, epoch, logs=None):# epoch starts from 0, so +1
        # generate samples
        if (epoch+1) % self.save_samples_iter == 0:
            random_latent_vectors = tf.random.normal(shape=(self.save_samples_num, self.latent_dim))
            generated_sequences = self.model.generator(random_latent_vectors)

            file_name = "./samples/gen_samples_epoch_"+str(epoch+1)+".txt"
            with open(file_name, "w") as f:
                for i in range(self.save_samples_num):
                    seq = generated_sequences[i].numpy()
                    seq_argmax = np.argmax(seq, 1)
                    s = "".join(rev_rna_vocab[ind] for ind in seq_argmax)+ "\n"
                    f.write(s)
            f.close()
            
        # save checkpoints
        if (epoch+1) % self.save_checkpoints_iter == 0:
            checkpoint_generator = "./checkpoints/generator_epoch_"+str(epoch+1)+".h5"
            self.model.generator.save(checkpoint_generator)
            
            checkpoint_discriminator = "./checkpoints/discriminator_epoch_"+str(epoch+1)+".h5"
            self.model.discriminator.save(checkpoint_discriminator)

            
if __name__ == "__main__":
    # Training

    # Instantiate the optimizer for both networks
    generator_optimizer = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

    # Instantiate the customer `GANMonitor` Keras callback.
    cbk = GANMonitor()

    # Get the wgan model
    g_model = define_generator()
    d_model = define_critic()

    wgan = WGAN(discriminator=d_model, generator=g_model, discriminator_extra_steps=5)

    # Compile the wgan model
    wgan.compile(d_optimizer=discriminator_optimizer, g_optimizer=generator_optimizer, g_loss_fn=generator_loss, d_loss_fn=discriminator_loss)

    # Start training
    history = wgan.fit(apt_onehot, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[cbk], verbose=1)

    # plot history
    plot_history(history)

