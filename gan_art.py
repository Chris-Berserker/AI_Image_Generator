import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from IPython import display
import PIL
from pathlib import Path

# GPU memory growth configuration to prevent OOM errors
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")
else:
    print("No GPU found, using CPU")

# Configuration class for easy parameter tuning
class GANConfig:
    def __init__(
        self,
        dataset='mnist',  # 'mnist', 'fashion_mnist', 'cifar10', or 'custom'
        custom_data_path=None,
        image_size=64,  # Increased from 28x28 to 64x64
        channels=3,  # Support for RGB images
        batch_size=128,  # Reduced batch size for better stability
        buffer_size=60000,
        z_dim=128,  # Increased latent dimension for more expressive generator
        epochs=100,
        save_every=10,  # Save checkpoints every N epochs
        learning_rate_g=2e-4,  # Adjusted learning rates
        learning_rate_d=2e-4,
        beta1=0.5,  # Adam optimizer parameters
        beta2=0.999,
        use_spectral_norm=True,  # Add spectral normalization for stability
        use_label_conditioning=False,  # Enable conditional GAN
        n_classes=10,  # Number of classes for conditional GAN
        use_attention=True,  # Self-attention layers
        dropout_rate=0.3
    ):
        self.dataset = dataset
        self.custom_data_path = custom_data_path
        self.image_size = image_size
        self.channels = channels
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.z_dim = z_dim
        self.epochs = epochs
        self.save_every = save_every
        self.learning_rate_g = learning_rate_g
        self.learning_rate_d = learning_rate_d
        self.beta1 = beta1
        self.beta2 = beta2
        self.use_spectral_norm = use_spectral_norm
        self.use_label_conditioning = use_label_conditioning
        self.n_classes = n_classes
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate

        # Derived settings
        self.checkpoint_dir = './training_checkpoints'
        self.output_dir = './generated_images'
        
        # Create necessary directories
        Path(self.checkpoint_dir).mkdir(exist_ok=True)
        Path(self.output_dir).mkdir(exist_ok=True)


# Self-attention module for better long-range dependencies
class SelfAttention(layers.Layer):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.query = layers.Conv2D(channels // 8, kernel_size=1, use_bias=False)
        self.key = layers.Conv2D(channels // 8, kernel_size=1, use_bias=False)
        self.value = layers.Conv2D(channels, kernel_size=1, use_bias=False)
        self.gamma = tf.Variable(0., trainable=True)
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        # It's okay to use the static channel dimension if it's defined
        channels = inputs.shape[-1]
        
        # Create query, key, value projections
        query = self.query(inputs)  # B x H x W x C//8
        key = self.key(inputs)      # B x H x W x C//8
        value = self.value(inputs)  # B x H x W x C
        
        # Use dynamic shapes for reshaping
        query_flat = tf.reshape(query, [batch_size, height * width, query.shape[-1]])  # B x HW x C//8
        key_flat = tf.reshape(key, [batch_size, height * width, key.shape[-1]])          # B x HW x C//8
        value_flat = tf.reshape(value, [batch_size, height * width, value.shape[-1]])      # B x HW x C
        
        # Calculate attention scores
        attention = tf.matmul(query_flat, key_flat, transpose_b=True)  # B x HW x HW
        attention = tf.nn.softmax(attention, axis=-1)
        
        # Apply attention to values
        out = tf.matmul(attention, value_flat)  # B x HW x C
        out = tf.reshape(out, [batch_size, height, width, self.channels])
        
        # Add skip connection with learnable parameter
        return inputs + self.gamma * out


# Improved data loading and preprocessing
class DataLoader:
    def __init__(self, config):
        self.config = config
        
    def load_dataset(self):
        if self.config.dataset == 'mnist':
            (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
            train_images = self._preprocess_images(train_images, expand_dims=True)
            
        elif self.config.dataset == 'fashion_mnist':
            (train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
            train_images = self._preprocess_images(train_images, expand_dims=True)
            
        elif self.config.dataset == 'cifar10':
            (train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()
            train_images = self._preprocess_images(train_images)
            
        elif self.config.dataset == 'custom' and self.config.custom_data_path:
            # Custom dataset loading logic
            train_images, train_labels = self._load_custom_dataset()
        else:
            raise ValueError("Invalid dataset configuration")
        
        # Return dataset and labels
        if self.config.use_label_conditioning:
            # One-hot encode labels for conditional GAN
            train_labels = tf.keras.utils.to_categorical(train_labels, self.config.n_classes)
            dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(train_images)
            
        return dataset.shuffle(self.config.buffer_size).batch(
            self.config.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    def _preprocess_images(self, images, expand_dims=False):
        # Resize images if needed
        if images.shape[1] != self.config.image_size or images.shape[2] != self.config.image_size:
            if expand_dims:
                images = np.expand_dims(images, axis=-1)
            
            # Resize images
            resized_images = []
            for img in images:
                img = tf.image.resize(img, [self.config.image_size, self.config.image_size])
                resized_images.append(img)
            images = np.array(resized_images)
        
        # Convert to appropriate channel count
        if expand_dims and self.config.channels == 3:
            # Expand grayscale to RGB
            images = np.repeat(images, 3, axis=-1)
        elif not expand_dims and images.shape[-1] == 1 and self.config.channels == 3:
            # Expand grayscale to RGB
            images = np.repeat(images, 3, axis=-1)
        
        # Normalize to [-1, 1]
        return (images.astype('float32') - 127.5) / 127.5
    
    def _load_custom_dataset(self):
        # Example custom dataset loading logic
        # You would implement your own logic here based on your data format
        data_path = Path(self.config.custom_data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Custom dataset path not found: {data_path}")
        
        # Example: loading images from a directory
        images = []
        labels = []
        
        image_paths = list(data_path.glob('*.jpg')) + list(data_path.glob('*.png'))
        
        for img_path in image_paths:
            # Load and resize image
            img = tf.keras.preprocessing.image.load_img(
                img_path, target_size=(self.config.image_size, self.config.image_size)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            
            # Example: extract label from filename or directory structure
            # This would depend on your specific dataset organization
            label = 0  # Default label
            labels.append(label)
        
        return np.array(images), np.array(labels)


# Improved Generator with residual blocks and upsampling
def make_generator_model(config):
    noise_input = layers.Input(shape=(config.z_dim,))
    
    # Optional conditional input
    if config.use_label_conditioning:
        label_input = layers.Input(shape=(config.n_classes,))
        # Concatenate noise and label
        combined_input = layers.Concatenate()([noise_input, label_input])
        x = combined_input
    else:
        x = noise_input
    
    # Initial dense layer and reshape
    # Calculate initial size based on final desired output size
    initial_size = config.image_size // 16  # For 64x64 output, start with 4x4
    
    # Initial projection and reshape
    x = layers.Dense(initial_size * initial_size * 512, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((initial_size, initial_size, 512))(x)
    
    # Residual upsampling blocks
    x = generator_upsample_block(x, 512, config)  # -> 8x8
    x = generator_upsample_block(x, 256, config)  # -> 16x16
    x = generator_upsample_block(x, 128, config)  # -> 32x32
    
    # Self-attention layer at 32x32 resolution
    if config.use_attention:
        x = SelfAttention(128)(x)
    
    x = generator_upsample_block(x, 64, config)  # -> 64x64
    
    # Final convolution to get the right number of channels
    x = layers.Conv2D(config.channels, kernel_size=3, padding='same', use_bias=False)(x)
    outputs = layers.Activation('tanh')(x)
    
    # Define model based on conditional or unconditional
    if config.use_label_conditioning:
        model = models.Model([noise_input, label_input], outputs)
    else:
        model = models.Model(noise_input, outputs)
        
    return model


def generator_upsample_block(x, filters, config):
    skip = x
    
    # Upsample
    x = layers.UpSampling2D(size=(2, 2))(x)
    
    # Two conv blocks with residual connection
    x = layers.Conv2D(filters, kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(filters, kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    
    # Process skip connection
    skip = layers.UpSampling2D(size=(2, 2))(skip)
    skip = layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(skip)
    
    # Add skip connection
    x = layers.Add()([x, skip])
    x = layers.LeakyReLU(0.2)(x)
    
    return x


# Improved Discriminator with spectral normalization and residual blocks
def make_discriminator_model(config):
    # Input layer
    img_input = layers.Input(shape=(config.image_size, config.image_size, config.channels))
    
    # Optional label input for conditional GAN
    if config.use_label_conditioning:
        label_input = layers.Input(shape=(config.n_classes,))
        # Process label (e.g., embed it to each pixel)
        label_embedding = layers.Dense(config.image_size * config.image_size)(label_input)
        label_embedding = layers.Reshape((config.image_size, config.image_size, 1))(label_embedding)
        # Concatenate with image
        x = layers.Concatenate()([img_input, label_embedding])
    else:
        x = img_input
    
    # Initial convolution
    if config.use_spectral_norm:
        x = SpectralNormalization(layers.Conv2D(64, kernel_size=4, strides=2, padding='same'))(x)
    else:
        x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(config.dropout_rate)(x)
    
    # Discriminator blocks with increasing filters
    x = discriminator_block(x, 128, config)  # 32x32 -> 16x16
    
    # Add self-attention at 16x16 resolution
    if config.use_attention:
        x = SelfAttention(128)(x)
        
    x = discriminator_block(x, 256, config)  # 16x16 -> 8x8
    x = discriminator_block(x, 512, config)  # 8x8 -> 4x4
    
    # Flatten and final dense layer
    x = layers.Flatten()(x)
    
    if config.use_spectral_norm:
        outputs = SpectralNormalization(layers.Dense(1))(x)
    else:
        outputs = layers.Dense(1)(x)
    
    # Create the model
    if config.use_label_conditioning:
        model = models.Model([img_input, label_input], outputs)
    else:
        model = models.Model(img_input, outputs)
        
    return model


def discriminator_block(x, filters, config):
    skip = x
    
    # First convolution
    if config.use_spectral_norm:
        x = SpectralNormalization(
            layers.Conv2D(filters, kernel_size=3, padding='same')
        )(x)
    else:
        x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(config.dropout_rate)(x)
    
    # Second convolution with stride for downsampling
    if config.use_spectral_norm:
        x = SpectralNormalization(
            layers.Conv2D(filters, kernel_size=4, strides=2, padding='same')
        )(x)
    else:
        x = layers.Conv2D(filters, kernel_size=4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(config.dropout_rate)(x)
    
    # Process skip connection
    if config.use_spectral_norm:
        skip = SpectralNormalization(
            layers.Conv2D(filters, kernel_size=1, strides=2, padding='same')
        )(skip)
    else:
        skip = layers.Conv2D(filters, kernel_size=1, strides=2, padding='same')(skip)
    
    # Add skip connection
    x = layers.Add()([x, skip])
    
    return x


# Spectral Normalization for GAN stability
class SpectralNormalization(layers.Wrapper):
    def __init__(self, layer, iteration=1, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        self.iteration = iteration

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
        
        # Initialize u vector (for power iteration)
        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]), 
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name='sn_u'
        )
        super(SpectralNormalization, self).build()

    def call(self, inputs):
        self.update_weights()
        output = self.layer(inputs)
        return output
    
    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        
        # Power iteration for approximating spectral norm
        for _ in range(self.iteration):
            v = tf.matmul(self.u, tf.transpose(w_reshaped))
            v = v / tf.maximum(tf.norm(v), 1e-12)
            
            self.u.assign(tf.matmul(v, w_reshaped))
            self.u.assign(self.u / tf.maximum(tf.norm(self.u), 1e-12))
        
        # Calculate spectral norm
        sigma = tf.matmul(tf.matmul(self.u, tf.transpose(w_reshaped)), tf.transpose(v))
        
        # Normalize weight
        self.layer.kernel.assign(self.w / sigma)


# Improved loss functions
class GANLoss:
    def __init__(self, loss_type='hinge'):
        self.loss_type = loss_type
    
    def discriminator_loss(self, real_output, fake_output):
        if self.loss_type == 'hinge':
            # Hinge loss for discriminator
            real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_output))
            fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_output))
            return real_loss + fake_loss
        
        elif self.loss_type == 'wasserstein':
            # Wasserstein loss for discriminator
            return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
        
        else:  # Default to standard cross-entropy
            cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            real_loss = cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
            return real_loss + fake_loss
    
    def generator_loss(self, fake_output):
        if self.loss_type == 'hinge':
            # Hinge loss for generator
            return -tf.reduce_mean(fake_output)
        
        elif self.loss_type == 'wasserstein':
            # Wasserstein loss for generator
            return -tf.reduce_mean(fake_output)
        
        else:  # Default to standard cross-entropy
            cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            return cross_entropy(tf.ones_like(fake_output), fake_output)


# GAN Trainer class
class GANTrainer:
    def __init__(self, config):
        self.config = config
        self.dataloader = DataLoader(config)
        self.dataset = self.dataloader.load_dataset()
        
        # Initialize models
        self.generator = make_generator_model(config)
        self.discriminator = make_discriminator_model(config)
        
        # Initialize loss function
        self.loss_fn = GANLoss(loss_type='hinge')  # You can change to 'standard' or 'wasserstein'
        
        # Initialize optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.learning_rate_g,
            beta_1=config.beta1,
            beta_2=config.beta2
        )
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.learning_rate_d,
            beta_1=config.beta1,
            beta_2=config.beta2
        )
        
        # Create checkpoints
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, 
            self.config.checkpoint_dir, 
            max_to_keep=5
        )
        
        # Create fixed noise for evaluation
        self.seed = tf.random.normal([16, config.z_dim])
        if config.use_label_conditioning:
            # Create fixed labels for evaluation
            self.seed_labels = tf.keras.utils.to_categorical(
                np.repeat(np.arange(4), 4), config.n_classes
            )
    
    def train(self):
        # Restore from checkpoint if available
        self.restore_checkpoint()
        
        # Start training loop
        for epoch in range(self.config.epochs):
            start_time = time.time()
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            
            # Train for one epoch
            self.train_epoch(epoch)
            
            # Generate and save images
            self.generate_and_save_images(epoch + 1)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.checkpoint_manager.save()
                print(f"Checkpoint saved for epoch {epoch+1}")
            
            print(f"Time for epoch {epoch+1}: {time.time() - start_time:.2f} sec")
    
    @tf.function
    def train_step(self, images, labels=None):
        # Generate random noise
        batch_size = tf.shape(images)[0]
        noise = tf.random.normal([batch_size, self.config.z_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake images
            if self.config.use_label_conditioning:
                generated_images = self.generator([noise, labels], training=True)
                real_output = self.discriminator([images, labels], training=True)
                fake_output = self.discriminator([generated_images, labels], training=True)
            else:
                generated_images = self.generator(noise, training=True)
                real_output = self.discriminator(images, training=True)
                fake_output = self.discriminator(generated_images, training=True)
            
            # Calculate losses
            gen_loss = self.loss_fn.generator_loss(fake_output)
            disc_loss = self.loss_fn.discriminator_loss(real_output, fake_output)
        
        # Calculate gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        # Apply gradients
        self.generator_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )
        
        return gen_loss, disc_loss
    
    def train_epoch(self, epoch):
        # Create progress bar
        progress_bar = tqdm(total=tf.data.experimental.cardinality(self.dataset).numpy(),
                          desc=f"Epoch {epoch+1}", unit="batch")
        
        # Initialize metrics
        gen_losses = []
        disc_losses = []
        
        # Train on batches
        for batch in self.dataset:
            if self.config.use_label_conditioning:
                images, labels = batch
                gen_loss, disc_loss = self.train_step(images, labels)
            else:
                images = batch
                gen_loss, disc_loss = self.train_step(images)
            
            # Track losses
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                'g_loss': float(gen_loss),
                'd_loss': float(disc_loss)
            })
        
        # Close progress bar
        progress_bar.close()
        
        # Print epoch summary
        print(f"Generator loss: {np.mean(gen_losses):.4f}, Discriminator loss: {np.mean(disc_losses):.4f}")
    
    def generate_and_save_images(self, epoch):
        # Generate images from seed noise
        if self.config.use_label_conditioning:
            predictions = self.generator([self.seed, self.seed_labels], training=False)
        else:
            predictions = self.generator(self.seed, training=False)
        
        # Create figure
        fig = plt.figure(figsize=(4, 4))
        
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            if self.config.channels == 1:
                plt.imshow(predictions[i, :, :, 0] * 0.5 + 0.5, cmap='gray')
            else:
                plt.imshow(predictions[i, :, :, :] * 0.5 + 0.5)  # Rescale from [-1,1] to [0,1]
            plt.axis('off')
        
        # Save figure
        output_path = os.path.join(self.config.output_dir, f'image_at_epoch_{epoch:04d}.png')
        plt.savefig(output_path)
        plt.close(fig)
        
        # Optionally save individual images
        if epoch % 10 == 0 or epoch == self.config.epochs:
            for i, prediction in enumerate(predictions):
                img = (prediction * 0.5 + 0.5) * 255  # Scale from [-1,1] to [0,255]
                img = tf.cast(img, tf.uint8)
                
                # Save as PNG
                img_path = os.path.join(self.config.output_dir, f'sample_{epoch:04d}_{i:02d}.png')
                if self.config.channels == 1:
                    img = tf.reshape(img, [self.config.image_size, self.config.image_size])
                    PIL.Image.fromarray(img.numpy()).save(img_path)
                else:
                    PIL.Image.fromarray(img.numpy()).save(img_path)
    
    def restore_checkpoint(self):
        latest_checkpoint = self.checkpoint_manager.latest_checkpoint
        if latest_checkpoint:
            self.checkpoint.restore(latest_checkpoint)
            print(f"Restored from checkpoint: {latest_checkpoint}")
        else:
            print("Initializing from scratch")
    
    def generate_images(self, num_images=16, random_seeds=True):
        """Generate images after training"""
        if random_seeds:
            noise = tf.random.normal([num_images, self.config.z_dim])
            if self.config.use_label_conditioning:
                # Generate varied classes
                labels = tf.keras.utils.to_categorical(
                    np.random.randint(0, self.config.n_classes, num_images), 
                    self.config.n_classes
                )
        else:
            noise = self.seed[:num_images]
            if self.config.use_label_conditioning:
                labels = self.seed_labels[:num_images]
        
        # Generate images
        if self.config.use_label_conditioning:
            generated_images = self.generator([noise, labels], training=False)
        else:
            generated_images = self.generator(noise, training=False)
        
        # Convert to numpy for easier handling
        return (generated_images * 0.5 + 0.5).numpy()


# Main execution function
def main():
    # Create configuration
    config = GANConfig(
        dataset='fashion_mnist',  # Try different datasets
        image_size=64,
        channels=3,
        batch_size=64,
        z_dim=128,
        epochs=100,
        use_spectral_norm=True,
        use_attention=True,
        use_label_conditioning=True
    )
    
    # Create and start trainer
    trainer = GANTrainer(config)
    trainer.train()
    
    # Generate a grid of sample images after training
    samples = trainer.generate_images(num_images=64, random_seeds=True)
    
    # Create a large grid figure
    rows, cols = 8, 8
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    
    for i, sample in enumerate(samples):
        ax = axes[i // cols, i % cols]
        if config.channels == 1:
            ax.imshow(sample[:, :, 0], cmap='gray')
        else:
            ax.imshow(sample)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, 'final_samples_grid.png'))
    plt.close(fig)
    
    print(f"Training completed. Check the generated images at {config.output_dir}")


if __name__ == "__main__":
    main()
