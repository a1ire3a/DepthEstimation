
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import UpSampling2D
import tensorflow as tf
# from keras.models import load_model
from keras.layers import Layer, UpSampling2D
# from keras.models import load_model
# from keras.layers import Layer, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, UpSampling2D
from utils import depth_loss_function, load_dataset
from costum_dataset import CustomDataset
from tqdm import tqdm
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

# Define your learning rate scheduler function
def lr_scheduler(epoch, lr):
    if epoch < 3:
        return lr  # Use initial learning rate for first 10 epochs
    else:
        return lr * 0.90  # Decrease learning rate by 5% every epoch after the 10th epoch

initial_learning_rate = 0.0005

class BilinearUpSampling2D(Layer):
    def __init__(self, size=(1, 1), data_format=None, **kwargs):
        self.size = tuple(size)
        self.data_format = data_format
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BilinearUpSampling2D, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.image.resize(inputs, size=self.size, method=tf.image.ResizeMethod.BILINEAR)

    def compute_output_shape(self, input_shape):
        height, width = self.size
        return (input_shape[0], input_shape[1] * height, input_shape[2] * width, input_shape[3])

    def get_config(self):
        config = super(BilinearUpSampling2D, self).get_config()
        config.update({'size': self.size, 'data_format': self.data_format})
        return config
custom_objects = {'BilinearUpSampling2D': tf.keras.layers.UpSampling2D, 'depth_loss_function': None}
model = load_model('fine_tuned_model02.h5',custom_objects=custom_objects, compile=False)


for i, layer in enumerate(model.layers):
    if isinstance(layer, BilinearUpSampling2D):
        model.layers[i] = UpSampling2D(size=layer.size, data_format=layer.data_format)


num_channels = 256

all_layers_except_last = model.layers[:-1]

new_model = Model(inputs=model.input, outputs=model.layers[-2].output)

for layer in new_model.layers:
    layer.trainable = False

new_output_layer = Conv2D(3, (2, 2), padding='same', activation='leaky_relu', name='conv3',strides=(2, 2))(new_model.output)

fine_tuned_model = Model(inputs=new_model.input, outputs=new_output_layer)

fine_tuned_model.compile(optimizer=Adam(learning_rate=initial_learning_rate), loss=depth_loss_function)


input_data_directory = '/home/alireza/datasets/CARLA/train/raw/'
target_data_directory = '/home/alireza/datasets/CARLA/train/depth/'
train_dataset = CustomDataset(input_data_directory, target_data_directory)

fine_tuned_model.compile(optimizer=Adam(learning_rate=initial_learning_rate), loss=depth_loss_function)
# Adjust loss as needed
lr_callback = LearningRateScheduler(lr_scheduler)
# fine_tuned_model.fit(X_train, Y_train, epochs=2, batch_size=2)
from tensorflow.keras.callbacks import ModelCheckpoint

# Define the checkpoint filepath
checkpoint_filepath = 'fine_tuned_model_best_checkpoint.h5'

# Define the ModelCheckpoint callback to save the model after 10 epochs
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='loss',  # Monitor validation loss
    mode='min',           # Save the model when validation loss decreases
    save_best_only=True,  # Only save the best model
    verbose=1
)

fine_tuned_model.fit(train_dataset, epochs=50, batch_size=8, callbacks=[lr_callback ,model_checkpoint_callback])
model.save('fine_tuned_model.h5')
