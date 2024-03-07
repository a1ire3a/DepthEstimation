import tensorflow as tf
import os

class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, directory_X, directory_y, batch_size=8):
        self.X_paths = [os.path.join(directory_X, file) for file in os.listdir(directory_X)]
        self.y_paths = [os.path.join(directory_y, 'depth_minmaxnorm' + file.split('_img')[-1]) for file in os.listdir(directory_X)]
        self.batch_size = batch_size

    def __len__(self):
        return len(self.X_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_X_paths = self.X_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y_paths = self.y_paths[idx * self.batch_size: (idx + 1) * self.batch_size]

        X_batch = []
        y_batch = []

        for X_path, y_path in zip(batch_X_paths, batch_y_paths):
            X = tf.io.read_file(X_path)
            X = tf.image.decode_jpeg(X, channels=3)
            y = tf.io.read_file(y_path)
            y = tf.image.decode_jpeg(y, channels=3)

            # Preprocess if necessary
            # X = preprocess_function(X)
            # y = preprocess_function(y)

            X_batch.append(X)
            y_batch.append(y)

        return tf.convert_to_tensor(X_batch), tf.convert_to_tensor(y_batch)
