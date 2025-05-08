import os
import tensorflow as tf

def build_lip_dataset(data_dir, words, seq_length=16, batch_size=8):
    all_paths = []
    all_labels = []
    for idx, word in enumerate(words):
        folder = os.path.join(data_dir, word)
        for fname in sorted(os.listdir(folder)):
            all_paths.append(os.path.join(folder, fname))
            all_labels.append(idx)

    paths_ds = tf.data.Dataset.from_tensor_slices(all_paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(all_labels)

    def _load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    imgs_ds = paths_ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = tf.data.Dataset.zip((imgs_ds, labels_ds)) \
        .batch(seq_length, drop_remainder=True) \
        .map(lambda imgs, labs: (imgs, labs[0])) \
        .shuffle(1000) \
        .batch(batch_size)

    return dataset.prefetch(tf.data.AUTOTUNE)
