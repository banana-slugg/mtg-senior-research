import os
from keras import layers, models
import csv
from kernels_gray import kernel_set

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf


def cnn(test_group: int):
    batch_size = 30
    img_height = 457
    img_width = 626
    for i in range(10):
        train_data_dir = (
            f"~/Code/python/tensorflow-test/final-datasets/group-{test_group}/training"
        )
        valid_data_dir = f"~/Code/python/tensorflow-test/final-datasets/group-{test_group}/validation"

        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_data_dir,
            seed=i,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            color_mode="grayscale",
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            valid_data_dir,
            seed=i,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            color_mode="grayscale",
        )
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        num_classes = 5

        kernel_initializer = tf.keras.initializers.constant(kernel_set)

        model = models.Sequential()
        model.add(layers.RandomFlip("horizontal_and_vertical"))
        model.add(layers.RandomRotation(0.1))
        model.add(layers.RandomZoom(0.1))
        model.add(layers.Rescaling(1.0 / 255))
        model.add(
            layers.Conv2D(
                14,
                3,
                activation="relu",
                bias_initializer="zeros",
                kernel_initializer=kernel_initializer,
                use_bias=False,
                padding="same",
            )
        )
        model.add(layers.MaxPooling2D(2))
        model.add(layers.Conv2D(8, 5, activation="relu", padding="same"))
        model.add(layers.MaxPooling2D(2))
        model.add(layers.Conv2D(8, 5, activation="relu", padding="same"))
        model.add(layers.MaxPooling2D(2))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(num_classes))
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        epochs = 50
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
        acc = history.history["accuracy"][-1]
        val_acc = history.history["val_accuracy"][-1]
        loss = history.history["loss"][-1]
        val_loss = history.history["val_loss"][-1]
        with open("./res/test_gray.csv", "a") as file:
            csv.writer(file).writerow(
                [
                    test_group,
                    i,
                    round(loss, 4),
                    round(acc, 4),
                    round(val_loss, 4),
                    round(val_acc, 4),
                ]
            )


def main():
    with open("./res/test_gray.csv", "w+") as file:
        csv.writer(file).writerow(
            [
                "Test Group",
                "Run Number",
                "Training Loss",
                "Training Accuracy",
                "Validation Loss",
                "Validation Accuracy",
            ]
        )
    for i in range(10):
        cnn(i)


if __name__ == "__main__":
    main()
