import os
from tensorflow.keras import models, layers, activations, \
    optimizers, utils, losses, initializers, metrics, callbacks

epochs = 100
batch_size = 8
patience = 10
learning_rate = 0.001
destiny_path = "ds"
model_path = f"checkpoints/{destiny_path}.keras"
exists = os.path.exists(model_path)

model = models.load_model(model_path) \
    if exists \
    else models.Sequential([
        layers.Resizing(32, 32),
        layers.Conv2D(16, (3, 3),
            activation = 'relu',
            kernel_initializer = initializers.RandomNormal()
        ),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(6,
            activation = 'sigmoid',
            kernel_initializer = initializers.RandomNormal()
        )
    ])

if exists:
    model.summary()
else: 
    model.compile(
        optimizer = optimizers.Adam(
            learning_rate = learning_rate
        ),
        loss = losses.SparseCategoricalCrossentropy(),
        metrics = ["accuracy"]
    )

train = utils.image_dataset_from_directory(
    "ds",
    validation_split= 0.2,
    subset= "training",
    seed= 123,
    shuffle= True,
    image_size= (128, 128),
    batch_size= batch_size
)

test = utils.image_dataset_from_directory(
    destiny_path,
    validation_split= 0.2,
    subset= "validation",
    seed= 123,
    shuffle= True,
    image_size= (128, 128),
    batch_size= batch_size
)

model.fit(train,
    epochs = epochs,
    validation_data = test,
    callbacks= [
        callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = patience,
            verbose = 1
        ),
        callbacks.ModelCheckpoint(
            filepath = model_path,
            save_weights_only = False,
            monitor = 'loss',
            mode = 'min',
            save_best_only = True
        )
    ]
)