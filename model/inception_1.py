from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf

def Model_train(train_generator, val_generator, epochs, batch_size, batch_size_val, path, checkpoint_filepath):
    model = InceptionV3(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(4, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=x)
    opt = Adam(learning_rate=0.0003)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    try:
        model.load_weights(checkpoint_filepath)
        print ('Load checkpoint success!')
    except:
        pass
    history = model.fit_generator(train_generator, steps_per_epoch=train_generator.n // batch_size, epochs=epochs,
                                  validation_data=val_generator, validation_steps=val_generator.n // batch_size_val,
                                  verbose=1, callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                                          patience=5, min_lr=0.000001),
                                                        ModelCheckpoint(filepath=checkpoint_filepath, verbose = 1,
                                                                        save_weights_only=True, monitor='val_accuracy',
                                                                        mode='max', save_best_only=True)],
                                  use_multiprocessing=False, shuffle=True)
    model.save(path)
    print('Model saved')
    return history
