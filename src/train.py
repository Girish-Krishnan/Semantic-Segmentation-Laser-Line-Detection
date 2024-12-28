import tensorflow as tf

def train_model(model, X_train, y_train, X_val, y_val, batch_size, epochs, save_path):
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    model.save(save_path)
