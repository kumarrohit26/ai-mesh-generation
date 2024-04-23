import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout, BatchNormalization
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from sklearn.metrics import confusion_matrix
import seaborn as sns
import cv2

def remove_white_images_without_neigbour(row):
    if (row['neighbours'] == 0) & (int(row['len_of_boundry_per_100px']) == 0):
        return False
    else:
        return True
    
def remove_white_images(row):
    if (int(row['len_of_boundry_per_100px']) == 0):
        return False
    else:
        return True
    
def clean_data(remove_white_image_with_neigbor=False, data=None ):
    if data is None:
        data = pd.read_csv('./output/training_data/train.csv')
    
    # With White Image','Remove white image with No neighbour','Remove all white image'
    if remove_white_image_with_neigbor == 'Remove white image with No neighbour':
        data = data[data.apply(remove_white_images_without_neigbour, axis=1)]
    elif remove_white_image_with_neigbor == 'Remove all white image':
        data = data[data.apply(remove_white_images, axis=1)]

    return data
    # if remove_white_image_with_neigbor:
    #     data = data[data.apply(remove_white_images, axis=1)]
    # else:
    #     data = data[data.apply(remove_white_images_without_neigbour, axis=1)]
    # return data

def only_data_with_component_name(row, component_name):
    if component_name in row['name'] and '250' in row['name']:
        return True
    else:
        return False


def get_data_for_component(component_name='microstrip_coupler', data=None):
    if data is None:
        data = pd.read_csv('./output/training_data/train.csv')
    
    data = data[data.apply(lambda row: only_data_with_component_name(row, component_name), axis=1)]
    return data

def data_without_rotation(row):
    if '_0_250_' in row['name']:
        return True
    else:
        return False

def get_data_without_rotation(data=None):
    if data is None:
        data = pd.read_csv('./output/training_data/train.csv')
    
    data = data[data.apply(data_without_rotation, axis=1)]
    return data


def data_250_data(row):
    if '_250_' in row['name']:
        return True
    else:
        return False

def get_data_with_250(data=None):
    if data is None:
        data = pd.read_csv('./output/training_data/train.csv')
    
    data = data[data.apply(data_250_data, axis=1)]
    return data

def load_images(data):
    image_data = []
    for image_file in data['name']:
        image = load_img('./output/tiles/' + image_file, color_mode='grayscale', target_size=(224, 224))
        image = img_to_array(image)
        #image = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=3)
        image /= 255.0  # Normalize pixel values
        image_data.append(image)

    image_data = np.array(image_data)
    return image_data

def create_model(noc, layers, nodes):
    # Define the input layers
    image_input = Input(shape=(224, 224, 1))
    feature_input = Input(shape=(noc,))

    # CNN for processing images
    # conv1 = Conv2D(cells, (3, 3), activation='relu')(image_input)
    # maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # conv2 = Conv2D(cells, (3, 3), activation='relu')(maxpool1)
    # maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # flatten = Flatten()(maxpool2)

    # Dynamic creation of convolutional layers
    x = image_input
    for _ in range(layers):
        x = Conv2D(nodes, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten the output of the last convolutional layer
    flatten = Flatten()(x)


    # Dense network for processing features
    dense1 = Dense(nodes, activation='relu')(feature_input)

    # Combine the two networks
    merged = Concatenate()([flatten, dense1])

    # Fully connected layers
    fc1 = Dense(2*nodes, activation='relu')(merged)
    output = Dense(1, activation='linear')(fc1)

    # Create the model
    model = keras.Model(inputs=[image_input, feature_input], outputs=output)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    return model

def create_categorical_model(noc, layers, cells, optimizer_name, learning_rate, batch_norm = False, drop_out_percent = 0):
    # Define the input layers
    image_input = Input(shape=(224, 224, 1))
    feature_input = Input(shape=(noc,))

    # CNN for processing images
    # conv1 = Conv2D(32, (3, 3), activation='relu')(image_input)
    # maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # conv2 = Conv2D(32, (3, 3), activation='relu')(maxpool1)
    # maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # flatten = Flatten()(maxpool1)

    x = image_input
    for _ in range(layers):
        x = Conv2D(cells, (3, 3), activation='relu')(x)
        if batch_norm is True:
            x = BatchNormalization()(x)
        if drop_out_percent > 0:
            x = Dropout(drop_out_percent)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        #cells = 2*cells

    flatten = Flatten()(x)

    # Dense network for processing features
    dense1 = Dense(cells, activation='relu')(feature_input)

    # Combine the two networks
    merged = Concatenate()([flatten, dense1])

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(merged)
    output = Dense(3, activation='softmax')(fc1)

    # Create the model
    optimizer = get_optimizer(optimizer_name, learning_rate)
    model = keras.Model(inputs=[image_input, feature_input], outputs=output)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def single_categorical_model(noc, layers, cells, optimizer_name, learning_rate, batch_norm = False, drop_out_percent = 0):
    # Define the input layers
    image_input = Input(shape=(224, 224, 1))
    feature_input = Input(shape=(noc,))

    x = image_input
    for _ in range(layers):
        x = Conv2D(cells, (3, 3), activation='relu')(x)
        if batch_norm is True:
            x = BatchNormalization()(x)
        if drop_out_percent > 0:
            x = Dropout(drop_out_percent)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    flatten = Flatten()(x)

    # Dense network for processing features
    #dense1 = Dense(cells, activation='relu')(feature_input)

    # Combine the two networks
    #merged = Concatenate()([flatten, dense1])

    # Fully connected layers
    fc1 = Dense(64, activation='relu')(flatten)
    output = Dense(3, activation='softmax')(fc1)

    # Create the model
    optimizer = get_optimizer(optimizer_name, learning_rate)
    model = keras.Model(inputs=image_input, outputs=output)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def showDetails(history):
    # Access training history
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    training_accuracy = history.history['mean_absolute_error']
    validation_accuracy = history.history['val_mean_absolute_error']

    # Plot training and validation loss
    fig = plt.figure(figsize=(12, 5))
    plt.subplot(2, 1, 1)
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Plot training and validation accuracy
    plt.subplot(2, 1, 2)
    plt.plot(training_accuracy, label='Training MAE')
    plt.plot(validation_accuracy, label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.title('Training and Validation Mean Absolute Error')

    plt.tight_layout()
    #plt.show()
    return fig

def showDetails_regression(history):
    fig1 = plt.figure(figsize=(8, 6))
    #plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylim([0,1.2])
    plt.ylabel('Accuracy')
    plt.legend()

    fig2 = plt.figure(figsize=(8, 6))
    #plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    #plt.show()
    return fig1, fig2

def confusion_mat(model, X_test, y_test):
    # Predict the labels on the test set
    y_pred = model.predict(X_test)

    # Convert the predictions to class indices
    y_pred_indices = y_pred.argmax(axis=1)
    y_test_indices = y_test.argmax(axis=1)

    # Create a confusion matrix
    cm = confusion_matrix(y_test_indices, y_pred_indices)
    labels = ['Low', 'Mid', 'High']
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    #plt.show()

    return plt
    

def run_linear(remove_white_image_with_neigbor, columns, layers, num_of_cell, epochs):
    data = clean_data(remove_white_image_with_neigbor)
    image_data = load_images(data)

    # Load numerical features and output
    features = data[columns].values
    #features = data[['len_of_boundry_inv', 'disjoint_image']].values
    labels = data['num_triangles'].values
    labels = labels.astype(float)
    robust_scaler = RobustScaler()
    features = robust_scaler.fit_transform(features)

    noc = features.shape[1]

    random_state = 0

    # Split the data into training, validation, and test sets for images
    X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)

    # Split the data into training, validation, and test sets for features
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=random_state)
    features_train, features_val, labels_train, labels_val = train_test_split(features_train, labels_train, test_size=0.2, random_state=random_state)

    model = create_model(noc, layers, num_of_cell)
    print('Running model')
    history = model.fit([X_train, features_train], y_train, validation_data=([X_val, features_val], y_val), epochs=epochs, batch_size=32)
    fig = showDetails(history)

    # Evaluate the model on the test data
    test_loss, test_mae = model.evaluate([X_test, features_test], y_test)
    # Print the test loss and MAE

    return test_loss, test_mae, fig

def run_regression(remove_white_image_with_neigbor,only_one_component, y ,columns, layers, num_of_cell, epochs, batch_norm = False, drop_out_percent = 0,
                   learning_rate=0.001, batch_size=16, optimizer='adam', data=None, single_model=False):
    #data = clean_data(remove_white_image_with_neigbor)
    #if only_one_component:
    #    data = get_data_for_component('microstrip_coupler', data)
    image_data = load_images(data)

    # Load numerical features and output
    features = data[columns].values
    #features = data[['len_of_boundry_inv', 'disjoint_image']].values

    labels = data[y].values

    # One-hot encode labels
    labels = to_categorical(labels)

    robust_scaler = RobustScaler()
    features = robust_scaler.fit_transform(features)

    noc = features.shape[1]

    random_state = 0

    # Split the data into training, validation, and test sets for images
    X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)

    # Split the data into training, validation, and test sets for features
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=random_state)
    features_train, features_val, labels_train, labels_val = train_test_split(features_train, labels_train, test_size=0.2, random_state=random_state)

    if single_model:
        print('Single model training...')
        model = single_categorical_model(noc, layers, num_of_cell, optimizer, learning_rate, batch_norm, drop_out_percent)
        print('Running model')
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

        fig = showDetails_regression(history)
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        cm = confusion_mat(model, X_test, y_test)
    
    else:
        print('Multi model training...')
        model = create_categorical_model(noc, layers, num_of_cell, optimizer, learning_rate, batch_norm, drop_out_percent)
        print('Running model')
        history = model.fit([X_train, features_train], y_train, validation_data=([X_val, features_val], y_val), epochs=epochs, batch_size=batch_size)

        fig = showDetails_regression(history)
        test_loss, test_accuracy = model.evaluate([X_test, features_test], y_test)
        cm = confusion_mat(model, [X_test, features_test], y_test)

    return test_loss, test_accuracy, fig, cm

def get_optimizer(name, learning_rate):
    if name == 'adam':
        return Adam(learning_rate=learning_rate)
    elif name == 'sgd':
        return SGD(learning_rate=learning_rate)
    elif name == 'rmsprop':
        return RMSprop(learning_rate=learning_rate)
    elif name == 'adagrad':
        return Adagrad(learning_rate=learning_rate)
    elif name == 'adadelta':
        return Adadelta(learning_rate=learning_rate)
    elif name == 'adamax':
        return Adamax(learning_rate=learning_rate)
    elif name == 'nadam':
        return Nadam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {name}")




