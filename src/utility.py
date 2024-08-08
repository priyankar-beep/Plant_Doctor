
# Standard Library Imports
import glob
import os
import random
import sys
import warnings

# Third-Party Library Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from transformers import TFBertModel, BertTokenizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from vit_keras import vit


random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

warnings.filterwarnings('ignore')
print('TensorFlow Version ' + tf.__version__)

class MultiTaskDataGenerator(Sequence):
    def __init__(self, gen_plant, gen_disease):
        self.gen_plant = gen_plant
        self.gen_disease = gen_disease

    def __len__(self):
        return min(len(self.gen_plant), len(self.gen_disease))

    def __getitem__(self, index):
        plant_batch = self.gen_plant[index]
        disease_batch = self.gen_disease[index]
        images, plant_labels = plant_batch
        _, disease_labels = disease_batch
        return images, {'plant_output': plant_labels, 'disease_output': disease_labels}

# Create TensorFlow dataset for text
def create_text_dataset(input_ids, attention_masks, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_masks))
    dataset = dataset.batch(batch_size)
    return dataset

def split_dataframe(df, test_size=0.2, random_state=42):
    # Split the DataFrame into training and testing sets
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    return df_train, df_test

def tokenize_text(texts, tokenizer, max_seq_length):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_seq_length, return_tensors='tf')
    return encodings['input_ids'], encodings['attention_mask']


def load_disease_captions(captions_path):
    disease_captions = {}
    with open(captions_path, 'r') as file:
        for line in file:
            if line.strip():
                parts = line.strip().split(': ')
                if len(parts) == 2:
                    disease_name = parts[0].strip()
                    if disease_name.startswith('_'):
                        disease_name = disease_name[1:]  # Remove leading underscore
                    description = parts[1].strip()
                    disease_captions[disease_name] = description
    return disease_captions

def prepare_labels(df):
    # Ensure labels are correctly aligned with text data
    return {
        'plant_output': np.array(df['plant_name']),
        'disease_output': np.array(df['disease_name'])
    }

def create_combined_dataframe(image_category_mapping, image_plant_mapping, captions):
    # Map image paths to their disease names and captions
    data = {
        'filename': list(image_category_mapping.keys()),
        'plant_name': [image_plant_mapping[img_path] for img_path in image_category_mapping.keys()],
        'disease_name': [image_category_mapping[img_path] for img_path in image_category_mapping.keys()],
    }
    df = pd.DataFrame(data)
    df['caption'] = df['disease_name'].map(captions)
    return df

def map_images_to_category_and_plant(images_path):
    image_category_mapping = {}
    image_plant_mapping = {}

    # Walk through the directory and map images to categories and plants
    for subdir, _, files in os.walk(images_path):
        directory_name = os.path.basename(subdir)
        if directory_name == 'data':
            continue
        print('---->', directory_name)
        
        # Extract plant and disease names
        plant_name = directory_name.split("__")[0]
        disease_name = directory_name.split("__")[-1]
        
        if not directory_name:
            continue
        
        # Remove leading underscores from disease names if present
        if disease_name[0] == '_':
            disease_name = disease_name[1:]
        
        # Map images to their categories and plants
        for file in files:
            image_path = os.path.join(subdir, file)
            image_category_mapping[image_path] = disease_name
            image_plant_mapping[image_path] = plant_name

    return image_category_mapping, image_plant_mapping

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_train_datagen(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                         shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True,
                         brightness_range=[0.5, 1.5], fill_mode='nearest', validation_split=0.2):
    return ImageDataGenerator(
        rescale=rescale,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        brightness_range=brightness_range,
        fill_mode=fill_mode,
        validation_split=validation_split
    )

def create_test_datagen(rescale=1./255):
    return ImageDataGenerator(rescale=rescale)

def create_data_generator(datagen, dataframe, x_col, y_col, subset, batch_size, seed, color_mode, shuffle, class_mode, target_size):
    return datagen.flow_from_dataframe(
        dataframe=dataframe,
        x_col=x_col,
        y_col=y_col,
        subset=subset,
        batch_size=batch_size,
        seed=seed,
        color_mode=color_mode,
        shuffle=shuffle,
        class_mode=class_mode,
        target_size=target_size
    )

def create_train_valid_generators(dataframe, x_col, y_col_plant, y_col_disease, batch_size, image_size, validation_split=0.2):

    train_datagen = create_train_datagen(validation_split=validation_split)
    
    train_gen_plant = create_data_generator(
        datagen=train_datagen,
        dataframe=dataframe,
        x_col=x_col,
        y_col=y_col_plant,
        subset='training',
        batch_size=batch_size,
        seed=1,
        color_mode='rgb',
        shuffle=True,
        class_mode='categorical',
        target_size=(image_size, image_size)
    )
    
    train_gen_disease = create_data_generator(
        datagen=train_datagen,
        dataframe=dataframe,
        x_col=x_col,
        y_col=y_col_disease,
        subset='training',
        batch_size=batch_size,
        seed=1,
        color_mode='rgb',
        shuffle=True,
        class_mode='categorical',
        target_size=(image_size, image_size)
    )
    
    valid_gen_plant = create_data_generator(
        datagen=train_datagen,
        dataframe=dataframe,
        x_col=x_col,
        y_col=y_col_plant,
        subset='validation',
        batch_size=batch_size,
        seed=1,
        color_mode='rgb',
        shuffle=False,
        class_mode='categorical',
        target_size=(image_size, image_size)
    )
    
    valid_gen_disease = create_data_generator(
        datagen=train_datagen,
        dataframe=dataframe,
        x_col=x_col,
        y_col=y_col_disease,
        subset='validation',
        batch_size=batch_size,
        seed=1,
        color_mode='rgb',
        shuffle=False,
        class_mode='categorical',
        target_size=(image_size, image_size)
    )
    
    return train_gen_plant, train_gen_disease, valid_gen_plant, valid_gen_disease

def create_test_generators(dataframe, x_col, y_col_plant, y_col_disease, batch_size, image_size):
    test_datagen = create_test_datagen()
    
    test_gen_plant = create_data_generator(
        datagen=test_datagen,
        dataframe=dataframe,
        x_col=x_col,
        y_col=y_col_plant,
        subset=None,
        batch_size=batch_size,
        seed=1,
        color_mode='rgb',
        shuffle=False,
        class_mode='categorical',
        target_size=(image_size, image_size)
    )
    
    test_gen_disease = create_data_generator(
        datagen=test_datagen,
        dataframe=dataframe,
        x_col=x_col,
        y_col=y_col_disease,
        subset=None,
        batch_size=batch_size,
        seed=1,
        color_mode='rgb',
        shuffle=False,
        class_mode='categorical',
        target_size=(image_size, image_size)
    )
    
    return test_gen_plant, test_gen_disease

# def create_vit_model(image_size, num_classes, vit_type='vit_b32', activation='softmax'):
#     # Create the base ViT model
#     vit_model = vit.vit_b32(
#         image_size=image_size,
#         activation=activation,
#         pretrained=True,
#         include_top=False,
#         pretrained_top=False,
#         classes=5
#     )

#     # Create a Sequential model and add layers
#     model = tf.keras.Sequential([
#         vit_model,
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(11, activation=tfa.activations.gelu),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(38, activation='softmax')
#     ], name='vision_transformer')

#     return model


def compile_model(model, learning_rate=1e-4):
    optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                  metrics=['accuracy'])
    return model

def create_vit_model(image_size,num_disease_classes=38, number_plant_classes=17):
    # Create the base ViT model
    vit_model = vit.vit_b32(
        image_size=image_size,
        activation='softmax',
        pretrained=True,
        include_top=False,
        pretrained_top=False,
        classes=5  # This parameter is only for the base model
    )

    # Create a Sequential model and add layers
    inputs = layers.Input(shape=(image_size, image_size, 3))
    x = vit_model(inputs)
    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    
    # Multi-task outputs
    plant_output = layers.Dense(11, activation=tfa.activations.gelu)(x)
    plant_output = layers.BatchNormalization()(plant_output)
    plant_output = layers.Dense(number_plant_classes, activation='softmax', name='plant_output')(plant_output)

    disease_output = layers.Dense(11, activation=tfa.activations.gelu)(x)
    disease_output = layers.BatchNormalization()(disease_output)
    disease_output = layers.Dense(num_disease_classes, activation='softmax', name='disease_output')(disease_output)

    # model = tf.keras.Model(inputs=inputs, outputs=disease_output, name='vit_disease_model')


    model = tf.keras.Model(inputs=inputs, outputs=[plant_output, disease_output], name='multi_task_vit_model')
    return model


# def compile_model(model, learning_rate=1e-4):
#     optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)
#     model.compile(
#         optimizer=optimizer,
#         loss={
#             'plant_output': tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
#             'disease_output': tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2)
#         },
#         metrics={
#             'plant_output': ['accuracy'],
#             'disease_output': ['accuracy']
#         }
#     )
#     return model

def get_callbacks():
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                     factor=0.2,
                                                     patience=2,
                                                     verbose=1,
                                                     min_delta=1e-4,
                                                     min_lr=1e-6,
                                                     mode='max')

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                     min_delta=1e-4,
                                                     patience=5,
                                                     mode='max',
                                                     restore_best_weights=True,
                                                     verbose=1)

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='./model.hdf5',
                                                      monitor='val_accuracy',
                                                      verbose=1,
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      mode='max')

    callbacks = [earlystopping, reduce_lr, checkpointer]
    return callbacks
