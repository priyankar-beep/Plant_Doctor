from utility import *
##############################################################
IMAGES_PATH = '/home/hubble/Downloads/plnat_disease_papers/plant_village/data'
##############################################################
image_category_mapping, image_plant_mapping = map_images_to_category_and_plant(IMAGES_PATH)
##############################################################
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 7
image_paths = list(image_category_mapping.keys())
NUM_IMAGES = len(image_paths)
TEST_SIZE = int(NUM_IMAGES*0.2)
TRAIN_SIZE = int(NUM_IMAGES*0.8)
VAL_SIZE = int(TRAIN_SIZE*0.2)
captions_path = 'captions.txt'
disease_captions = load_disease_captions(captions_path)
#################################################################
random.shuffle(image_paths)
train_paths = image_paths[:TRAIN_SIZE]
val_paths=image_paths[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
test_paths = image_paths[NUM_IMAGES-TEST_SIZE:]
#################################################################
DF_TRAIN = pd.DataFrame({'filename': train_paths, 'disease_name': [image_category_mapping[path] for path in train_paths], 'plant_name': [image_plant_mapping[path] for path in train_paths]})
DF_TEST = pd.DataFrame({'filename': test_paths,   'disease_name': [image_category_mapping[path]for path in   test_paths], 'plant_name': [image_plant_mapping[path] for path in test_paths]})

df_combined = create_combined_dataframe(image_category_mapping, image_plant_mapping, disease_captions)
df_combined['caption'] = df_combined['caption'].astype(str) ## for healthy images there is no string, so it may create nan, that is why i need to make a string
DF_TRAIN, DF_TEST = split_dataframe(df_combined)
## Create data augumentation
train_gen_plant, train_gen_disease, valid_gen_plant, valid_gen_disease = create_train_valid_generators(
    dataframe=df_combined,#DF_TRAIN,
    x_col='filename',
    y_col_plant='plant_name',
    y_col_disease='disease_name',
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE
)

images, labels = next(train_gen_plant)

# # Print shapes
# print(f"Images shape: {images.shape}")
# print(f"Labels shape: {labels.shape}")

# # Optionally, print some sample data
# print(f"First image in the batch: {images[0]}")
# print(f"First label in the batch: {labels[0]}")

# Create data generators for testing
test_gen_plant, test_gen_disease = create_test_generators(
    dataframe=DF_TEST,
    x_col='filename',
    y_col_plant='plant_name',
    y_col_disease='disease_name',
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE
)


model = create_vit_model(image_size=224,num_disease_classes=38,number_plant_classes=17)
print(model.summary())
model = compile_model(model, learning_rate=.001)
image_branch = model
callbacks = get_callbacks()
STEP_SIZE_TRAIN = train_gen_disease.n // train_gen_disease.batch_size
STEP_SIZE_VALID = valid_gen_disease.n // valid_gen_disease.batch_size

multi_task_train_gen = MultiTaskDataGenerator(train_gen_plant, train_gen_disease)
multi_task_valid_gen = MultiTaskDataGenerator(valid_gen_plant, valid_gen_disease)

history = model.fit(
    multi_task_train_gen,
    steps_per_epoch=5,  
    validation_data=multi_task_valid_gen,
    validation_steps=5,
    epochs=2,
    callbacks=callbacks
)

### Now update the code according to captions.txt
# text_dataset = create_text_dataset(input_ids, attention_masks, batch_size=32)


def create_text_branch(max_length=128):
    input_ids = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")
    
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    bert_outputs = bert_model([input_ids, attention_mask])
    
    pooled_output = bert_outputs.pooler_output
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(pooled_output)
    
    text_branch = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output_layer)
    
    return text_branch

text_branch = create_text_branch(max_length=128)

# Combine both branches
def create_combined_model(image_branch, text_branch):
    # Image inputs
    image_inputs = image_branch.input
    image_outputs = image_branch(image_inputs)
    
    # Text inputs
    text_inputs = text_branch.input
    text_outputs = text_branch(text_inputs)
    
    combined_output = tf.keras.layers.Concatenate()([
        image_outputs[0],  # plant_output
        image_outputs[1],  # disease_output
        text_outputs       # text branch output
    ])
    
    combined_output = tf.keras.layers.Dense(128, activation='relu')(combined_output)
    final_output = tf.keras.layers.Dense(1, activation='sigmoid')(combined_output)
    combined_model = tf.keras.Model(inputs=[image_inputs, text_inputs], outputs=final_output)
    return combined_model

# Create the combined model
image_branch = create_vit_model(image_size=224, num_disease_classes=38, number_plant_classes=17)
text_branch = create_text_branch(max_length=128)
combined_model = create_combined_model(image_branch, text_branch)
combined_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
combined_model.summary()

######################################
class CombinedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_generator, text_inputs, labels, batch_size):
        self.image_generator = image_generator
        self.text_inputs = text_inputs
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return len(self.image_generator)

    def __getitem__(self, index):
        # Get image data
        images, image_labels = self.image_generator[index]
        # Get corresponding text inputs
        text_batch = self.text_inputs[index * self.batch_size:(index + 1) * self.batch_size]
        # Assume labels are properly aligned
        combined_labels = {
            'plant_output': image_labels['plant_output'],
            'disease_output': image_labels['disease_output']
        }
        return [images, text_batch['input_ids'], text_batch['attention_mask']], combined_labels

combined_train_gen = CombinedDataGenerator(train_gen_disease, train_texts_inputs, train_labels, batch_size=16)

def encode_texts(texts, max_length=128):
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='tf'
    )
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

encoded_texts = encode_texts(df_combined['caption'].tolist())

import tensorflow as tf
from transformers import BertTokenizer
import numpy as np

class TextDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, texts, plant_labels, disease_labels, tokenizer, max_length, batch_size):
        self.texts = texts
        self.plant_labels = plant_labels
        self.disease_labels = disease_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_samples = len(texts)

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        # Get batch indices
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, self.num_samples)
        
        # Get batch texts and labels
        batch_texts = self.texts[batch_start:batch_end]
        batch_plant_labels = self.plant_labels[batch_start:batch_end]
        batch_disease_labels = self.disease_labels[batch_start:batch_end]
        
        # Tokenize batch texts
        encodings = self.tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='tf'
        )
        
        # Prepare input tensors
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        
        # Prepare labels
        plant_labels = tf.convert_to_tensor(batch_plant_labels, dtype=tf.float32)  # Adjust dtype if necessary
        disease_labels = tf.convert_to_tensor(batch_disease_labels, dtype=tf.float32)  # Adjust dtype if necessary
        
        combined_labels = {
            'plant_output': plant_labels,
            'disease_output': disease_labels
        }
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask}, combined_labels

    def on_epoch_end(self):
        # Shuffle data if needed
        pass


labels = prepare_labels(DF_TRAIN)

texts_train = DF_TRAIN['caption'].tolist() # List of text samples for training
plant_labels_train = labels['plant_output']  # List of plant labels for training
disease_labels_train = labels['disease_output']  # List of disease labels for training


text_gen = TextDataGenerator(
    texts=texts_train,
    plant_labels=plant_labels_train,
    disease_labels=disease_labels_train,
    tokenizer=tokenizer,
    max_length=128,
    batch_size=32
)
# Example of preparing labels for the generator



# # Split data for training and validation
# train_size = int(0.8 * len(df_combined))
# train_df = df_combined[:train_size]
# valid_df = df_combined[train_size:]

# # Encode texts for training and validation
# train_encoded_texts = encode_texts(train_df['caption'].tolist())
# valid_encoded_texts = encode_texts(valid_df['caption'].tolist())

# # Prepare labels for training and validation
# train_labels = prepare_labels(train_df)
# valid_labels = prepare_labels(valid_df)

# # Create data generators
# train_gen_disease_text = TextDataGenerator(train_encoded_texts, train_labels, batch_size=32)
# valid_gen_disease_text = TextDataGenerator(valid_encoded_texts, valid_labels, batch_size=32)


