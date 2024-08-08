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

