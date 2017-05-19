from utils import *

def get_model_bn():
	model = Sequential([
		Lambda(norm_input, input_shape=(3,30,30), output_shape=(3,30,30)),
		Convolution2D(32,3,3, activation='relu', border_mode='same'),
		BatchNormalization(axis=1),
		Convolution2D(32,3,3, activation='relu', border_mode='same'),
		MaxPooling2D(),
		BatchNormalization(axis=1),
		Convolution2D(64,3,3, activation='relu', border_mode='same'),
		BatchNormalization(axis=1),
		Convolution2D(64,3,3, activation='relu', border_mode='same'),
		MaxPooling2D(),
		BatchNormalization(axis=1),
		Convolution2D(128,3,3, activation='relu', border_mode='same'),
		BatchNormalization(axis=1),
		Convolution2D(128,3,3, activation='relu', border_mode='same'),
		BatchNormalization(axis=1),
		Convolution2D(128,3,3, activation='relu', border_mode='same'),
		MaxPooling2D(),
		Flatten(),
		BatchNormalization(),
		Dense(1024, init='he_normal'),
		BatchNormalization(),
		Activation('relu'),
		Dropout(0.5),
		Dense(1024, init='he_normal'),
		BatchNormalization(),
		Activation('relu'),
		Dropout(0.5),
		Dense(10, activation='softmax')
		])
	model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def predict(model):
	test_batches = gen.flow_from_directory(path, target_size=(30,30), 
		shuffle=False, batch_size=256, class_mode=None)
	return model.predict_generator(test_batches, test_batches.nb_sample)

path = '/Users/radream/Desktop/cifar-10/kaggle-cifar10-test'
gen = image.ImageDataGenerator()
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 
			'ship', 'truck']

#model = get_model_bn()
#model.load_weights('cnn_10l.h5')
#t0 = time()
#pred = predict(model)
#print 'Spending %.3f seconds predicting.' %(time()-t0)  # 2348.654 sec
#pred_test = np.argmax(pred, 1)

all_preds = []
for i in range(6):
	model = get_model_bn()
	model.load_weights('model_'+str(i+1)+'.h5')
	pred = predict(model)
	all_preds.append(pred)
all_preds = np.stack(all_preds)
avg_pred = np.mean(all_preds, axis=0)
pred_test = np.argmax(avg_pred, 1)

submit_labels = [labels[y] for y in pred_test]
submit = pd.DataFrame({'id': np.arange(len(submit_labels))+1, 'label': submit_labels})
submit.to_csv('submit_cifar10.csv', index=False)

# 2457.75121021 seconds for original neurons
# 4273.35562301 seconds for double neurons