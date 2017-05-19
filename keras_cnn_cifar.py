
## Libraries
import numpy as np
import random
from time import time
import keras
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import keras.callbacks as kcb

# Global Contrast Normalization
def norm_input(x): return (x-mean_px)/std_px


## Read data
def unpickle(file):
    # format
    # with keys ['data', 'labels', 'batch_label', 'filenames']
    # batch['data']: (10000, 3072) numpy array
    # batch['labels']: list with length 10000
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

train = np.zeros((50000, 3072), dtype=np.uint8)
train_label = []
path = './cifar-10-batches-py/'
for i in range(5):
    data_dict = unpickle(path + 'data_batch_' + str(i+1))
    train[i*10000:(i*10000+10000), :] = data_dict['data']
    train_label = train_label + data_dict['labels']


## Split to train, validation sets
train_label = np.array(train_label)
split_size = int(train.shape[0]*0.85)
train_x, val_x = train[:split_size, :], train[split_size:, :]
train_y, val_y = train_label[:split_size].reshape(-1,1), train_label[split_size:].reshape(-1,1)
train_y, val_y = (np.arange(10)==train_y).astype(np.float32), (np.arange(10)==val_y).astype(np.float32)


# Data augmentation: cropping, horizontal reflection
nChannel = 3
img_size = int(np.sqrt(train_x.shape[1]/nChannel))
tr_expand_x = np.zeros((4*train_x.shape[0], nChannel*(img_size-2)**2), dtype='float32')
tr_expand_y = np.zeros((4*train_y.shape[0], train_y.shape[1]), dtype='int8')
i, j = 0, 0
for x, y in zip(train_x, train_y):
	image = np.reshape(x, (-1, img_size, img_size))
	j += 1
	if j % 1000 == 0: print "Expanding image number", j
	# iterate over data telling us the details of how to
	# do the displacement
	for d1, d2 in [(0, 0), (1, 1)]:  # [(0, 0), (0, 2), (2, 0), (2, 2)]
		new_im = image[:, d1:(d1+img_size-2), d2:(d2+img_size-2)]
		tr_expand_x[i], tr_expand_y[i] = new_im.reshape(np.prod(new_im.shape)), y
		i += 1
		new_im_refl = new_im[:, :, ::-1]
		tr_expand_x[i], tr_expand_y[i] = new_im_refl.reshape(np.prod(new_im_refl.shape)), y
		i += 1


# create model
model = Sequential([
	Lambda(norm_input, input_shape=(3,30,30), output_shape=(3,30,30)),
	Convolution2D(32,3,3, activation='relu', border_mode='same'),
	BatchNormalization(axis=1),
	Convolution2D(32,3,3, activation='relu', border_mode='same'),
	MaxPooling2D(pool_size=(2,2)),
	BatchNormalization(axis=1),
	Convolution2D(64,3,3, activation='relu', border_mode='same'),
	BatchNormalization(axis=1),
	Convolution2D(64,3,3, activation='relu', border_mode='same'),
	MaxPooling2D(pool_size=(2,2)),
	BatchNormalization(axis=1),
	Convolution2D(128,3,3, activation='relu', border_mode='same'),
	BatchNormalization(axis=1),
	Convolution2D(128,3,3, activation='relu', border_mode='same'),
	BatchNormalization(axis=1),
	Convolution2D(128,3,3, activation='relu', border_mode='same'),
	MaxPooling2D(pool_size=(2,2)),
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


# compile the model with necessary attributes
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

transfer = False
if transfer:
	model.load_weights('./best_model_cifar.hdf5')


## Training!!
class CallMetric(kcb.Callback):
    def on_train_begin(self, logs={}):
        self.best_acc = 0.0
        self.accs = []
        self.val_accs = []
        self.losses = []
        self.val_losses = []
    def on_epoch_end(self, batch, logs={}):
    	self.accs.append(logs.get('acc'))
    	self.val_accs.append(logs.get('val_acc'))
    	self.losses.append(logs.get('loss'))
    	self.val_losses.append(logs.get('val_loss'))
    	if logs.get('val_acc') > self.best_acc:
    		self.best_acc = logs.get('val_acc')
    		print "\nThe BEST val_acc to date."

epochs = 12
batch_size = 64

print "Start training..."
t0 = time()
metricRecords = CallMetric()
checkpointer = kcb.ModelCheckpoint(filepath="./cifar10_10l.h5", monitor='val_acc', save_best_only=True)
trained_model = model.fit(tr_expand_x.reshape(-1, nChannel, img_size-2, img_size-2), tr_expand_y, 
	nb_epoch=epochs, batch_size=batch_size, callbacks=[metricRecords, checkpointer], 
	validation_data=(val_x.reshape(-1, nChannel, img_size, img_size)[:,:,1:31,1:31], val_y))
print "\nElapsed time:", time()-t0, '\n\n'


model.load_weights('./cifar10_10l.h5')
randSample = random.sample(np.arange(train_dataset.shape[0]), 10000)
pred_tr = model.predict_classes(
	train_x.reshape(-1, nChannel, img_size, img_size)[randSample,:,1:31,1:31])
print "\ntraining accuracy:", np.mean(pred_tr==np.argmax(train_y, 1))

pred_val = model.predict_classes(val_x.reshape(-1, nChannel, img_size, img_size)[:,:,1:31,1:31])
print "validation accuracy:", np.mean(pred_val==np.argmax(val_y, 1))

# Save performance data figure
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.plot(np.arange(epochs)+1, metricRecords.accs, '-o', label='bch_train')
plt.plot(np.arange(epochs)+1, metricRecords.val_accs, '-o', label='validation')
plt.xlim(0, epochs+1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig('cifar10_10l.png')

test_dict = unpickle(path + 'test_batch')
test_x, test_y = test_dict['data'], test_dict['labels']
pred = model.predict_classes(test_x.reshape(-1, nChannel, img_size, img_size)[:,:,1:31,1:31])
print "test accuracy:", sum(pred==test_y)/float(test_x.shape[0])


# export Kaggle submission file
'''
labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
fh = open('submit_cifar10.txt','w+')
fh.write('id,label\n')
for i in xrange(len(pred)):
    fh.write('%d,%s\n' %(i+1, labels[pred[i]]))
fh.close()
'''
path = 'kaggle-cifar10-test/'
test_batches = gen.flow_from_directory(path, target_size=(30,30), 
	shuffle=False, batch_size=64, class_mode=None)
model.load_weights('cifar10_10l.h5')
pred = model.predict_generator(test_batches, test_batches.nb_sample)
pred_test = np.argmax(pred, 1)
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
submit_labels = [labels[y] for y in pred_test]
submit = pd.DataFrame({'id': np.arange(len(submit_labels))+1, 'label': submit_labels})
submit.to_csv('submit_cifar10.csv', index=False)

