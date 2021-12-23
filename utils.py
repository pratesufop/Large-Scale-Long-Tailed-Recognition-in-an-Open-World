import tensorflow
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import cv2
import umap as umap
import umap.plot
import tensorflow_datasets as tfds
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.stats import pareto
from collections import Counter
import seaborn as sns

def load_minst_data(outdir, open_class = None, b = 6):

	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = (x_train.astype(np.float32) - 127.5)/127.5

	x_test = (x_test.astype(np.float32) - 127.5)/127.5
	
	open_test = []
	
	if open_class != None:
	    x_train = x_train[y_train!= open_class]
	    y_train = y_train[y_train!= open_class]

	    open_test = x_test[y_test == open_class]
	    x_test = x_test[y_test!= open_class]
	    y_test = y_test[y_test!= open_class]

	    num_cls = 9 # (10 minus the open_class)
	else:
	    num_cls = 10 # 
    
	# generating a distribution of samples/class with Pareto dist.
	x = np.linspace(pareto.ppf(0.01, b),
			pareto.ppf(0.99, b), num_cls)
	rv = pareto(b)
	probs = rv.pdf(x)/max(rv.pdf(x))

	counts = dict(Counter(y_train))

	nums = list(counts.keys())
	values = list(counts.values())

	idx = np.argsort(values)[::-1]
	values = np.array(values)[idx]
	nums = np.array(nums)[idx]
	
	x_train_lt, y_train_lt = [], []
	final_idx = []
	
	np.random.shuffle(probs)
	
	for v,n,p in zip(values, nums, probs):
	    final_idx.extend(np.random.choice(np.where(y_train == n)[0], int(v*p)))

	    
	np.random.shuffle(final_idx)
	x_train_lt = x_train[final_idx]
	y_train_lt = y_train[final_idx]
	
	plt.figure(figsize=(20,10))

	plt.subplot(121)
	w = Counter(y_train_lt)
	plt.bar(w.keys(), w.values())
	plt.title('Long Tail')

	plt.subplot(122)
	w = Counter(y_train)
	plt.bar(w.keys(), w.values())
	plt.title('Normal')
	
	plt.savefig(os.path.join(outdir,'MNIST-LT.png'))
	
	return (x_train, y_train, x_test, y_test, x_train_lt, y_train_lt, open_test)
    
# generating balanced data
def data_generator(x, y, y_enc, batch_size, oltr = False):
    
	cls_ = np.unique(y)
	num_samples = int(batch_size/len(cls_))

	idx_cls = {}
	for i in cls_:
		idx_cls[i] = np.where(y == i)[0]

	while True:

		x_batch, y_batch, y_batch_enc = [], [], []

		for i in cls_:

			idx = np.random.choice(idx_cls[i], num_samples)

			x_batch.extend(x[idx])
			y_batch.extend(y[idx])
			y_batch_enc.extend(y_enc[idx])

			if len(x_batch) == batch_size:

				x_batch, y_batch, y_batch_enc = np.array(x_batch), np.array(y_batch), np.array(y_batch_enc)

				if oltr:
				    yield([x_batch, y_batch_enc],[y_batch_enc, y_batch])
				else:
				    yield(x_batch,y_batch_enc)
				    
				x_batch, y_batch_enc, y_batch = [], [], []

   
def umap_plot(oltr_on, model, x_test, y_test, test_labels_enc, outdir):
	
	if oltr_on:
	    target_layer = 'v_meta'
	    input_test = [x_test, test_labels_enc]
	else:
	    target_layer = 'feats'
	    input_test = x_test

	feat_model = tensorflow.keras.Model(inputs = model.inputs, outputs= model.get_layer(target_layer).output)
	all_feats = feat_model.predict(input_test)

	classes = np.unique(y_test)
	embedding = umap.UMAP(n_neighbors=11).fit_transform(np.array(all_feats))

	_, ax = plt.subplots(1, figsize=(14, 10))
	plt.scatter(*embedding.T, s=20.0, c=y_test, cmap='jet_r', alpha=1.0)
	plt.setp(ax, xticks=[], yticks=[])
	plt.title('UMAP Embedding', fontsize=14)
	cbar = plt.colorbar(boundaries=np.arange(len(classes)+1)-0.5)
	cbar.set_ticks(np.arange(len(classes)))
	cbar.set_ticklabels(classes)
	plt.tight_layout()
	
	plt.savefig(os.path.join(outdir,'umap.png'))
	
def open_set_eval(model, open_test, x_test, test_labels_enc, outdir, num_cls = 9):
	# Evaluating the reachabilty

	reachability_model = tensorflow.keras.Model(inputs = model.inputs, outputs= model.get_layer('reachability').output)
	open_data = reachability_model.predict([open_test, np.random.rand(len(open_test), num_cls)])
	closed_data = reachability_model.predict([x_test, test_labels_enc])
	
	fig = plt.figure(figsize=(10,6))
	
	sns.distplot(open_data)
	sns.distplot(closed_data)
	
	fig.legend(labels=['open','closed'])
	
	plt.savefig(os.path.join(outdir,'reachability.png'))

def convert_labels(test_labels_enc, pred):
	map_ = '012345789'
	
	y_pred = np.argmax(pred, axis=1)
	y_pred = [int(map_[i]) for i in y_pred]

	y_true = np.argmax(test_labels_enc, axis=1)
	y_true = [int(map_[i]) for i in y_true]
	
	return y_true, y_pred
	
def evaluate_model(y_test, y_pred, outdir):
	
	y_test, y_pred = convert_labels(y_test, y_pred)
	
	cm = confusion_matrix(y_test, y_pred)
	cm = cm/np.sum(cm, axis=1)
	cm = np.round(cm, 2)

	df_cm = pd.DataFrame(cm, index = [i for i in "012345789"],
		      columns = [i for i in "012345789"])

	plt.figure(figsize = (10,7))
	sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
	
	plt.savefig(os.path.join(outdir, 'confusion_matrix.png'))
	
	norm_acc = 100*accuracy_score(y_test, y_pred, normalize= True)	
	print('Norm. Acc. %.2f' % (norm_acc))
	
	return norm_acc
