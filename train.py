from utils import load_minst_data, data_generator, evaluate_model, open_set_eval, umap_plot
from models import get_model
from scheduler import warmup_cosine_method
import tensorflow
import numpy as np
import os
import tensorflow.keras.backend as K
import json
import argparse
from sklearn.preprocessing import OneHotEncoder


np.random.seed(3010)

# evitar erro de alocação de memória
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

my_parser = argparse.ArgumentParser()

my_parser.add_argument('Path',
                       metavar='path',
                       type=str,
                       help='Path to the experiment config.json')

args = my_parser.parse_args()

config_path = args.Path

with open(config_path) as data_file:
    params = json.load(data_file)

params['outdir'] = os.path.dirname(config_path)

oltr_on = params['oltr_on']
modulate = params['modulate']
bs = params['batch_size']
num_epochs = params['num_epochs']
cls_weight, cl_weight = params['cls_weight'], params['cl_weight']
outdir = params['outdir']

if not os.path.exists(outdir):
    os.makedirs(outdir)

# loading the data
x_train, y_train, x_test, y_test, x_train_lt, y_train_lt, open_test =  load_minst_data(outdir = outdir, open_class = 2, b = 6)

model = get_model(num_outputs = 9, oltr = oltr_on, modulate = modulate)
model.summary()

losses = []
lw = []
losses.append(tensorflow.keras.losses.CategoricalCrossentropy())
lw.append(cls_weight)

if oltr_on:
   losses.append(lambda y_true,y_pred: y_pred)
   lw.append(cl_weight)

model.compile(loss= losses,
            optimizer= tensorflow.keras.optimizers.Adam(),
            loss_weights=lw,
            metrics={'cls': tensorflow.keras.metrics.CategoricalAccuracy()})

lr_method = warmup_cosine_method(total_epochs = num_epochs, batch_size = bs, num_files = len(x_train), warmup_epoch = 10, base_lr = 0.001)
callback_list = [lr_method]

enc = OneHotEncoder(handle_unknown='ignore')
train_labels_enc = enc.fit_transform(y_train_lt.reshape([-1,1])).toarray()
test_labels_enc = enc.transform(y_test.reshape([-1,1])).toarray()

gen_ = data_generator(x_train_lt, y_train_lt, train_labels_enc, bs, oltr = oltr_on)

if oltr_on:
    val_data = ([x_test, test_labels_enc], [test_labels_enc, np.random.rand(len(y_test),1)])
else:
    val_data =  (x_test,test_labels_enc)

training_output = model.fit(gen_, 
                            validation_data = val_data, batch_size = bs , 
                            epochs =  num_epochs, steps_per_epoch =  int(len(x_train_lt) / bs ), 
                            callbacks= callback_list, shuffle= True)

pred = model.predict(val_data[0])
    
if isinstance(pred, list):
    pred = pred[0]
    
# evaluating the model
nacc = evaluate_model(pred, test_labels_enc, outdir)

# open set evaluation
if oltr_on:
	open_set_eval(model, open_test, x_test, test_labels_enc, outdir)

# umap plot
umap_plot(oltr_on, model, x_test, y_test, test_labels_enc, outdir)

# saving the result
output_params = {}
output_params['nacc'] =  '%.2f' % (nacc)

with open(os.path.join(outdir, 'result.json'), 'w') as data_file:
    json.dump(output_params, data_file)
