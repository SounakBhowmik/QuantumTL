# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:57:30 2024

@author: sbhowmi2
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
#%% normalize data
def normalize(a, t_max=1, t_min=0):
    return (a-min(a))/(max(a)-min(a)) * (t_max - t_min) + t_min



#%%
classical_model_results_file_path = "RESULTS/test_classical_results_dictionary_3.pkl"
quanvolution_model_results_file_path = "RESULTS/HybridModel_Quanvolution_k_fold_cross_val/hybridModel_Quanvolution_based_k-fold_cross_val_results_dictionary.pkl"
hybrid_models_results_file_path = "RESULTS/HybridModels_k_fold_cross_val/hybridModel_transfer_learning_k-fold_cross_val_results_dictionary.pkl"


with open(classical_model_results_file_path, 'rb') as f:
    classical_results = pickle.load(f)
    
with open(quanvolution_model_results_file_path, 'rb') as f:
    quanvolution_results = pickle.load(f)
    

with open(hybrid_models_results_file_path, 'rb') as f:
    hybrid_tf_results = pickle.load(f)

#%%
plt.figure(figsize=(10,8), dpi=200)
plt.plot(np.array(range(0,120,4)), classical_results['ClassicalModel2_5132595725441423083']['test_acc'], label="classical_model")
plt.plot(quanvolution_results['HybridModel_Quanvolution_based']['test_acc'], label="Quanvolutional_model")
plt.xlabel('epochs')
plt.xlabel('test_accuracy')
plt.legend()
plt.show()

#%%
for key in classical_results.keys():
    model = classical_results[key]
    print(f"{key}: recall: {model['recall']}, precision: {model['precision']}, F1_score: {model['F1_score']}, parameters: {model['params_number']}")

#%%
import pandas as pd
results_df = pd.read_excel('results_quantum_surface_anomaly_detectionh.xlsx')
print(results_df)


#%%
plt.figure(figsize=(10,8), dpi=100)
for key in hybrid_model1.keys():
    model = hybrid_model1[key]
    #plt.plot(model['test_loss'], label=key)
    print(f"{key}: best_test_accuracy: {max(model['test_acc'])}")
'''
plt.legend()
plt.plot()
'''

#%%
def get_best_results(model_dict):
    best_results = {}
    for key in model_dict.keys():
        model = model_dict[key]
        
        #best_res_loc = np.argmax(np.array(model['F1_score']))
        best_results[key] = [model['test_acc'][-1], model['precision'], model['recall'], model['F1_score']]
        
        #best_results[key] = [max(model['test_acc']), model['precision'], model['recall'], model['F1_score']]
    return best_results

#%%
import pandas as pd
performance_data_df = pd.read_excel('results_quantum_surface_anomaly_detection.xlsx', sheet_name='Sheet3')
print(performance_data_df.head())

#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8), dpi=100)
plt.scatter(np.array([1,2,3,4]),performance_data_df[performance_data_df['type' ]== 'Classical']["Best test accuracy"], label="Classical")
plt.scatter(np.array([1,2,3,4]),performance_data_df[performance_data_df['type' ]== 'Hybrid']["Best test accuracy"], label="Hybrid")
plt.legend()
    

#%%
import matplotlib.pyplot as plt
cm_res = ['ClassicalModel1_1683709579050736241','ClassicalModel2_11926550140455977890','ClassicalModel3_6849243949351946134','ClassicalModel4_4305358886776407372']
plt.figure(figsize=(10,8), dpi=200)
plt.style.use('seaborn-v0_8-poster')
ma = 1
mi = 0
colors = ['blue', 'orange', 'brown', 'green']  # Colors for the models
p=0
idx = np.array(range(1,120,3))
for k in classical_results.keys():
    if(k in cm_res):
        #plt.plot(normalize(np.array(classical_results[k]['test_acc'])[idx], ma, mi), label=f"Model {k.split('_')[0][-1]}", c=colors[p], marker='.')
        plt.plot(idx, np.array(classical_results[k]['test_acc'])[idx], label=f"Model {k.split('_')[0][-1]}", c=colors[p], marker='.')
        p+=1
plt.xlabel('Epochs')
plt.ylabel('Test accuracy')
plt.legend()


#%%
hybrid_model1_TL_results_file_path = "RESULTS/HybridModel_TL_based_k_fold_cross_val_4/hybridModel1_transfer_learning_k-fold_cross_val_results_dictionary.pkl"


with open(hybrid_model1_TL_results_file_path, 'rb') as f:
    hybrid_model1 = pickle.load(f)
    
#%%
hybrid_model2_TL_results_file_path = "RESULTS/HybridModel_TL_based_k_fold_cross_val_2/hybridModel2_transfer_learning_k-fold_cross_val_results_dictionary.pkl"


with open(hybrid_model2_TL_results_file_path, 'rb') as f:
    hybrid_model2 = pickle.load(f)
#%%
hybrid_model3_TL_results_file_path = "RESULTS/HybridModel_TL_based_k_fold_cross_val_2/hybridModel3_transfer_learning_k-fold_cross_val_results_dictionary.pkl"


with open(hybrid_model3_TL_results_file_path, 'rb') as f:
    hybrid_model3 = pickle.load(f)

#%%
hybrid_model4_TL_results_file_path = "RESULTS/HybridModel_TL_based_k_fold_cross_val_2/hybridModel4_transfer_learning_k-fold_cross_val_results_dictionary.pkl"


with open(hybrid_model4_TL_results_file_path, 'rb') as f:
    hybrid_model4 = pickle.load(f)
#%%
#Classical Transfer learning equivalent models
classical_model1_TL_results_file_path = "RESULTS/ClassicalModel_TL_based_k_fold_cross_val/classicalModel1_transfer_learning_k-fold_cross_val_results_dictionary.pkl"
classical_model2_TL_results_file_path = "RESULTS/ClassicalModel_TL_based_k_fold_cross_val/classicalModel2_transfer_learning_k-fold_cross_val_results_dictionary.pkl"
classical_model3_TL_results_file_path = "RESULTS/ClassicalModel_TL_based_k_fold_cross_val/classicalModel4_transfer_learning_k-fold_cross_val_results_dictionary.pkl"

with open(classical_model1_TL_results_file_path, 'rb') as f:
    cm1_res = pickle.load(f)
with open(classical_model2_TL_results_file_path, 'rb') as f:
    cm2_res = pickle.load(f)
with open(classical_model3_TL_results_file_path, 'rb') as f:
    cm3_res = pickle.load(f)

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import torch
from matplotlib import style
import matplotlib

#%%
font = {'family' : 'arial',
        'size'   : 30}



#%%
import torch
Styles = plt.style.available

#for s in Styles:
s = 'seaborn-v0_8-poster'
plt.style.use(s)



hmtl_1 = 'HybridModel1_transferLearning_based_1'
hmtl_2 = 'HybridModel2_transferLearning_based_1'
hmtl_3 = 'HybridModel3_transferLearning_based_1'
hmtl_4 = 'HybridModel4_transferLearning_based_1'

r = 2

test_accuracies = [hybrid_model2[hmtl_2]['test_acc'][:40],
                   hybrid_model3[hmtl_3]['test_acc'][:40],
                   hybrid_model4[hmtl_4]['test_acc'][:40]]
 
test_losses = [normalize(torch.tensor(hybrid_model2[hmtl_2]['test_loss'][:40]).numpy(), r,0),
               normalize(torch.tensor(hybrid_model3[hmtl_3]['test_loss'][:40]).numpy(), r,0),
               normalize(torch.tensor(hybrid_model4[hmtl_4]['test_loss'][:40]).numpy(), r,0)]
 



# Assuming epochs or some other x-axis metric is common across models
epochs = np.array(range(len(test_losses[0])))+1
 
# Plotting
colors = ['blue', 'orange',  'green']  # Colors for the models
#plt.figure(figsize=(10, 8), dpi=400)
fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
 
for i in range(len(test_accuracies)):
    # Plot test accuracy with solid line
    ax.plot(epochs, test_accuracies[i], color=colors[i], linestyle='-')
    # Plot test loss with dotted line
    ax.plot(epochs, test_losses[i], color=colors[i], linestyle='--')
 
# Creating custom legends
model_legends = [mlines.Line2D([], [], color=color, marker='_', linestyle='-', label=f'QTL-M-3 (CM{i+1})') for i, color in enumerate(colors)]
accuracy_legend = mlines.Line2D([], [], color='black', linestyle='-', label='Test Accuracy')
loss_legend = mlines.Line2D([], [], color='black', linestyle='--', label='Test Loss')
 

 
#plt.title(s)
#ax.ylim(-0.1,2.1)
#ax.xlim(0,42)
ax.set_xlabel('Epochs', fontsize=25)
ax.set_ylabel('Metrics', fontsize=25)
ax.grid(True) 


# Adding legends to the plot
leg = ax.legend(handles=model_legends + [accuracy_legend, loss_legend], loc='best', prop = {"size":25})

plt.show()


#%%
def std(lst):
    lst = np.array(lst)
    return (lst-min(lst))/(max(lst)-min(lst))

#%% Plots for hybrid quanvolution based model
import pickle
file_path = "RESULTS/HybridModel_Quanvolution_k_fold_cross_val/ConvModel1_for_comparison_k-fold_cross_val_results_dictionary.pkl"
with open(file_path, 'rb') as f:
    hybrid_quanv_model = pickle.load(f)

#%%
Styles = plt.style.available

#for s in Styles:
s = 'seaborn-v0_8-poster'
plt.style.use(s)

r = 2    
    
test_accuracies = []
test_losses = []

for k in hybrid_quanv_model.keys():
    m = hybrid_quanv_model[k]
    test_accuracies.append(m['test_acc'])
    test_losses.append(normalize(torch.tensor(m['test_loss']).numpy(), r,0))

c = ['royalblue', 'maroon', 'darkorange', 'cyan', 'saddlebrown','magenta']

fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
for i in range(len(test_accuracies)):
    ax.plot(test_accuracies[i], label=f"Fold-{i+1}", c=c[i])
    
ax.set_xlabel('Epochs', fontsize=25)
ax.set_ylabel('Test accuracy', fontsize=25)
ax.grid(True) 
ax.legend()
#%%











