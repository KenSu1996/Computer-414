import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
from sklearn.metrics import det_curve, roc_curve, roc_auc_score
with open('../results/vacc_ethnicity_all_synth_finetuned.pkl', 'rb') as f:
        loaded_dict_g_cmu = pickle.load(f)
loaded_list_g_cmu = []
for k in  loaded_dict_g_cmu.keys():
    loaded_list_g_cmu.append(loaded_dict_g_cmu[k])
std_gar_cmu = []
far = []
for key in loaded_list_g_cmu[0].keys():
    temp = []
    for i in range(len(loaded_list_g_cmu)):
        try:
            temp.append(loaded_list_g_cmu[i][key])
        except:
            continue
    std_gar_cmu.append(np.std(temp, ddof=1)*100)
    far.append(key*100)

# Define fixed x positions for four bars
x = [1, 2, 3, 4]
bar_width = 0.35  # Fixed width for each bar

# Plot the first set of bars
plt.bar(x, std_gar_cmu[0:4], bar_width, align='edge', facecolor='dodgerblue', edgecolor='white', lw=2, label='Synthetic Faces')
for i in range(4):
    plt.annotate(str(std_gar_cmu[i]), xy=(x[i], std_gar_cmu[i]), ha='left', va='bottom')



plt.xlabel("FAR(%)")
plt.ylabel("GAR Standard Deviation")
plt.title("FAR Vs Std(GAR) - Ethnicity")
plt.legend()
plt.show()

with open('../results/vacc_gender_all_synth_finetuned.pkl', 'rb') as f:
        loaded_dict_g_cmu = pickle.load(f)
loaded_list_g_cmu = []
for k in  loaded_dict_g_cmu.keys():
    loaded_list_g_cmu.append(loaded_dict_g_cmu[k])
std_gar_cmu = []
far = []
for key in loaded_list_g_cmu[0].keys():
    temp = []
    for i in range(len(loaded_list_g_cmu)):
        try:
            temp.append(loaded_list_g_cmu[i][key])
        except:
            continue
    std_gar_cmu.append(np.std(temp, ddof=1)*100)
    far.append(key*100)
    


plt.bar(x, std_gar_cmu[0:4], bar_width, align='edge', facecolor='dodgerblue', edgecolor='white', lw=2, label='Synthetic Faces')
for i in range(4):
    plt.annotate(str(std_gar_cmu[i]), xy=(x[i], std_gar_cmu[i]), ha='left', va='bottom')
plt.xlabel("FAR(%)")
plt.ylabel("GAR Standard Deviation")
plt.title("FAR Vs Std(GAR) - Gender")
plt.legend()
plt.show()


with open('../results/vacc_attrib_all_synth_finetuned.pkl', 'rb') as f:
        loaded_dict_g_cmu = pickle.load(f)
loaded_list_g_cmu = []
for k in  loaded_dict_g_cmu.keys():
    loaded_list_g_cmu.append(loaded_dict_g_cmu[k])
std_gar_cmu = []
far = []
for key in loaded_list_g_cmu[0].keys():
    temp = []
    for i in range(len(loaded_list_g_cmu)):
        try:
            temp.append(loaded_list_g_cmu[i][key])
        except:
            continue
    std_gar_cmu.append(np.std(temp, ddof=1)*100)
    far.append(key*100)
    
    
plt.bar(x, std_gar_cmu[0:4], bar_width, align='edge', facecolor='dodgerblue', edgecolor='white', lw=2, label='Synthetic Faces')
for i in range(4):
    plt.annotate(str(std_gar_cmu[i]), xy=(x[i], std_gar_cmu[i]), ha='left', va='bottom')
plt.xlabel("FAR(%)")
plt.ylabel("GAR Standard Deviation")
plt.title("FAR Vs Std(GAR) - Attributes")
plt.legend()
plt.show()


plt.bar(x, std_gar_cmu[0:4], bar_width, align='edge', facecolor='dodgerblue', edgecolor='white', lw=2, label='Synthetic Faces')
for i in range(4):
    plt.annotate(str(std_gar_cmu[i]), xy=(x[i], std_gar_cmu[i]), ha='left', va='bottom')
plt.xlabel("FAR(%)")
plt.ylabel("GAR Standard Deviation")
plt.title("FAR Vs Std(GAR) - Attributes")
plt.legend()
plt.show()