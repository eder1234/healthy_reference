# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 09:12:18 2022

@author: felima
"""

from tkinter import *
from tkinter.filedialog import askopenfilename

import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import seaborn as sns

#%% Ouverture de fichiers
fichier_mean = askopenfilename()
df2_mean = pd.read_csv(fichier_mean, usecols=range(3,108))
df_mean = df2_mean.rename({'mm': 'X1', 'mm.1': 'Y1', 'mm.2': 'Z1', 'mm.3': 'X2', 'mm.4': 'Y2', 'mm.5': 'Z2', 'mm.6': 'X3', 'mm.7': 'Y3', 'mm.8': 'Z3'}, axis=1)
display(df_mean)

fichier_sd = askopenfilename()
df2_sd = pd.read_csv(fichier_sd, usecols=range(3,108))
df_sd = df2_sd.rename({'mm': 'X1', 'mm.1': 'Y1', 'mm.2': 'Z1', 'mm.3': 'X2', 'mm.4': 'Y2', 'mm.5': 'Z2', 'mm.6': 'X3', 'mm.7': 'Y3', 'mm.8': 'Z3'}, axis=1)
display(df_sd)

#%% Nomination des matrices


Mean = df_mean.to_numpy()
sd = df_sd.to_numpy()

#%% Calcul de la moyenne et des limites de sd
Size_L = np.size(Mean[:,0])
Size_C = np.size(Mean[0,:])

Mean_Final = np.zeros((105,1))
Deviation = np.zeros((105,1))


for i in range(Size_L): #i représente les lignes
    for j in range(105): #j représente les colonnes
        Mean_Final[j]= np.mean(Mean[1:Size_L,j])
        Deviation[j] = np.std(Mean[1:Size_L,j])
        
        
Deviation_1 = Mean_Final + Deviation
Deviation_2 = Mean_Final - Deviation

#%% Reorganisation des données pour le Faciograph

Mean_Final_order1=np.array([[Mean_Final[6,0]],[Mean_Final[5,0]],[Mean_Final[4,0]],[Mean_Final[17,0]],[Mean_Final[16,0]],[Mean_Final[15,0]],[Mean_Final[14,0]],[Mean_Final[13,0]],[Mean_Final[31,0]],[Mean_Final[30,0]],
[Mean_Final[29,0]],[Mean_Final[28,0]],[Mean_Final[27,0]],[Mean_Final[103,0]],[Mean_Final[104,0]],[Mean_Final[34,0]],[Mean_Final[35,0]],[Mean_Final[36,0]],[Mean_Final[37,0]],[Mean_Final[38,0]],
[Mean_Final[39,0]],[Mean_Final[33,0]],[Mean_Final[32,0]],[Mean_Final[92,0]],[Mean_Final[91,0]],[Mean_Final[90,0]],[Mean_Final[89,0]],[Mean_Final[88,0]],                                                                                                                                          
[Mean_Final[83,0]],[Mean_Final[84,0]],[Mean_Final[85,0]],[Mean_Final[86,0]],[Mean_Final[87,0]],[Mean_Final[82,0]],[Mean_Final[81,0]],[Mean_Final[80,0]],[Mean_Final[79,0]],[Mean_Final[78,0]],
[Mean_Final[77,0]],[Mean_Final[40,0]],[Mean_Final[41,0]],[Mean_Final[42,0]],[Mean_Final[67,0]],[Mean_Final[74,0]],[Mean_Final[75,0]],[Mean_Final[76,0]],[Mean_Final[73,0]],[Mean_Final[72,0]]])                                                                                                                                    

Mean_Final_order2=np.array([[Mean_Final[68,0]],[Mean_Final[69,0]],[Mean_Final[58,0]],[Mean_Final[57,0]],[Mean_Final[56,0]],[Mean_Final[65,0]],[Mean_Final[43,0]],[Mean_Final[44,0]],[Mean_Final[45,0]],[Mean_Final[59,0]],
[Mean_Final[60,0]],[Mean_Final[61,0]],[Mean_Final[62,0]],[Mean_Final[63,0]],[Mean_Final[64,0]],[Mean_Final[55,0]],[Mean_Final[54,0]],[Mean_Final[53,0]],[Mean_Final[52,0]],[Mean_Final[51,0]],
[Mean_Final[46,0]],[Mean_Final[47,0]],[Mean_Final[48,0]],[Mean_Final[49,0]],[Mean_Final[50,0]],[Mean_Final[100,0]],[Mean_Final[99,0]],[Mean_Final[93,0]],[Mean_Final[98,0]],[Mean_Final[97,0]],                                                                                                                                          
[Mean_Final[96,0]],[Mean_Final[95,0]],[Mean_Final[94,0]],[Mean_Final[102,0]],[Mean_Final[101,0]],[Mean_Final[26,0]],[Mean_Final[25,0]],[Mean_Final[24,0]],[Mean_Final[23,0]],[Mean_Final[22,0]],
[Mean_Final[11,0]],[Mean_Final[10,0]],[Mean_Final[9,0]],[Mean_Final[8,0]],[Mean_Final[7,0]],[Mean_Final[2,0]],[Mean_Final[1,0]],[Mean_Final[0,0]]])                                                                                                                                    
Mean_Final_reorder = np.concatenate((0,Mean_Final_order1,0,Mean_Final_order2),axis=None)

Deviation_1_order1=np.array([[Deviation_1[6,0]],[Deviation_1[5,0]],[Deviation_1[4,0]],[Deviation_1[17,0]],[Deviation_1[16,0]],[Deviation_1[15,0]],[Deviation_1[14,0]],[Deviation_1[13,0]],[Deviation_1[31,0]],[Deviation_1[30,0]],
[Deviation_1[29,0]],[Deviation_1[28,0]],[Deviation_1[27,0]],[Deviation_1[103,0]],[Deviation_1[104,0]],[Deviation_1[34,0]],[Deviation_1[35,0]],[Deviation_1[36,0]],[Deviation_1[37,0]],[Deviation_1[38,0]],
[Deviation_1[39,0]],[Deviation_1[33,0]],[Deviation_1[32,0]],[Deviation_1[92,0]],[Deviation_1[91,0]],[Deviation_1[90,0]],[Deviation_1[89,0]],[Deviation_1[88,0]],                                                                                                                                          
[Deviation_1[83,0]],[Deviation_1[84,0]],[Deviation_1[85,0]],[Deviation_1[86,0]],[Deviation_1[87,0]],[Deviation_1[82,0]],[Deviation_1[81,0]],[Deviation_1[80,0]],[Deviation_1[79,0]],[Deviation_1[78,0]],
[Deviation_1[77,0]],[Deviation_1[40,0]],[Deviation_1[41,0]],[Deviation_1[42,0]],[Deviation_1[67,0]],[Deviation_1[74,0]],[Deviation_1[75,0]],[Deviation_1[76,0]],[Deviation_1[73,0]],[Deviation_1[72,0]]])                                                                                                                                    

Deviation_1_order2=np.array([[Deviation_1[68,0]],[Deviation_1[69,0]],[Deviation_1[58,0]],[Deviation_1[57,0]],[Deviation_1[56,0]],[Deviation_1[65,0]],[Deviation_1[43,0]],[Deviation_1[44,0]],[Deviation_1[45,0]],[Deviation_1[59,0]],
[Deviation_1[60,0]],[Deviation_1[61,0]],[Deviation_1[62,0]],[Deviation_1[63,0]],[Deviation_1[64,0]],[Deviation_1[55,0]],[Deviation_1[54,0]],[Deviation_1[53,0]],[Deviation_1[52,0]],[Deviation_1[51,0]],
[Deviation_1[46,0]],[Deviation_1[47,0]],[Deviation_1[48,0]],[Deviation_1[49,0]],[Deviation_1[50,0]],[Deviation_1[100,0]],[Deviation_1[99,0]],[Deviation_1[93,0]],[Deviation_1[98,0]],[Deviation_1[97,0]],                                                                                                                                          
[Deviation_1[96,0]],[Deviation_1[95,0]],[Deviation_1[94,0]],[Deviation_1[102,0]],[Deviation_1[101,0]],[Deviation_1[26,0]],[Deviation_1[25,0]],[Deviation_1[24,0]],[Deviation_1[23,0]],[Deviation_1[22,0]],
[Deviation_1[11,0]],[Deviation_1[10,0]],[Deviation_1[9,0]],[Deviation_1[8,0]],[Deviation_1[7,0]],[Deviation_1[2,0]],[Deviation_1[1,0]],[Deviation_1[0,0]]])                                                                                                                                    
Deviation_1_reorder = np.concatenate((0,Deviation_1_order1,0,Deviation_1_order2),axis=None)

Deviation_2_order1=np.array([[Deviation_2[6,0]],[Deviation_2[5,0]],[Deviation_2[4,0]],[Deviation_2[17,0]],[Deviation_2[16,0]],[Deviation_2[15,0]],[Deviation_2[14,0]],[Deviation_2[13,0]],[Deviation_2[31,0]],[Deviation_2[30,0]],
[Deviation_2[29,0]],[Deviation_2[28,0]],[Deviation_2[27,0]],[Deviation_2[103,0]],[Deviation_2[104,0]],[Deviation_2[34,0]],[Deviation_2[35,0]],[Deviation_2[36,0]],[Deviation_2[37,0]],[Deviation_2[38,0]],
[Deviation_2[39,0]],[Deviation_2[33,0]],[Deviation_2[32,0]],[Deviation_2[92,0]],[Deviation_2[91,0]],[Deviation_2[90,0]],[Deviation_2[89,0]],[Deviation_2[88,0]],                                                                                                                                          
[Deviation_2[83,0]],[Deviation_2[84,0]],[Deviation_2[85,0]],[Deviation_2[86,0]],[Deviation_2[87,0]],[Deviation_2[82,0]],[Deviation_2[81,0]],[Deviation_2[80,0]],[Deviation_2[79,0]],[Deviation_2[78,0]],
[Deviation_2[77,0]],[Deviation_2[40,0]],[Deviation_2[41,0]],[Deviation_2[42,0]],[Deviation_2[67,0]],[Deviation_2[74,0]],[Deviation_2[75,0]],[Deviation_2[76,0]],[Deviation_2[73,0]],[Deviation_2[72,0]]])                                                                                                                                    

Deviation_2_order2=np.array([[Deviation_2[68,0]],[Deviation_2[69,0]],[Deviation_2[58,0]],[Deviation_2[57,0]],[Deviation_2[56,0]],[Deviation_2[65,0]],[Deviation_2[43,0]],[Deviation_2[44,0]],[Deviation_2[45,0]],[Deviation_2[59,0]],
[Deviation_2[60,0]],[Deviation_2[61,0]],[Deviation_2[62,0]],[Deviation_2[63,0]],[Deviation_2[64,0]],[Deviation_2[55,0]],[Deviation_2[54,0]],[Deviation_2[53,0]],[Deviation_2[52,0]],[Deviation_2[51,0]],
[Deviation_2[46,0]],[Deviation_2[47,0]],[Deviation_2[48,0]],[Deviation_2[49,0]],[Deviation_2[50,0]],[Deviation_2[100,0]],[Deviation_2[99,0]],[Deviation_2[93,0]],[Deviation_2[98,0]],[Deviation_2[97,0]],                                                                                                                                          
[Deviation_2[96,0]],[Deviation_2[95,0]],[Deviation_2[94,0]],[Deviation_2[102,0]],[Deviation_2[101,0]],[Deviation_2[26,0]],[Deviation_2[25,0]],[Deviation_2[24,0]],[Deviation_2[23,0]],[Deviation_2[22,0]],
[Deviation_2[11,0]],[Deviation_2[10,0]],[Deviation_2[9,0]],[Deviation_2[8,0]],[Deviation_2[7,0]],[Deviation_2[2,0]],[Deviation_2[1,0]],[Deviation_2[0,0]]])                                                                                                                                    
Deviation_2_reorder = np.concatenate((0,Deviation_2_order1,0,Deviation_2_order2),axis=None)


#%% Faciograph
labels=(' ', 'Forehead', ' ', ' ', ' ', 'Eyebrow', ' ', ' ',
'External Canthus', ' ', 'Suborbital', ' ', 'Internal Canthus ', ' ',
'Eyelid ', ' ', ' ', 'Paranasal ', ' ', ' ', ' ', 'Nostril', ' ',
' ', ' ','Zygomatic', ' ', ' ', ' ', ' ', 'Cheek', ' ', ' ',
' ', ' ', 'Jawline', ' ', ' ', ' ', 'Labial commissure', 'Upper Lip',
' ', 'Lower Lip', ' ', 'D.A.O.*', ' ', 'Mentalis',' ', '0', ' ',
'Mentalis', ' ', 'D.A.O.*', ' ', 'Lower lip', ' ', 'Upper lip', 'Labial commissure', ' ', ' ',
  ' ', 'Jawline', ' ', ' ', ' ', ' ',
'Cheek', ' ', ' ', ' ', ' ', 'Zygomatic', ' ', ' ', ' ',
'Nostril', ' ', ' ', ' ', 'Paranasal', ' ', ' ', 'Eyelid', ' ',
'Internal Canthus', ' ', 'Suborbital', ' ', 'External Canthus', ' ', ' ',
'Eyebrow',' ', ' ', ' ', 'Forehead',' ','0')


label_loc = np.linspace(start=0, stop=2*np.pi, num=len(Mean_Final_reorder))

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, polar=True)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.plot(label_loc,Mean_Final_reorder,color='grey')
plt.fill_between(label_loc,Deviation_1_reorder,Deviation_2_reorder, color='grey', alpha=.15)

lines, labels = plt.thetagrids(np.degrees(label_loc), labels=labels)
plt.show()
