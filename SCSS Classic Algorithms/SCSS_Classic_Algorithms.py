# -*- coding: utf-8 -*-
"""
Created on Mon April 11, 2022
@author: ZhangZhou J.

Classic algorithms for SCSS based on Rubie et al. (2016 Science); Ding et al.(2018 GCA), Blanchard et al. (2021 American Mineralogist)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from math import exp

# below are the function to convert weight percent into atomic ratios as they are used in the calculation of SCSS following Ding et al. (2018 GCA) and Blanchard et al. (2021)

def oxide_wt_at(SiO2,TiO2,Al2O3,MgO,FeO,CaO,Na2O,K2O,Cr2O3):  #mole fraction of SiO2, AlO1.5.. or cation fraction 
    total_oxide_atom=SiO2/60+TiO2/89.8+Al2O3/51+MgO/40+FeO/73.8+CaO/56+Na2O/31+K2O/47+Cr2O3/76
    X_SiO2=SiO2/60/total_oxide_atom
    X_TiO2=TiO2/89.8/total_oxide_atom
    X_Al2O3=Al2O3/51/total_oxide_atom
    X_MgO=MgO/40/total_oxide_atom
    X_FeO=FeO/73.8/total_oxide_atom
    X_CaO=CaO/56/total_oxide_atom
    X_Na2O=Na2O/31/total_oxide_atom
    X_K2O=K2O/47/total_oxide_atom
    X_Cr2O3=Cr2O3/76/total_oxide_atom
    return (X_SiO2,X_TiO2,X_Al2O3,X_MgO,X_FeO,X_CaO,X_Na2O,X_K2O,X_Cr2O3)

def sulfide_wt_at(Fe,Ni,S,O): 
    total_sulfide_atom=Fe/55.8+Ni/58.5+S/32+O/16
    X_Fe=Fe/55.8/total_sulfide_atom
    X_Ni=Ni/58.5/total_sulfide_atom
    X_S=S/32/total_sulfide_atom
    X_O=O/16/total_sulfide_atom
    X_FeS=X_Fe/(X_Fe+X_Ni+X_O)
    return (X_Fe,X_Ni,X_S,X_O,X_FeS)

def Mole_Sum(X_SiO2,A_SiO2,X_TiO2,A_TiO2,X_Al2O3,A_Al2O3,X_MgO,A_MgO,X_FeO,A_FeO,X_CaO,A_CaO,X_Na2O,A_Na2O,X_K2O,A_K2O,X_H2O,A_H2O,A_SiFe):
    XmAm=X_SiO2*A_SiO2+X_TiO2*A_TiO2+X_Al2O3*A_Al2O3+X_MgO*A_MgO+X_FeO*A_FeO+X_CaO*A_CaO+X_Na2O*A_Na2O+X_K2O*A_K2O+X_H2O*A_H2O+X_SiO2*X_FeO*A_SiFe
    return (XmAm)

def CiXm_sum(C_Ti,X_TiO2,C_Ca,X_CaO,C_Si,X_SiO2,C_Al,X_Al2O3,C_Fe,X_FeO):
    CiXm=C_Ti*X_TiO2+C_Ca*X_CaO+C_Si*X_SiO2+C_Al*X_Al2O3+C_Fe*X_FeO
    return (CiXm)
    
np.set_printoptions(suppress=True) #不使用科学计数法

# load an excel file (here using Steenstra et al. 2022 data as an example)
# with input parameters of P, T, compositions of silicate and sulfide
 
File_A=pd.read_excel('Steenstra example.xlsx') 
File_A=np.array(File_A)
n=len(File_A) # the number of rows in the table

# input parameters
P=np.zeros((n,1))
T=np.zeros((n,1))
SiO2=np.zeros((n,1))
TiO2=np.zeros((n,1))
Al2O3=np.zeros((n,1))
FeO=np.zeros((n,1))
MgO=np.zeros((n,1))
CaO=np.zeros((n,1))
Na2O=np.zeros((n,1))
K2O=np.zeros((n,1))
Cr2O3=np.zeros((n,1))
H2O=np.zeros((n,1))
Fe=np.zeros((n,1))
Ni=np.zeros((n,1))
S=np.zeros((n,1))
O=np.zeros((n,1))

X_SiO2=np.zeros((n,1))
X_TiO2=np.zeros((n,1))
X_Al2O3=np.zeros((n,1))
X_MgO=np.zeros((n,1))
X_FeO=np.zeros((n,1))
X_CaO=np.zeros((n,1))
X_Na2O=np.zeros((n,1))
X_K2O=np.zeros((n,1))
X_Cr2O3=np.zeros((n,1))
X_H2O=np.zeros((n,1))
X_Fe=np.zeros((n,1))
X_Ni=np.zeros((n,1))
X_S=np.zeros((n,1))
X_O=np.zeros((n,1))
X_FeS=np.zeros((n,1))


XmAm_Blanchard=np.zeros((n,1))  # calculated coefficients in Blanchard (2021) model 2
CiXm_Ding=np.zeros((n,1))

SCSS_Rubie=np.zeros((n,1))
SCSS_Ding=np.zeros((n,1))
SCSS_Blanchard=np.zeros((n,1))
               
S_Rubie=np.zeros((n,1))  # Sulfur stored equivalent to BSE at that pressure
S_Smythe=np.zeros((n,1)) 
S_Ding=np.zeros((n,1))
S_Blanchard=np.zeros((n,1)) 
S_XGBoost=np.zeros((n,1)) 
S_RandomForest=np.zeros((n,1)) 


def SCSS_Algorithm_Rubie(pressure, temperature):
    SCSS_value=exp(14.2-11032/temperature-379*pressure/temperature)
    return (SCSS_value)

for i in range(0,n,1): # assign weight wt.% values to each position                                                        

  P[i]=File_A[i,1]
  T[i]=File_A[i,2]
  SiO2[i]=File_A[i,3]
  TiO2[i]=File_A[i,4]
  Al2O3[i]=File_A[i,5]
  FeO[i]=File_A[i,6]
  MgO[i]=File_A[i,7]
  CaO[i]=File_A[i,8]
  Cr2O3[i]=File_A[i,9]  
  Na2O[i]=File_A[i,10]
  K2O[i]=File_A[i,11]
  H2O[i]=File_A[i,12]                      
  # oxide is silciate melt, elemental is sulfide     
  Fe[i]=File_A[i,13]
  Ni[i]=File_A[i,14]
  S[i]=File_A[i,15]
  O[i]=File_A[i,16]                
                                             
  # below is the Blanchard et al. 2022 et al. American Mineralogist model for SCSS, Coefficients Model 2, Eqn (12)
  a_blanchard=7.95
  b_blanchard=18159
  c_blanchard=-190
  # capitial A is the fitting coefficient from Blanchard et al. (2021)
  A_blanchard_SiO2=-32677
  A_blanchard_TiO2=-15014
  A_blanchard_Al2O3=-23071
  A_blanchard_MgO=-18258
  A_blanchard_FeO=-41706
  A_blanchard_CaO=-14668
  A_blanchard_Na2O=-19529
  A_blanchard_K2O=-34641
  A_blanchard_H2O=-22677
  A_blanchard_SiFe=120662
                                                
  # below is the Smythe 2017 suggesed in Blanchard et al. 2022 et al. American Mineralogist
  a_smythe=8.03
  b_smythe=-14683
  c_smythe=-265.80
  # capitial A is the fitting coefficient from Blanchard
  A_smythe_SiO2=0
  A_smythe_TiO2=16430
  A_smythe_Al2O3=9295
  A_smythe_MgO=13767
  A_smythe_FeO=-7080
  A_smythe_CaO=19893
  A_smythe_Na2O=14197
  A_smythe_K2O=0
  A_smythe_H2O=-17495
  A_smythe_SiFe=117827  
  
  # below is the coefficients of Ding et al. 2018 GCA euqation for SCSS
  A_Ding=12.10023817
  B_Ding=-4951.220517
  C_Ti=4.02527185
  C_Ca=4.173632609
  C_Si=-3.643865073
  C_Al=-3.936000202
  C_Fe=5.574892678
  D_Ding=-40.67763841
  E_Ding=-273.4844764

  (X_SiO2[i],X_TiO2[i],X_Al2O3[i],X_MgO[i],X_FeO[i],X_CaO[i],X_Na2O[i],X_K2O[i],X_Cr2O3[i])= oxide_wt_at(SiO2[i],TiO2[i],Al2O3[i],MgO[i],FeO[i],CaO[i],Na2O[i],K2O[i],Cr2O3[i])
  (X_Fe[i],X_Ni[i],X_S[i],X_O[i],X_FeS[i])=sulfide_wt_at(Fe[i],Ni[i],S[i],O[i])
    
  XmAm_Blanchard[i]=Mole_Sum(X_SiO2[i],A_blanchard_SiO2,X_TiO2[i],A_blanchard_TiO2,X_Al2O3[i],A_blanchard_Al2O3,X_MgO[i],A_blanchard_MgO,X_FeO[i],A_blanchard_FeO,X_CaO[i],A_blanchard_CaO,X_Na2O[i],A_blanchard_Na2O,X_K2O[i],A_blanchard_K2O,X_H2O[i],A_blanchard_H2O,A_blanchard_SiFe)
  CiXm_Ding[i]=CiXm_sum(C_Ti,X_TiO2[i],C_Ca,X_CaO[i],C_Si,X_SiO2[i],C_Al,X_Al2O3[i],C_Fe,X_FeO[i])
                 
  SCSS_Rubie[i]=SCSS_Algorithm_Rubie(P[i],T[i])
  SCSS_Ding[i]=np.exp(A_Ding+B_Ding/T[i]+CiXm_Ding[i]+D_Ding*X_FeO[i]*X_TiO2[i]+E_Ding*P[i]/T[i])
  SCSS_Blanchard[i]=np.exp(a_blanchard+b_blanchard/T[i]+c_blanchard*P[i]/T[i]+XmAm_Blanchard[i]/T[i]+math.log(X_FeS[i])-math.log(X_FeO[i]))
 
#df=pd.DataFrame(SCSS_Blanchard)
#writer=pd.ExcelWriter('Earth result Blanchard.xlsx',engine='xlsxwriter',options={'string_to_urls':False})

# df=pd.DataFrame(SCSS_RandomForest)
# writer=pd.ExcelWriter('Earth result Random Forest.xlsx',engine='xlsxwriter',options={'string_to_urls':False})

# df.to_excel(writer, index=False)
# writer.save()

df=pd.DataFrame(SCSS_Blanchard)   # if needs the results of Rubie et al. (2016) algorithm, repalce SCSS_Rubie to SCSS_Blanchard
writer=pd.ExcelWriter('Steenstra Blanchard prediction.xlsx',engine='xlsxwriter',options={'string_to_urls':False})

df.to_excel(writer, index=False)
writer.save()