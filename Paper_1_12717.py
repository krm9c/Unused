"""
Created on Fri Apr 22 12:36:08 2016
@author: Krishnan Raghavan
"""
######################################################################################
# Pre Existing Libraries
from math import *
import os,sys
import random
import time
import itertools
import math
import matplotlib.pyplot                 as plt
######################################################################################
import numpy                             as np
from   collections                   import Counter
from   sklearn.decomposition         import PCA
from   matplotlib.colors             import ListedColormap
from   sklearn.datasets              import make_moons, make_circles, make_classification
from   sklearn.neighbors             import KNeighborsClassifier
from   sklearn.svm                   import SVC
from   sklearn.tree                  import DecisionTreeClassifier
from   sklearn.ensemble              import RandomForestClassifier, AdaBoostClassifier
from   sklearn.naive_bayes           import GaussianNB
from   sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from   sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from   sklearn.random_projection 	 import SparseRandomProjection
from   sklearn 						 import manifold
from   sklearn                       import cluster, datasets
from   sklearn.neighbors             import kneighbors_graph
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from   sklearn.preprocessing         import StandardScaler
#######################################################################################
# Libraries created by us
sys.path.append('/Users/krishnanraghavan/Dropbox/Work/Research/Common_Libraries')
sys.path.append('/Users/krishnanraghavan/Dropbox/Work/Research/Paper_1_codes')
sys.path.append('/Users/krishnanraghavan/Dropbox/Work/Research/Paper_2_codes')
from Library_Paper_two       import *
from Library_Paper_one       import import_data, traditional_MTS
from Data_import             import *


############################################################################
# Classification Methodology (Fischer Discriminant Analysis)
############################################################################
def fisher(DM, M_1, M_2, M_3, M_4):
    # First let us decide on the prior probabilities
    pi1=1/float(4);
    pi2=1/float(4);
    pi3=1/float(4);
    pi4=1/float(4);
    for i in range(0,2):
        Prob=[];
        Prob_label= [];
        # print "Priors ", "(", pi1, ",", pi2, ",", pi3, ",", pi4, ")"
        for element in DM:
            P_1 = ((1+(1/float(M_1)))*pi1*exp(-0.5*element[0])) / (((1+(1/float(M_1)))*pi1*exp(-0.5*element[0])) +((1+(1/float(M_2)))*pi2*exp(-0.5*element[1]))+((1+(1/float(M_3)))*pi3*exp(-0.5*element[2]))+((1+(1/float(M_4)))*pi4*exp(-0.5*element[3])));
            P_2 = ((1+(1/float(M_2)))*pi2*exp(-0.5*element[1])) / (((1+(1/float(M_1)))*pi1*exp(-0.5*element[0])) +((1+(1/float(M_2)))*pi2*exp(-0.5*element[1]))+((1+(1/float(M_3)))*pi3*exp(-0.5*element[2]))+((1+(1/float(M_4)))*pi4*exp(-0.5*element[3])));
            P_3 = ((1+(1/float(M_3)))*pi3*exp(-0.5*element[2])) / (((1+(1/float(M_1)))*pi1*exp(-0.5*element[0])) +((1+(1/float(M_2)))*pi2*exp(-0.5*element[1]))+((1+(1/float(M_3)))*pi3*exp(-0.5*element[2]))+((1+(1/float(M_4)))*pi4*exp(-0.5*element[3])));
            P_4 = ((1+(1/float(M_4)))*pi4*exp(-0.5*element[3])) / (((1+(1/float(M_1)))*pi1*exp(-0.5*element[0])) +((1+(1/float(M_2)))*pi2*exp(-0.5*element[1]))+((1+(1/float(M_3)))*pi3*exp(-0.5*element[2]))+((1+(1/float(M_4)))*pi4*exp(-0.5*element[3])));
            Prob. append(P_1);
            Prob. append(P_2);
            Prob. append(P_3);
            Prob. append(P_4);
            Prob_label.append(np.argmax(Prob));
            Prob =[]
        Prob_label=np.array(Prob_label)
        c = [0,0,0,0];
        c[0]=list(Prob_label).count(0)
        c[1]=list(Prob_label).count(1)
        c[2]=list(Prob_label).count(2)
        c[3]=list(Prob_label).count(3)
        pi1=((c[0])/float(c[0] + c[1] + c[2] +c[3]))
        pi2=((c[1])/float(c[0] + c[1] + c[2] +c[3]))
        pi3=((c[2])/float(c[0] + c[1] + c[2] +c[3]))
        pi4=((c[3])/float(c[0] + c[1] + c[2] +c[3]))

        # print "Fisher--Iteration", i
    return np.resize(np.array(Prob_label),(len(Prob_label),))
############################################################################
def Classification_Faults_Bearing(N, T):
    p_value_0=[]
    p_value_1=[]
    p_value_2=[]
    p_value_3=[]
    for ola in xrange(4):
        print "ola --", ola

        Test = np.array(T[ola].copy())
        print "Shape -- ", Test.shape

        D_M= np.zeros((Test.shape[0],4))
        print "D_M shape", D_M.shape

        D_M[:,0]= traditional_MTS(N[0], Test, 0).reshape((Test.shape[0],))
        D_M[:,1]= traditional_MTS(N[1], Test, 0).reshape((Test.shape[0],))
        D_M[:,2]= traditional_MTS(N[2], Test, 0).reshape((Test.shape[0],))
        D_M[:,3]= traditional_MTS(N[3], Test, 0).reshape((Test.shape[0],))

        Label = fisher(D_M, N[0].shape[0], N[1].shape[0], N[2].shape[0], N[3].shape[0])

        c = [0,0,0,0];
        c[0]=list(Label).count(0)
        c[1]=list(Label).count(1)
        c[2]=list(Label).count(2)
        c[3]=list(Label).count(3)
        print "Counted detection", "(",c[0],",", c[1], ",",c[2],",", c[3],")"
        if ola ==0:
            p_value_0.append((c[1] + c[2] +c[3])/float(c[0] + c[1] + c[2] +c[3]))
        if ola ==1:
            p_value_1.append((c[0] + c[2] +c[3])/float(c[0] + c[1] + c[2] +c[3]))
        if ola ==2:
            p_value_2.append((c[1] + c[0] +c[3])/float(c[0] + c[1] + c[2] +c[3]))
        if ola ==3:
            p_value_3.append((c[1] + c[2] +c[0])/float(c[0] + c[1] + c[2] +c[3]))
    P=[]
    P.append(sum(p_value_0)/float(len(p_value_0)))
    P.append(sum(p_value_1)/float(len(p_value_1)))
    P.append(sum(p_value_2)/float(len(p_value_2)))
    P.append(sum(p_value_3)/float(len(p_value_3)))
    return P
############################################################################
#Classify Data
def Classification(Temp_IR, Temp_OR, Temp_NL, Temp_N, TIR, TOR, TNL, TN):
    P_value =[]
    sample_size_T=2000;
    sample_size_N=10000;
    for p in xrange(1):
        T = []
        N = []
        # Create the Test Sample
        rand_1 = [random.randint(0, (TN.shape[0]-1)) for i in xrange(sample_size_T)]
        rand_2 = [random.randint(0, (TIR.shape[0]-1)) for i in xrange(sample_size_T)]
        rand_3 = [random.randint(0, (TNL.shape[0]-1)) for i in xrange(sample_size_T)]
        rand_4 = [random.randint(0, (TOR.shape[0]-1)) for i in xrange(sample_size_T)]
        Test_N  = TN[rand_1,:];
        Test_IR = TIR[rand_2,:];
        Test_OR = TOR[rand_4,:];
        Test_NL = TNL[rand_3,:];
        # temp = np.concatenate((Test_N, Test_IR, Test_OR, Test_NL))
        # T.append(temp)
        # T.append(temp)
        # T.append(temp)
        # T.append(temp)
        T.append(Test_N)
        T.append(Test_IR)
        T.append(Test_OR)
        T.append(Test_NL)
        T = np.array(T)
        # Create the Normal Data
        rand_1 = [random.randint(0, (Temp_N.shape[0]-1)) for i in xrange(sample_size_N)]
        rand_2 = [random.randint(0,(Temp_IR.shape[0]-1)) for i in xrange(sample_size_N)]
        rand_3 = [random.randint(0,(Temp_NL.shape[0]-1)) for i in xrange(sample_size_N)]
        rand_4 = [random.randint(0,(Temp_OR.shape[0]-1)) for i in xrange(sample_size_N)]
        Test_N  = Temp_N[rand_1,:];
        N.append(Test_N)
        Test_IR = Temp_IR[rand_2,:];
        N.append(Test_IR)
        Test_OR = Temp_OR[rand_4,:];
        N.append(Test_OR)
        Test_NL = Temp_NL[rand_3,:];
        N.append(Test_NL)
        N = np.array(N)
        print T.shape, N.shape
        # Classify the Data
        print "Iteration -- classify", p
        P_value.append(Classification_Faults_Bearing(N, T))
    P = np.array(P_value)
    print "The p value for the N is", (sum(P[:,0])/len(P[:,0]))
    print "The p value for the IR is", (sum(P[:,1])/len(P[:,1]))
    print "The p value for the OR is", (sum(P[:,2])/len(P[:,2]))
    print "The p value for the NL is" , (sum(P[:,3])/len(P[:,3]))

############################################################################
# Type One and Type Two Errors
############################################################################
def TypeOneError(Ref, Data):
	p_values = []
	detect = traditional_MTS(Ref, np.array(Data), 0)
	# print "The mean of the MD values are", detect.mean()
	p =4.615
	index1=list([i for i,v in enumerate(detect) if v  < p]);
	p =6.42
	index2=list([i for i,v in enumerate(detect) if v  < p]);
	p =9.26
	index3=list([i for i,v in enumerate(detect) if v  < p]);
	p_values. append((len(index1))/float(len(Data)))
	p_values. append((len(index2))/float(len(Data)))
	p_values. append((len(index3))/float(len(Data)))
	return p_values
#######################################################################################
def TypeTwoError(Ref, Data):
    detect = traditional_MTS(Ref, np.array(Data), 0)
    p =4.615
    index1=list([i for i,v in enumerate(detect) if v  < p]);
    p =6.42
    index2=list([i for i,v in enumerate(detect) if v  < p]);
    p =9.26
    index3=list([i for i,v in enumerate(detect) if v  < p]);
    p_values   = []
    pow_values = []
    p_values. append((len(index1))/float(len(Data)))
    p_values. append((len(index2))/float(len(Data)))
    p_values. append((len(index3))/float(len(Data)))
    pow_values. append(1-((len(index1))/float(len(Data))))
    pow_values. append(1-((len(index2))/float(len(Data))))
    pow_values. append(1-((len(index3))/float(len(Data))))
    return p_values, pow_values
#######################################################################################
def Error_T_1(Ref, T):
	sample_size = 1000
	Data = np.zeros((0,sample_size))
	P=[]
	Power = []
	# Get_Num(Ref, Data)
	rand= [0 for i in xrange(sample_size)]
	for i in range(0,100):
		for i in range(0,sample_size):
				rand[i]= random.randint(0,(T.shape[0]-1))
		Data = T[rand];
		p = TypeOneError(Ref, Data);
		P.append(p)
	P = np.array(P)
	print "Type 1 Error"
	print "p = 4.615; 0.1", sum(P[0,:])/len(P[0,:])
	print "p = 6.028; 0.05",sum(P[1,:])/len(P[1,:])
	print "p = 9.26; 0.01", sum(P[2,:])/len(P[2,:])
########################################################################
def Error_T_2(Ref, T):
	sample_size = 1000
	Data = np.zeros((0,sample_size))
	P=[]
	Power = []
	rand= [0 for i in xrange(sample_size)]
	for i in range(0,100):
		for i in range(0,sample_size):
				rand[i]= random.randint(0,(T.shape[0]-1))
		Data = T[rand];
		p, pow_1 = TypeTwoError(Ref, Data);
		P.append(p)
		Power.append(pow_1)
    # Display stuff..
	P = np.array(P)
	Power = np.array(Power)
	print "Type 2 Error"
	print "p = 4.615; 0.1", sum(P[0,:])/len(P[0,:])
	print "p = 6.028; 0.05",sum(P[1,:])/len(P[1,:])
	print "p = 9.26; 0.01", sum(P[2,:])/len(P[2,:])
	print "Power of the Test"
	print "p = 4.615; 0.1", sum(Power[0,:])/len(Power[0,:])
	print "p = 6.028; 0.05",sum(Power[1,:])/len(Power[1,:])
	print "p = 9.26; 0.01", sum(Power[2,:])/len(Power[2,:])
########################################################################
if __name__ == "__main__":
    # That, Yhat = DataImport(1)
    T_N, T_NL, T_IR, T_OR = DataImport_Rolling_ones(path, 1)

    from sklearn.cross_validation import train_test_split
    T_N, TN  = train_test_split(T_N,  test_size=0.20, random_state=42)
    T_NL, TNL= train_test_split(T_NL, test_size=0.20, random_state=42)
    T_IR, TIR= train_test_split(T_IR, test_size=0.20, random_state=42)
    T_OR, TOR= train_test_split(T_OR, test_size=0.20, random_state=42)
    g_size = 3
    # Train for parameters
    N, Tree = initialize_calculation(T = None, Data = T_N, gsize = g_size, par_train = 0)
    # Reduce the dimensions
    NL, Tree   = initialize_calculation(T = Tree, Data = T_NL, gsize = g_size, par_train = 1)
    IR, Tree   = initialize_calculation(T = Tree, Data = T_IR, gsize = g_size, par_train = 1)
    OR, Tree   = initialize_calculation(T = Tree, Data = T_OR, gsize = g_size, par_train = 1)
    N, Tree    = initialize_calculation(T = Tree,  Data = T_N, gsize = g_size, par_train = 1)
    TNL, Tree  = initialize_calculation(T = Tree, Data = TNL, gsize = g_size, par_train = 1)
    TIR, Tree  = initialize_calculation(T = Tree, Data = TIR, gsize = g_size, par_train = 1)
    TOR, Tree  = initialize_calculation(T = Tree, Data = TOR, gsize = g_size, par_train = 1)
    TN, Tree   = initialize_calculation(T = Tree,  Data = TN, gsize = g_size, par_train = 1)
    # The function for Classification
    Classification(IR, OR, NL, N, TIR, TOR, TNL, TN)
    # Type One error
    # plt.figure(1)
    # MD = traditional_MTS(T[0:1000,:], T[1000:len(T),:], 0)
    # plt.plot(MD)
    # print MD.mean()
    # Error_T_1(T[0:1000,:], T[1000:len(T),:])

    ## Type Two error
    # plt.figure(2)
    # MD = traditional_MTS(T[0:1000,:], T[1000:len(T),:], 0)
    # plot(MD)
    # Error_T_2(T[0:1000,:], T[0:1000,:])
    # plt.show()
