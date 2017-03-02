import numpy as np
import matplotlib.pyplot as plt
import sys
from copulalib.copulalib import Copula
import math
import random
import time
import collections
#######################################################################################
sys.path.append('C:\Users\krm9c\Desktop\Research\Common_Libraries')
path= "E:\Research_Krishnan\Data\Data_case_study_1"
from Library_Paper_one import *



def plot_copula(X1,Y1,X2,Y2,filename):
    import matplotlib.pyplot as plt
    import seaborn as sns
    labels=['Data Relationship_Ref', 'Data Relationship_Test', 'Data Relationship', 'Data Relationship'];
    colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c"];
    markers = ['x','o','v','^','<'];
    # Start off by setting  the details of the plots
    sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=0.4,  rc=None)
    fig, ax = plt.subplots(1)
    fig.set_size_inches(8, 4)
    j=0;
    ax.scatter(X1,Y1,label=labels[j],color=colors[j],marker=markers[j]);
    plt.hold(True)
    j=1
    ax.scatter(X2,Y2,label=labels[j],color=colors[j],marker=markers[j]);
    plt.yticks(fontsize = 20)
    fig.tight_layout(pad=0.05)
    plt.grid()
    ax.legend( loc='upperright',markerscale=1.5,ncol=2,fancybox='True',shadow='True',frameon='True',fontsize=22)
    plt.xticks(fontsize = 20)
    # Saving the plots using a filename.
    filename=filename+'.png'
    plt.savefig(filename, format='png', dpi=200)
    plt.show()

def plot_copula_time(data,gamma,label,filename,colors):
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Setup the formatting for the plots
    sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=0.4,  rc=None)

    # Create a figure in python plt
    fig, ax = plt.subplots(1)
    fig.set_size_inches(7,2.5)

    # The plotting is done by this line
    ax.plot(gamma,data,label=label,linewidth=2.5,color=colors)

    # Defining legend and different formatting for the plots
    ax.legend(loc='best', ncol=2,fancybox='True',shadow='True',frameon='True',fontsize=15)
    #plt.xlim([0,limB])
    #plt.ylim([0,limA])
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.ylabel('NGDM-HDR', fontsize = 15)
    plt.xlabel('Euclidean Distance',fontsize = 15)
    fig.tight_layout(pad=0.05)
    plt.grid()

    # Save the file with a name
    filename=filename+'.png'
    plt.savefig(filename, format='png', dpi=300)
    plt.close('all');

def Weibull():
    N_points=100;
    N_Features=10;
    size_init_data=100;
    Data_array_Ref=np.zeros((N_points,N_Features));
    Data_array_Test=np.zeros((N_points,N_Features));
    # Let us create the numpy array
    for i in range(0,N_Features):
        # Generate random (normal distributed) numbers::
        x = np.random.weibull(10,size=size_init_data);
        y = np.power(x,2)+np.random.weibull(10,size=size_init_data);
        # Make the instance of Copula class with x, y and clayton family::
        foo = Copula(x, y, family='frank');
        X1, Y1 = foo.generate_xy(N_points);
        #print X1.shape
        X2=X1;
        Y2=Y1;
        #plot_copula(X1,Y1,X2,Y2,'Copula_parabola_T')
        Data_array_Ref[:,i]=X1.copy();
        Data_array_Ref[:,i]=Y1.copy();
        Data_array_Test[:,i]=X2.copy();
        Data_array_Test[:,i]=Y2.copy();
        i=i+1;
    np.savetxt("foo_Ref.csv", Data_array_Ref, delimiter=",");
    np.savetxt("foo_Test.csv", Data_array_Test, delimiter=",");

    ###########################################################
    from numpy import genfromtxt
    dev= np.random.rand(100,1)*1000;
    dev=np.sort(dev, axis=None);
    Data_array_Ref= genfromtxt('foo_Ref.csv', delimiter=',');
    Data_array_Test= genfromtxt('foo_Test.csv', delimiter=',');
    Data_array_Test=Data_array_Test+40;
    # Calculate Distance between Class 2 and Class 1
    row_ref= Data_array_Ref.shape[0];
    column_ref=Data_array_Ref.shape[1];
    row_test=Data_array_Test.shape[0];
    column_test=Data_array_Test.shape[1];
    for gam in [0.000001,0.00001,0.0001,0.1]:
        Res=[];
        for element in dev:

            # Declare the variables
            print("--- %s True Distance ---" % (element));
            Data_array_Ref= genfromtxt('foo_Ref.csv', delimiter=',')  ;
            Data_array_Test= genfromtxt('foo_Test.csv', delimiter=',');
            Data_array_Test=Data_array_Test+element;
            mean_normal=[]
            std_normal=[]

            # Declare the Matrices
            Normalized_matrix_test_Data_class_1 = np.zeros((row_ref,column_ref));
            Normalized_matrix_test_Data_class_2 = np.zeros((row_test,column_test));

            # First Calculate the mean and the standard deviation\
            for i in range(0 , column_ref):
                a=np.array(Data_array_Ref[i]);
                mean_normal.append(a.mean());
                std_normal.append(a.std());

            #Calcualte Normalized matrix
            for j in range(0 , column_ref):
                for i in range(0 , row_ref):
                    Normalized_matrix_test_Data_class_1[i,j]=(Data_array_Ref[i,j]-mean_normal[j])/std_normal[j];
            for j in range(0 , column_test):
                for i in range(0 , row_test):
                    Normalized_matrix_test_Data_class_2[i,j]=(Data_array_Test[i,j]-mean_normal[j])/std_normal[j];

            numberFlag=1;
            CurrentFile=[]
            start_time=time.time()
            MD,CurrentFile=Initialization_kernel(Normalized_matrix_test_Data_class_1,  Normalized_matrix_test_Data_class_2, 'Simulated_Data',1,2, numberFlag,CurrentFile, gam);
            print("--- %s seconds ---" % (time.time() - start_time));
            Res.append(MD.mean());
            print "The mean of Class 2 from Ref is", MD.mean();
        print "The resulting distances are", Res;
        print("--- %s seconds ---" % (time.time() - start_time));
        colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c"];
        Data_plot=np.zeros((len(Res), 2))
        Data_plot[:,0]=np.array(Res).copy();
        Data_plot[:,1]=np.array(dev).copy();
        np.savetxt("foo_data_plot_Weibull" + str(gam) +".csv", Data_plot, delimiter=",");
        plot_copula_time(Res,dev,str(gam),'Weibull1'+str(gam),colors[0]);

###############################################################################################################
def Normal():
    deviation=10
    N_points=10000;
    N_Features=200;
    size_init_data=100;
    Data_array_Ref=np.zeros((N_points,N_Features));
    Data_array_Test=np.zeros((N_points,N_Features));
    # Let us create the numpy array
    for i in range(0,N_Features):
        # Generate random (normal distributed) numbers::
        x = np.random.normal(size=size_init_data);
        y = x+np.random.normal(size=size_init_data);

        # Make the instance of Copula class with x, y and clayton family::
        foo = Copula(x, y, family='clayton');
        X1, Y1 = foo.generate_xy(N_points);

        #print X1.shape
        #print Y1.shape
        print "The iteration going on is", i;
        X2=X1+10;
        Y2=Y1+10;
        # plot_copula(X1,Y1,X2,Y2,'Copula_parabola_T')
        Data_array_Ref[:,i]=X1.copy();
        Data_array_Ref[:,i]=Y1.copy();
        Data_array_Test[:,i]=X2.copy();
        Data_array_Test[:,i]=Y2.copy();
        i=i+1;

    np.savetxt("foo_Ref.csv", Data_array_Ref, delimiter=",");
    np.savetxt("foo_Test.csv", Data_array_Test, delimiter=",");


    # from numpy import genfromtxt
    # dev= np.random.rand(100,1)*1000;
    # dev=np.sort(dev, axis=None);
    # Data_array_Ref= genfromtxt('foo_Ref.csv', delimiter=',')
    # Data_array_Test= genfromtxt('foo_Test.csv', delimiter=',');
    # Data_array_Test=Data_array_Test+40;
    # # Calculate Distance between Class 2 and Class 1:
    # row_ref= Data_array_Ref.shape[0];
    # column_ref=Data_array_Ref.shape[1];
    # row_test=Data_array_Test.shape[0];
    # column_test=Data_array_Test.shape[1];
    #
    # for gam in [0.000001,0.00001,0.0001,0.1]:
    #     Res=[];
    #     for element in dev:
    #
    #         # Declare the variables
    #         print("--- %s True Distance ---" % (element));
    #         Data_array_Ref= genfromtxt('foo_Ref.csv', delimiter=',')
    #         Data_array_Test= genfromtxt('foo_Test.csv', delimiter=',');
    #         Data_array_Test=Data_array_Test+element;
    #         mean_normal=[]
    #         std_normal=[]
    #
    #         # Declare the Matrices
    #         Normalized_matrix_test_Data_class_1 = np.zeros((row_ref,column_ref));
    #         Normalized_matrix_test_Data_class_2 = np.zeros((row_test,column_test));
    #
    #         # First Calculate the mean and the standard deviation\
    #         for i in range(0 , column_ref):
    #             a=np.array(Data_array_Ref[i]);
    #             mean_normal.append(a.mean());
    #             std_normal.append(a.std());
    #
    #         #Calcualte Normalized matrix
    #         for j in range(0 , column_ref):
    #             for i in range(0 , row_ref):
    #                 Normalized_matrix_test_Data_class_1[i,j]=(Data_array_Ref[i,j]-mean_normal[j])/std_normal[j];
    #         for j in range(0 , column_test):
    #             for i in range(0 , row_test):
    #                 Normalized_matrix_test_Data_class_2[i,j]=(Data_array_Test[i,j]-mean_normal[j])/std_normal[j];
    #
    #         numberFlag=1;
    #         CurrentFile=[]
    #         start_time=time.time()
    #         MD,CurrentFile=Initialization_kernel(Normalized_matrix_test_Data_class_1,  Normalized_matrix_test_Data_class_2, 'Simulated_Data',1,2, numberFlag,CurrentFile, gam);
    #         print("--- %s seconds ---" % (time.time() - start_time));
    #         Res.append(MD.mean());
    #         print "The mean of Class 2 from Ref is", MD.mean();
    #     colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c"];
    #     plot_copula_time(Res,dev,str(gam),'Normal1'+str(gam) ,colors[1]);
    #     print "The resulting distances are", Res;
    #     Data_plot=np.zeros((len(Res), 2))
    #     Data_plot[:,0]=np.array(Res).copy();
    #     Data_plot[:,1]=np.array(dev).copy();
    #     np.savetxt("foo_data_plot_Normal" + str(gam) +".csv", Data_plot, delimiter=",");
    #     print("--- %s seconds ---" % (time.time() - start_time));


#################################################################################
def Exponential():
    deviation=30
    N_points=100;
    N_Features=10;
    size_init_data=100;
    Data_array_Ref=np.zeros((N_points,N_Features));
    Data_array_Test=np.zeros((N_points,N_Features));
    # Let us create the numpy array
    for i in range(0,N_Features):
        # Generate random (normal distributed) numbers::
        x = np.random.exponential(5,size=size_init_data);
        y = np.power(x,3)+np.random.exponential(5,size=size_init_data);
        # Make the instance of Copula class with x, y and clayton family::
        foo = Copula(x, y, family='clayton');
        X1, Y1 = foo.generate_xy(N_points);
        #print X1.shape
        #print Y1.shape
        print "The iteration going on is", i;
        X2=X1;
        Y2=Y1;
        #plot_copula(X1,Y1,X2,Y2,'Copula_parabola_T')
        Data_array_Ref[:,i]=X1.copy();
        Data_array_Ref[:,i]=Y1.copy();
        Data_array_Test[:,i]=X2.copy();
        Data_array_Test[:,i]=Y2.copy();
        i=i+1;

    np.savetxt("foo_Ref.csv", Data_array_Ref, delimiter=",");
    np.savetxt("foo_Test.csv", Data_array_Test, delimiter=",");


    from numpy import genfromtxt
    dev= np.random.rand(100,1)*1000;
    dev=np.sort(dev, axis=None);
    Data_array_Ref= genfromtxt('foo_Ref.csv', delimiter=',')
    Data_array_Test= genfromtxt('foo_Test.csv', delimiter=',');
    Data_array_Test=Data_array_Test+40;
    # Calculate Distance between Class 2 and Class 1:
    row_ref= Data_array_Ref.shape[0];
    column_ref=Data_array_Ref.shape[1];
    row_test=Data_array_Test.shape[0];
    column_test=Data_array_Test.shape[1];

    for gam in [0.000001,0.00001,0.0001,0.001,0.01,0.1]:
        Res=[];
        for element in dev:

            # Declare the variables
            print("--- %s True Distance ---" % (element));
            Data_array_Ref= genfromtxt('foo_Ref.csv', delimiter=',')
            Data_array_Test= genfromtxt('foo_Test.csv', delimiter=',');
            Data_array_Test=Data_array_Test+element;
            mean_normal=[]
            std_normal=[]

            # Declare the Matrices
            Normalized_matrix_test_Data_class_1 = np.zeros((row_ref,column_ref));
            Normalized_matrix_test_Data_class_2 = np.zeros((row_test,column_test));

            # First Calculate the mean and the standard deviation\
            for i in range(0 , column_ref):
                a=np.array(Data_array_Ref[i]);
                mean_normal.append(a.mean());
                std_normal.append(a.std());

            #Calcualte Normalized matrix
            for j in range(0 , column_ref):
                for i in range(0 , row_ref):
                    Normalized_matrix_test_Data_class_1[i,j]=(Data_array_Ref[i,j]-mean_normal[j])/std_normal[j];
            for j in range(0 , column_test):
                for i in range(0 , row_test):
                    Normalized_matrix_test_Data_class_2[i,j]=(Data_array_Test[i,j]-mean_normal[j])/std_normal[j];

            numberFlag=1;
            CurrentFile=[]
            start_time=time.time()
            MD,CurrentFile=Initialization_kernel(Normalized_matrix_test_Data_class_1,  Normalized_matrix_test_Data_class_2, 'Simulated_Data',1,2, numberFlag,CurrentFile, gam);
            print("--- %s seconds ---" % (time.time() - start_time));
            Res.append(MD.mean());
            print "The mean of Class 2 from Ref is", MD.mean();
        colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c"];
        plot_copula_time(Res,dev,str(gam),'Exponential1'+str(gam) ,colors[2]);

        print "The resulting distances are", Res;
        Data_plot=np.zeros((len(Res), 2))
        Data_plot[:,0]=np.array(Res).copy();
        Data_plot[:,1]=np.array(dev).copy();
        np.savetxt("foo_data_plot_exponential" + str(gam) +".csv", Data_plot, delimiter=",");
        print("--- %s seconds ---" % (time.time() - start_time));

#########################################################################################################
def GammachangeNormal():
    deviation=[30];
    N_points=10000;
    N_Features=100;
    size_init_data=1000;
    Data_array_Ref=np.zeros((N_points,N_Features));
    Data_array_Test=np.zeros((N_points,N_Features));
    # Let us create the numpy array
    for i in range(0,N_Features):
        # Generate random (normal distributed) numbers::
        x = np.random.normal(size=size_init_data);
        y = x  + np.random.normal(size=size_init_data);
        # Make the instance of Copula class with x, y and clayton family::
        foo = Copula(x, y, family='clayton');
        X1, Y1 = foo.generate_xy(N_points);
        #print X1.shape
        #print Y1.shape
        print "The iteration going on is", i;
        X2=X1;
        Y2=Y1;
        #plot_copula(X1,Y1,X2,Y2,'Copula_parabola_T')
        Data_array_Ref[:,i]=X1.copy();
        Data_array_Ref[:,i]=Y1.copy();
        Data_array_Test[:,i]=X2.copy();
        Data_array_Test[:,i]=Y2.copy();
        i=i+1;

    np.savetxt("foo_Ref.csv", Data_array_Ref, delimiter=",");
    np.savetxt("foo_Test.csv", Data_array_Test, delimiter=",");


    from numpy import genfromtxt
    #dev= np.random.rand(100,1)*1000;
    #dev=np.sort(dev, axis=None);
    Data_array_Ref= genfromtxt('foo_Ref.csv', delimiter=',')
    Data_array_Test= genfromtxt('foo_Test.csv', delimiter=',');
    Data_array_Test=Data_array_Test+40;
    # Calculate Distance between Class 2 and Class 1:
    row_ref= Data_array_Ref.shape[0];
    column_ref=Data_array_Ref.shape[1];
    row_test=Data_array_Test.shape[0];
    column_test=Data_array_Test.shape[1];
    MD=traditional_MTS(Data_array_Ref,Data_array_Test)
    a1= (np.random.rand(50,1)/10000 ).tolist() # [0.000001,0.000002,0.000003,0.000005,0.000009,0.00001,0.00002,0.00004,0.00006,0.00008,0.00009,0.0001,0.0002,0.0003,0.0004,0.0006,0.0008,0.001,0.002,0.003,0.004];
    a2= (np.random.rand(50,1)/1000).tolist()
    a3= (np.random.rand(50,1)/100).tolist()


    gamma_array= np.asarray(a1+a2+a3);
    # Declare the variables
    Data_array_Ref= genfromtxt('foo_Ref.csv', delimiter=',')
    Data_array_Test= genfromtxt('foo_Test.csv', delimiter=',');
    Data_array_Test=Data_array_Test+30;
    mean_normal=[]
    std_normal=[]

    # Declare the Matrices
    Normalized_matrix_test_Data_class_1 = np.zeros((row_ref,column_ref));
    Normalized_matrix_test_Data_class_2 = np.zeros((row_test,column_test));

    # First Calculate the mean and the standard deviation\
    for i in range(0 , column_ref):
        a=np.array(Data_array_Ref[i]);
        mean_normal.append(a.mean());
        std_normal.append(a.std());

    #Calcualte Normalized matrix
    for j in range(0 , column_ref):
        for i in range(0 , row_ref):
            Normalized_matrix_test_Data_class_1[i,j]=(Data_array_Ref[i,j]-mean_normal[j])/std_normal[j];
    for j in range(0 , column_test):
        for i in range(0 , row_test):
            Normalized_matrix_test_Data_class_2[i,j]=(Data_array_Test[i,j]-mean_normal[j])/std_normal[j];
    Res=[];
    numberFlag=1;

    P= np.sort(gamma_array, axis=None)


    for gam in P:
        CurrentFile=[]
        start_time=time.time()
        MD,CurrentFile=Initialization_kernel(Normalized_matrix_test_Data_class_1,  Normalized_matrix_test_Data_class_2, 'Simulated_Data',1,2, numberFlag,CurrentFile, gam);
        print("--- %s seconds ---" % (time.time() - start_time));
        Res.append(MD.mean());
        print "The mean of Class 2 from Ref is", MD.mean();

    colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c"];
    Data_plot=np.zeros((len(Res),P))
    Data_plot[:,0]=np.array(Res).copy();
    Data_plot[:,1]=np.array(dev).copy();
    np.savetxt("foo_data_plot_Normal_gamma_change" + str(gam) +".csv", Data_plot, delimiter=",");
    plot_copula_time(Res,P, 'NGDM-HDR Vs Gamma','GammaNorm_1'+str(gam) ,colors[3]);





def Test_RBF():
    deviation=40
    N_points=1000;
    N_Features=2;
    size_init_data=100;

    # Declare the data arrays
    Data_array_Ref=np.zeros((N_points,N_Features));
    Data_array_Test=np.zeros((N_points,N_Features));


    # Generate random (normal distributed) numbers::
    x = np.random.normal(size=size_init_data);
    y = np.power(x,1)+np.exp(x)+np.random.normal(size=size_init_data);

    # Make the instance of Copula class with x, y and clayton family::
    foo = Copula(x, y, family='clayton');
    X1, Y1 = foo.generate_xy(N_points);


    # Plot the
    Data_array_Ref[:,0] = X1.copy();
    Data_array_Ref[:,1] = Y1.copy();
    Data_array_Test[:,0]= X1.copy()+deviation;
    Data_array_Test[:,1]= Y1.copy()+deviation;

    # plot the copula
    # plot_copula(Data_array_Ref[:,0], Data_array_Ref[:,1], Data_array_Test[:,0], Data_array_Test[:,1], 'Copula_parabola_T');
    start_time=time.time();
    for gam in [0.000001,0.00001,0.0001,0.001,0.01,0.1,0.5,1,10]:
        MD_test=np.zeros((N_points,2));
        MD_normal=np.zeros((N_points,2));
        alphas, lambdas = stepwise_kpca(Data_array_Ref, gamma=gam, n_components=2);
        for i in range(0 , N_points):
            MD_normal[i][:]=project_x(Data_array_Ref[i][:], Data_array_Ref, gamma=gam, alphas=alphas, lambdas=lambdas)
        for i in range(0 , N_points):
            MD_test[i][:]=project_x(Data_array_Test[i][:], Data_array_Ref , gamma=gam, alphas=alphas, lambdas=lambdas)
        np.savetxt("Trans_normal" + str(gam) +".csv", MD_normal, delimiter=",");
        np.savetxt("Trans_Test" + str(gam) +".csv", MD_test, delimiter=",");
        print("--- %s Seconds ---" % (time.time() - start_time));
        print("--- %s Gamma ---" % gam);
        # plot_copula(MD_normal[:,0], MD_normal[:,1], MD_test[:,0], MD_test[:,1], 'Copula_parabola_T');

def main():
    #Run and start analysis....
    print "Lets start Analysis";
    print "The First run is the Weibull Distribution";
    ##Weibull();
    print "The next run is for normal Distribution";
    Normal();
    print "Now we run for exponential distribution"
    #Exponential();
    print "\n\n Next let us go about changing the gamma values for a particular deviation";
    # GammachangeNormal()
    print "Test the RBF functions";
    # Test_RBF();

main()
