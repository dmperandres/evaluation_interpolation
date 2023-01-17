# Copyright (c) 2022, 2023 Domingo Martin Perandrés (dmartin@ugr.es)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this program, to deal in the program without restriction, including without limitation the rights to use, copy, modify, merge, publish and distribute.
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# 
# The data of "The Man" can be freely downloaded (https://doi.org/10.25573/data.14588262)
# 
# The data of "The Transfiguration" must be requested to mrblanc@ugr.es.

import math
import sys
import random
import os
import glob
import statistics
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import pathlib

# Version: FULL, SMALL
Version='SMALL'

# For saving the results as images
Save=True

File_type='.pdf'

Num_points=1314

# percentages
Vec_percentages=[5,10,20,30,40,50,60,70,80,90]

# Number of test points
Vec_num_points=[int(round(float(x)*float(Num_points)/100.0)) for x in Vec_percentages]

if Version=='FULL':
    # Elements=["As","Ba","Ca","Cd","Co","Cr","Cu","Fe","Hg","K","Mn","Ni","Pb","Sb","Se","Sn","Ti","Zn"]
    Elements=["Cu","Fe","Hg","Pb","Ti"]
    Test_methods = ["MHD", "BARNES", "CRESSMAN", "KRIGING", "RBF"]
    Test_folders = ["mhd", "barnes", "cressman", "kriging", "rbf"]
    #  Kriging functions
    Krigin_variogram_model = ["linear", "power", "gaussian", "spherical", "exponential", "hole-effect"]
    # RBF functions
    RBF_functions = ["multiquadric", "inverse", "gaussian", "linear", "cubic", "quintic", "thin_plate"]
    # BC frunctions
    Barnes_functions = ['barnes', 'cressman']
elif Version=='SMALL':
    Elements = ["Cu", "Fe", "Hg", "Pb", "Ti"]
    Test_methods = ["MHD", "BARNES", "CRESSMAN", "KRIGING", "RBF"]
    Test_folders = ["mhd", "barnes", "cressman", "kriging", "rbf"]
    #  Kriging functions
    Krigin_variogram_model = ["exponential"]
    # RBF functions
    RBF_functions = ["linear"]
    # BC frunctions
    Barnes_functions = ['barnes']

#
MHD_functions=["cccpp"]

# folders
Input_folder = "input_data/interpolation_methods"
Output_folder = "output_data"

def read_map_data(File_name,Num_points):
    File_in = open(File_name, "r")

    Lines = [Line for Line in File_in]

    Values=[0.0]*(165-Num_points)
    for i in range(165-Num_points):
        Line = Lines[i+5]
        Line = Line.rstrip()
        # print(i, Line)
        Tokens = Line.split(";")
        Values[i]=float(Tokens[1])

    File_in.close()

    return Values


def read_mse_data(Element, Folder):
    print(Folder + "/" + Element + "_mse.txt")

    File_in = open(Folder + "/" + Element + "_mse.txt", "r")

    Lines = [Line for Line in File_in]

    Vec_mean=[]
    for i in range(len(Lines)):
        # msq, std, min, max
        Line = Lines[i]
        Line = Line.rstrip()
        Tokens = Line.split(";")
        Vec_mean.append(float(Tokens[0]))

    File_in.close()

    # Max=-1;
    # for i in range(len(Vec_max)):
    #     if Vec_max[i]>Max:
    #         Max=Vec_max[i]

    return Vec_mean


def main():
    print("Working")

    # create the Input_paths and Output_paths
    Input_paths = []
    Output_paths = []

    Methods = []
    for Test_folder in Test_folders:
        if Test_folder == 'kriging':
            for Kriging_function in Krigin_variogram_model:
                Input_paths.append(
                    Input_folder + '/' + Test_folder + '/output_data/' + Test_folder + '/' + Kriging_function)
                Methods.append(Test_folder.upper() + " " + Kriging_function.upper())
                Output_paths.append(Output_folder + '/' + Test_folder + '/' + Kriging_function)
        elif Test_folder == 'rbf':
            for Radial_function in RBF_functions:
                Input_paths.append(
                    Input_folder + '/' + Test_folder + '/output_data/' + Test_folder + '/' + Radial_function)
                Methods.append(Test_folder.upper() + " " + Radial_function.upper())
                Output_paths.append(Output_folder + '/' + Test_folder + '/' + Radial_function)
        elif Test_folder == 'mhd':
            for MHD_function in MHD_functions:
                Input_paths.append(
                    Input_folder + '/' + Test_folder + '/output_data/' + Test_folder + '/' + MHD_function)
                Methods.append(Test_folder.upper())
                Output_paths.append(Output_folder + '/' + Test_folder + '/' + MHD_function)
        elif Test_folder == 'barnes':
                Input_paths.append(Input_folder + '/' + Test_folder)
                Methods.append(Test_folder.upper())
                Output_paths.append(Output_folder + '/' + Test_folder)
        elif Test_folder == 'cressman':
            Input_paths.append(Input_folder + '/' + Test_folder)
            Methods.append(Test_folder.upper())
            Output_paths.append(Output_folder + '/' + Test_folder)

    # print(Input_paths)
    # print(Output_paths)
    # print(Methods)
    # return


    Means_folder_element = []
    Stds_folder_element = []
    Pvalues_folder_element = []

    Max_means = -1;
    # As, Ti,..
    for Element_name in Elements:

        Means_folder=[]
        Stds_folder = []
        Pvalues_folder = []

        # cccpp, ccc,
        for Input_path in Input_paths:
            Means = []
            Stds = []
            Pvalues = []

            for Num_points in Vec_num_points:
                Folder_aux = Input_path + "/" + "{0:03d}".format(Num_points) + "_samples"

                # get the 100 values
                Vec_mean = read_mse_data(Element_name, Folder_aux)

                # computes if it is a normal distribution
                w, pvalor = ss.shapiro(Vec_mean)
                Pvalues.append(pvalor)

                # compute the mean
                Mean = np.mean(Vec_mean, axis=-1)

                # convert to a np.array
                Data_test = np.array(Vec_mean)

                # for the scipy bootstrap
                Data_test = (Data_test,)

                # print(Data_test)
                # boot strap
                # Result = bs.confidence_interval(Data_test)
                Result = ss.bootstrap(Data_test, np.mean)

                # save result
                Means.append(Mean)
                # print("Mean=",Mean,"Ci=",Result[0])

                # std deviation
                # Std_value=Result[0] - Mean
                Std_value = Result.standard_error
                Stds.append(Std_value)

                if Mean + math.fabs(Std_value) > Max_means:
                    print("Mean ", Mean, Std_value)

                    Max_means = Mean + math.fabs(Std_value)

            # print(Means)
            # print(Stds)

            # save the list of results (10,30,50,...
            Means_folder.append(Means)
            Stds_folder.append(Stds)
            Pvalues_folder.append(Pvalues)

        #save for each path
        Means_folder_element.append(Means_folder)
        Stds_folder_element.append(Stds_folder)
        Pvalues_folder_element.append(Pvalues_folder)
        # compute the max_y


    # computes the Max_y to show the same range for all the images
    #Max_y=round(Max_means*1.05)/10

    Max_y = 0.10
    Num_steps_y=5

    for Pos_element in range(len(Elements)):
        Element_name=Elements[Pos_element]

        Means_folder=Means_folder_element[Pos_element]
        Stds_folder=Stds_folder_element[Pos_element]
        Pvalues_folder=Pvalues_folder_element[Pos_element]

        for i in range(len(Output_paths)):
            Output_path = Output_paths[i]

            if os.path.isdir(Output_path + "/results_images_bars_mse_mean_ci_bootstrap") == False:
                os.mkdir(Output_path + "/results_images_bars_mse_mean_ci_bootstrap")

            Means = Means_folder[i]
            Stds = Stds_folder[i]
            Pvalues = Pvalues_folder[i]

            # print(Means)
            # print(Stds)

            plt.rcParams['font.size'] = '16'

            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.set_title(Element_name + " " + Methods[i], fontsize=20, y=1.0, pad=-20)

            Vec_x = [i for i in range(len(Means))]
            rects1 = ax.bar(Vec_x, Means, yerr=Stds)

            for patch, pvalor in zip(rects1.patches, Pvalues):
                # print(result)
                if pvalor < 0.05:
                    patch.set_facecolor('pink')
                else:
                    patch.set_facecolor('lightgreen')

            # process to create the distribution
            Vec_x_ticks = ["{:d}".format(i) + '%' for i in Vec_percentages]

            Vec_y=[0.0]*Num_steps_y
            Vec_y_ticks = [''] * Num_steps_y

            for j in range(Num_steps_y):
                Vec_y[j] = Max_y * float(j) / float(Num_steps_y - 1)
                Vec_y_ticks[j] = "{0:d}%".format(int(Vec_y[j] * 100))

            ax.yaxis.grid(True)

            plt.ylim(ymin=0, ymax=Max_y)

            ax.set_xticks(Vec_x)
            ax.set(xticklabels=Vec_x_ticks)

            ax.set_yticks(Vec_y)
            ax.set(yticklabels=Vec_y_ticks)

            ax.set_xlabel('% of used points')
            ax.set_ylabel('MSE + 95% CI')

            if Save == False:
                plt.show();
            else:
                # print("Saving "+Path+"/results_images_mean_bootstrap/"+Element_name+".png")
                Method=Methods[i]

                Tokens=Method.split(' ')
                Name=''
                if len(Tokens)>1:
                    Name="_"+Tokens[0]+"_"+Tokens[1]
                else:
                    Name = "_" + Tokens[0]

                plt.savefig(Output_path+"/results_images_bars_mse_mean_ci_bootstrap/"+Element_name+Name+File_type,bbox_inches='tight',transparent=True)
                plt.close()

if __name__=="__main__":
    main()