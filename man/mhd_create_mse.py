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
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os
import time

# For saving the results as images
Save=False

Map='rainbow'
Output_folder = "output_data"
Num_rows=357
Num_cols=204
Num_points=1314

# elements
#Elements=["As","Ba","Ca","Cd","Co","Cr","Cu","Fe","Hg","K","Mn","Ni","Pb","Sb","Se","Sn","Ti","Zn"]
Elements=["Cu","Fe","Hg","Pb","Ti"]

# percentages
Vec_percentages=[5,10,20,30,40,50,60,70,80,90]

# Number of test points
Vec_num_points=[int(round(float(x)*float(Num_points)/100.0)) for x in Vec_percentages]

# Functions
Krigin_variogram_model=["linear","power","gaussian","spherical","exponential","hole-effect"]

Num_tests=100

# Modes
# MHD_modes=['cccpp','ccc','pp']
MHD_modes=['cccpp']

Test_method="MHD"
Test_folder="mhd"

def read_positions(File_name):
    File_in = open(File_name, "r")
    Lines = [Line for Line in File_in]

    # Hidth
    Line = Lines[0]
    Line = Line.rstrip()
    Tokens = Line.split(";")
    Width = float(Tokens[1])

    # Height
    Line = Lines[1]
    Line = Line.rstrip()
    Tokens = Line.split(";")
    Height = float(Tokens[1])

    Vec_pos = []
    Coordinates_x = []
    Coordinates_y = []
    for i in range(3, len(Lines)):
        Line = Lines[i]
        if Line != "":
            # print(Line)
            Line = Line.rstrip()
            Tokens = Line.split(";")
            Vec_pos.append(int(Tokens[0]) - 1)
            Coordinates_x.append(float(Tokens[1]) / float(Width - 1))
            Coordinates_y.append(float(Tokens[2]) / float(Height - 1))

    File_in.close()

    # print("X=",Coordinates_x)

    return Width, Height, Vec_pos, Coordinates_x, Coordinates_y	
	

def read_colors(File_name):
	File_in=open(File_name,"r")

	Lines = [Line for Line in File_in]

	Colors=[]
	for i in range(1,len(Lines)):
		Line = Lines[i]
		Line = Line.rstrip()
		Tokens = Line.split(";")
		Colors.append((float(Tokens[0])/255.0,float(Tokens[1])/255.0,float(Tokens[2])/255.0))

	File_in.close()

	return Colors


def read_normalized_data(File_name):
	File_in=open(File_name,"r")

	Lines = [Line for Line in File_in]

	Values=[0.0]*Num_points
	for i in range(Num_points):
		Line = Lines[i]
		Line = Line.rstrip()
		Values[i]=float(Line)

	File_in.close()

	return Values


def read_intensities(File_name):
	File_in=open(File_name,"r")

	Lines = [Line for Line in File_in]

	Intensities=[]
	for i in range(len(Lines)):
		Line = Lines[i]
		Line = Line.rstrip()
		Token = Line
		Intensities.append(float(Token))

	File_in.close()

	return Intensities


def get_random_positions(Num_positions,Num_total_positions):

	Valid=[False]*Num_total_positions

	# Selected_x= [0] * Num_positions
	# Selected_y = [0] * Num_positions
	#
	# Selected_x_normalized=[0.0]*Num_positions
	# Selected_y_normalized= [0.0] * Num_positions

	Selected_pos=[0]*Num_positions


	Num_pos=0
	while Num_pos<Num_positions:
		Pos=random.randint(0,Num_total_positions-1)
		if Valid[Pos]==False:
			Valid[Pos]=True
			Selected_pos[Num_pos]=Pos
			Num_pos=Num_pos+1


	return Valid,Selected_pos


def mhd(Mode,Coordinates_x,Coordinates_y,Colors,Intensities,Selected_pos):

	Intensities_result=[0.0]*Num_points

	# print(Intensities)

	for i in range(Num_points):
		x=Coordinates_x[i]
		y=Coordinates_y[i]
		Color=Colors[i]

		Min=1e10
		Pos_min=-1
		for j in range(len(Selected_pos)):
			x_aux=Coordinates_x[Selected_pos[j]]
			y_aux = Coordinates_y[Selected_pos[j]]
			Color_aux = Colors[Selected_pos[j]]


			#compute the distance
			if Mode=='cccpp':
				Distance = math.sqrt(
					math.pow(Color[0] - Color_aux[0], 2) + math.pow(Color[1] - Color_aux[1], 2) + math.pow(
						Color[2] - Color_aux[2], 2) + math.pow(x - x_aux, 2) + math.pow(y - y_aux, 2))
			elif Mode=='ccc':
				Distance = math.sqrt(math.pow(Color[0] - Color_aux[0], 2) + math.pow(Color[1] - Color_aux[1], 2) + math.pow(Color[2] - Color_aux[2], 2))
			else:
				Distance = math.sqrt(math.pow(x - x_aux, 2) + math.pow(y - y_aux, 2))

			if Distance<Min:
				Min=Distance
				Pos_min=Selected_pos[j]

		Intensities_result[i]=Intensities[Pos_min]

	return  Intensities_result


def to_mat(Intensities):
	# convert to matrix shape [15,11]
	Mat = []
	while Intensities != []:
		Mat.append(Intensities[:Num_cols])
		Intensities = Intensities[Num_cols:]

	return Mat


def matrix_positions():
	Coordinates_x=[0.0]*Num_points
	Coordinates_y=[0.0]*Num_points
	for i in range(Num_points):
		# print("col=",i%Num_cols, " row=",int(i /Num_cols))
		Coordinates_x[i]=float(i%Num_cols)/float(Num_cols-1)
		Coordinates_y[i] = float(int(i /Num_cols))/float(Num_rows-1)

	# print("x_n=",Coordinates_x)

	return Coordinates_x,Coordinates_y



def draw_result(Element,Num_random_pos,Mat,Selected_x,Selected_y):
	fig, ax = plt.subplots(1, 1, figsize=(10.0, 8.0 * (float(Num_rows) / float(Num_cols))))
	ax.set_title(Element + " MHD")
	img_plot = plt.imshow(Mat, vmin=0, vmax=1, cmap=plt.get_cmap(Map))
	img_plot.set_cmap(plt.get_cmap(Map))
	plt.scatter(Selected_x, Selected_y, c='black')

	# create an axes on the right side of ax. The width of cax will be 5%
	# of ax and the padding between cax and ax will be fixed at 0.05 inch.
	ax = plt.gca()
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.25)
	plt.colorbar(img_plot, cax=cax)

	if Save == False:
		plt.show();
	else:
		plt.savefig(Output_folder + Element +"_{:03d}".format(Num_random_pos)+"_mhd_map.png")
		plt.close()


def draw_error(Element,Num_random_pos,Mat,Selected_x,Selected_y):
	fig, ax = plt.subplots(1, 1, figsize=(10.0, 8.0 * (float(Num_rows) / float(Num_cols))))
	ax.set_title(Element + " MHD Squared Error")
	img_plot = plt.imshow(Mat, vmin=0, vmax=1, cmap=plt.get_cmap(Map))
	img_plot.set_cmap(plt.get_cmap(Map))
	plt.scatter(Selected_x, Selected_y, c='black')

	# create an axes on the right side of ax. The width of cax will be 5%
	# of ax and the padding between cax and ax will be fixed at 0.05 inch.
	ax = plt.gca()
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.25)
	plt.colorbar(img_plot, cax=cax)

	if Save == False:
		plt.show();
	else:
		plt.savefig(Output_folder + Element +"_{:03d}".format(Num_random_pos)+"_mhd_map_error.png")
		plt.close()


def save_mse(Folder,Element,Vec_mse):
	print("Writing in "+Folder + "/" + Element + "_mse.txt")

	File_out = open(Folder + "/" + Element + "_mse.txt", "w")

	for i in range(len(Vec_mse)):
		# MSE
		File_out.write("{:8.6f}".format(Vec_mse[i]) + '\n')

	File_out.close()


def save_time(Folder,Element,Vec_time):
	print("Writing in "+Folder + "/" + Element + "_time.txt")

	File_out = open(Folder + "/" + Element + "_time.txt", "w")

	for i in range(len(Vec_time)):
		# MSE
		File_out.write("{:8.6f}".format(Vec_time[i]) + '\n')

	File_out.close()


def main():
	print("Working")

	Folder = Output_folder + "/" + Test_folder
	if os.path.isdir(Folder) == False:
		os.mkdir(Folder)

	for Mode in MHD_modes:
		Folder1 = Folder + "/" + Mode
		if os.path.isdir(Folder1) == False:
			os.mkdir(Folder1)

		Folder_time = Folder + "/" + Mode + "/time"
		if os.path.isdir(Folder_time) == False:
			os.mkdir(Folder_time)

		for Num_points1 in Vec_num_points:
			Folder2 = Folder1 + "/" + "{0:03d}".format(Num_points1) + "_samples"
			if os.path.isdir(Folder2) == False:
				os.mkdir(Folder2)

	# read colors
	Colors = read_colors("input_data/image_colors.txt")

	# usinf the small position coordinates
	Width, Height, Vec_pos, Coordinates_x, Coordinates_y = read_positions("input_data/positions.txt")

	for Element in Elements:

		Intensities = read_intensities("input_data/normalized_data/" + Element + "_normalized.txt")

		for Mode in MHD_modes:

			random.seed(2022)

			Folder_time = Folder + "/" + Mode + "/time"

			Vec_time = [0.0] * len(Vec_num_points)

			Pos_time = 0
			for Num_random_pos in Vec_num_points:
				Folder3 = Folder + "/"+Mode+"/"+ "{0:03d}".format(Num_random_pos) + "_samples"

				Vec_mse=[0.0]*Num_tests

				Time = time.time()
				for Num_test in range(Num_tests):
					# get random positions
					Valid, Selected_pos = get_random_positions(Num_random_pos, Num_points)
					
					Intensities_result = mhd(Mode,Coordinates_x, Coordinates_y, Colors, Intensities, Selected_pos)

					# error
					MSE = 0.0
					for i in range(Num_points):
						if Valid[i] == False:
							MSE = MSE + math.pow(Intensities[i] - Intensities_result[i], 2)

					# print(Num_points,Num_random_pos)
					Vec_mse[Num_test] = MSE / float(Num_points - Num_random_pos)

				save_mse(Folder3, Element, Vec_mse)

				Vec_time[Pos_time] = (time.time() - Time) / 100.0
				Pos_time = Pos_time + 1

			save_time(Folder_time, Element, Vec_time)




if __name__=="__main__":
	main()
