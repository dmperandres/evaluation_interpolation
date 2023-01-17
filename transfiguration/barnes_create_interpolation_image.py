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
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os
from scipy.spatial import cKDTree
import time
from metpy.interpolate.geometry import dist_2
from metpy.interpolate.points import barnes_point, cressman_point
from metpy.interpolate.tools import average_spacing, calc_kappa
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# colormap for color blind people
cdict = {'red':   [[0.0,  100/255, 100/255],
                   [0.25,  120/255, 120/255],
                   [0.5,  220/255, 220/255],
                   [0.75,  254/255, 254/255],
                   [1.0,  1, 1]],
         'green': [[0.0,  143/255, 143/255],
                   [0.25,  94/255, 94/255],
                   [0.5,  38/255, 38/255],
                   [0.75,  97/255, 97/255],
                   [1.0,  176/255, 176/255]],
         'blue':  [[0.0,  1.0, 1.0],
                   [0.25,  240/255, 240/255],
                   [0.5,  127/255, 127/255],
                   [0.75,  0, 0],
                   [1.0,  0, 0]]}

Blindcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)

# elements to be computed
# Elements=["As","Ba","Ca","Cd","Co","Cr","Cu","Fe","Hg","K","Mn","Ni","Pb","Sb","Se","Sn","Ti","Zn"]
Elements=["Pb"]

# selected color map
# Map='rainbow'
Map=Blindcmp

# parameters
Save=True
Output_folder = "output_data"
Num_rows=1112
Num_cols=756
Num_points=165

Draw_color_map=False
Draw_ticks_labels=False
Draw_title=False
Color_map_blind=True
File_type='.png'

# percentages
Vec_percentages=[5,10,20,30,40,50,60,70,80,90]

# Number of test points
# Vec_num_points=[int(round(float(x)*float(Num_points)/100.0)) for x in Vec_percentages]
Vec_num_points=[Num_points]

# output density
dpi=100
# pixels
Output_max_size=2000

Test_method="BARNES"
Test_folder="barnes"


def read_positions(File_name):
	File_in=open(File_name,"r")
	Lines = [Line for Line in File_in]

	# Hidth
	Line = Lines[0]
	Line = Line.rstrip()
	Tokens = Line.split(";")
	Width=float(Tokens[1])

	# Height
	Line = Lines[1]
	Line = Line.rstrip()
	Tokens = Line.split(";")
	Height=float(Tokens[1])
	
	Coordinates_x=[]
	Coordinates_y=[]
	for i in range(3,len(Lines)):
		Line = Lines[i]
		if Line!="":
			# print(Line)
			Line = Line.rstrip()
			Tokens = Line.split(";")
			Coordinates_x.append(float(Tokens[1])/float(Width-1))
			Coordinates_y.append(float(Tokens[2])/float(Height-1))
		
	File_in.close()

	# print("X=",Coordinates_x)
	
	return Width,Height,Coordinates_x,Coordinates_y
	

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


def mhd(Coordinates_x,Coordinates_y,Colors,Intensities,Selected_pos):

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
			Distance=math.sqrt(math.pow(Color[0]-Color_aux[0],2)+math.pow(Color[1]-Color_aux[1],2)+math.pow(Color[2]-Color_aux[2],2)+math.pow(x-x_aux,2)+math.pow(y-y_aux,2))

			if Distance<Min:
				Min=Distance
				Pos_min=Selected_pos[j]

		Intensities_result[i]=Intensities[Pos_min]

	return  Intensities_result


def to_mat(Intensities):
	# convert to matrix shape [15,11]
	Mat = []
	while len(Intensities)>0:
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



def draw_result(File_name,Test_name,Element,Num_random_pos,Mat,Selected_x,Selected_y):
	if Num_rows > Num_cols:
		Height_aux = Output_max_size / dpi;
		Width_aux = Height_aux * (float(Num_rows) / float(Num_cols))
	else:
		Width_aux = Output_max_size / dpi;
		Height_aux = Width_aux * (float(Num_rows) / float(Num_cols))

	fig, ax = plt.subplots(1, 1, figsize=(Width_aux, Height_aux))
	fig.subplots_adjust(wspace=0,hspace=0)

	if Draw_title==True:
		ax.set_title(Element+" "+Test_name,fontsize=20)

	img_plot = plt.imshow(Mat, vmin=0, vmax=1, cmap=plt.get_cmap(Map))

	plt.axis('off')

	#
	if Draw_ticks_labels==False:
		ax.axes.xaxis.set_visible(False)
		ax.axes.yaxis.set_visible(False)

	# create an axes on the right side of ax. The width of cax will be 5%
	# of ax and the padding between cax and ax will be fixed at 0.05 inch.
	if Draw_color_map==True:
		ax = plt.gca()
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="5%", pad=0.25)
		plt.colorbar(img_plot, cax=cax)

	if Save == False:
		plt.show();
	else:
		plt.savefig(File_name,bbox_inches='tight',transparent=True,pad_inches=0.0,dpi=dpi)
		plt.close()


def draw_error(Test_name,Element,Num_random_pos,Mat,Selected_x,Selected_y):
	fig, ax = plt.subplots(1, 1, figsize=(12.0, 10.0 * (float(Num_rows) / float(Num_cols))))
	if Draw_title == True:
		ax.set_title(Element + " "+Test_name+" Squared Error")

	img_plot = plt.imshow(Mat, vmin=0, vmax=1, cmap=plt.get_cmap(Map))
	img_plot.set_cmap(plt.get_cmap(Map))
	plt.scatter(Selected_x, Selected_y, c='black')

	if Draw_ticks_labels==False:
		ax.axes.xaxis.set_visible(False)
		ax.axes.yaxis.set_visible(False)

	# create an axes on the right side of ax. The width of cax will be 5%
	# of ax and the padding between cax and ax will be fixed at 0.05 inch.
	if Draw_color_map == True:
		ax = plt.gca()
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="5%", pad=0.25)
		plt.colorbar(img_plot, cax=cax)

	if Save == False:
		plt.show();
	else:
		plt.savefig(Output_folder + Element +"_{:03d}".format(Num_random_pos)+"_"+Test_name+"_map_error.png")
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

	Folder2 = Folder + "/result_images_interpolation"
	if os.path.isdir(Folder2) == False:
		os.mkdir(Folder2)

	for Num_points1 in Vec_num_points:
		Folder3 = Folder2 + "/" + "{0:03d}".format(Num_points1) + "_samples"
		if os.path.isdir(Folder3) == False:
			os.mkdir(Folder3)

	Width, Height, Coordinates_x, Coordinates_y = read_positions("input_data/positions.txt")

	radius = 0.5

	x_unknown=[0.0]*(Num_rows*Num_cols)
	y_unknown = [0.0] * (Num_rows * Num_cols)

	Pos=0
	for Row in range(Num_rows):
		for Col in range(Num_cols):
			x_unknown[Pos]=float(Col)/float(Num_cols-1)
			y_unknown[Pos] = float(Row)/float(Num_rows-1)
			Pos=Pos+1

	Unknown_points_positions = np.array(list(zip(x_unknown, y_unknown)))

	for Element in Elements:
		Intensities = read_intensities("input_data/normalized_data/" + Element + "_normalized.txt")

		random.seed(2022)

		for Num_random_pos in Vec_num_points:
			# get random positions
			Valid, Selected_pos = get_random_positions(Num_random_pos, Num_points)

			# create the set of vectors where the data is true
			x_known = [Coordinates_x[Selected_pos[i]] for i in range(len(Selected_pos))]
			y_known = [Coordinates_y[Selected_pos[i]] for i in range(len(Selected_pos))]
			z_known = [Intensities[Selected_pos[i]] for i in range(len(Selected_pos))]

			#unknown
			Known_points_tree = cKDTree(list(zip(x_known, y_known)))

			# for each unknown position, returns the list of know positions inside the circle of radius r
			Indices_true_points_in_circle = Known_points_tree.query_ball_point(Unknown_points_positions, r=radius)

			kappa = calc_kappa(average_spacing(list(zip(x_known, y_known))))

			Intensities_result=[0.0]*(Num_rows*Num_cols)

			for i in range(Num_rows*Num_cols):
				# get the points that are in the circle for the unknown point at pos Pos
				# get the points that are in the circle for the unknown point at pos Pos
				x2, y2 = Known_points_tree.data[Indices_true_points_in_circle[i]].T

				# computes the distance of the selected point to the points in the circle
				barnes_dist = dist_2(x_unknown[i], y_unknown[i], x2, y2)

				# get the values of the positions that are in the circle
				barnes_obs = [Intensities[x] for x in Indices_true_points_in_circle[i]]

				# computes the barnes value
				barnes_val = barnes_point(barnes_dist, barnes_obs, kappa)

				Intensities_result[i] = barnes_val

			Mat_intensities1=to_mat(Intensities_result)
			#
			Selected_x = [0.0] * Num_random_pos
			Selected_y = [0.0] * Num_random_pos
			for i in range(Num_random_pos):
				Selected_x[i] = int(Coordinates_x[Selected_pos[i]] * float(Num_cols))
				Selected_y[i] = int(Coordinates_y[Selected_pos[i]] * float(Num_rows))

			File_name = Output_folder + "/" + Test_folder +"/result_images_interpolation/" + "{0:03d}".format(
				Num_random_pos) + "_samples/" + Element + "_"+Test_method+"_" + "{0:03d}".format(Num_random_pos) + File_type

			print("Saving " + File_name)
			draw_result(File_name, Test_method, Element, Num_random_pos, Mat_intensities1, Selected_x,	Selected_y)

if __name__=="__main__":
	main()
