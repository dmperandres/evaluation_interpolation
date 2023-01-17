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
from skimage import io, color
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

# modes
# MHD_modes=['cccpp','ccc','pp']
MHD_modes=['cccpp']

# selected color map
# Map='rainbow'
Map=Blindcmp

# parameters
Save=True
Output_folder = "output_data"
Num_rows=357
Num_cols=204
Num_points=1314

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

Test_method="MHD"
Test_folder="mhd"

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


def mhd(Mode,Coordinates_x,Coordinates_y,Colors,Intensities,Selected_pos,Image_rgb):
	Height1, Width1, Depth = Image_rgb.shape

	Mat_intensities= np.zeros((Height1,Width1))

	for Row in range(Height1):
		y = float(Row) / (Height1 - 1.0)
		for Col in range(Width1):
			x = float(Col) / (Width1 - 1.0)
			Color = Image_rgb[Row, Col]
			Color = [float(Color[i]) / 255.0 for i in range(3)]

			Min=1e10
			Pos_min=-1
			for j in range(len(Selected_pos)):
				x_aux=Coordinates_x[Selected_pos[j]]
				y_aux = Coordinates_y[Selected_pos[j]]
				Color_aux = Colors[Selected_pos[j]]

				#compute the distance
				Distance = math.sqrt(
						math.pow(Color[0] - Color_aux[0], 2) + math.pow(Color[1] - Color_aux[1], 2) + math.pow(
							Color[2] - Color_aux[2], 2) + math.pow(x - x_aux, 2) + math.pow(y - y_aux, 2))

				if Distance<Min:
					Min=Distance
					Pos_min=Selected_pos[j]

			Mat_intensities[Row][Col]=Intensities[Pos_min]

	return  Mat_intensities


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


def draw_error(Element,Num_random_pos,Mat,Selected_x,Selected_y):
	Num_rows1=len(Mat)
	Num_cols1=len(Mat[0])

	fig, ax = plt.subplots(1, 1, figsize=(10.0, 8.0 * (float(Num_rows1) / float(Num_cols1))))
	if Draw_title == True:
		ax.set_title(Element + " MHD Squared Error")

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

		Folder2 = Folder + "/" + Mode + "/result_images_interpolation"
		if os.path.isdir(Folder2) == False:
			os.mkdir(Folder2)

		for Num_points1 in Vec_num_points:
			Folder3 = Folder2 + "/" + "{0:03d}".format(Num_points1) + "_samples"
			if os.path.isdir(Folder3) == False:
				os.mkdir(Folder3)


	# image
	Image_rgb = io.imread("input_data/vis_visible_small.png")

	# positions
	Width, Height, Coordinates_x, Coordinates_y = read_positions("input_data/positions.txt")

	# read colors
	Colors = read_colors("input_data/image_colors.txt")

	for Element in Elements:
		Intensities = read_intensities("input_data/normalized_data/" + Element + "_normalized.txt")

		for Mode in MHD_modes:
			random.seed(2022)

			for Num_random_pos in Vec_num_points:
				# get random positions
				Valid, Selected_pos = get_random_positions(Num_random_pos, Num_points)

				Mat_intensities1 = mhd(Mode,Coordinates_x, Coordinates_y, Colors, Intensities, Selected_pos,Image_rgb)

				Selected_x = [0.0] * Num_random_pos
				Selected_y = [0.0] * Num_random_pos
				for i in range(Num_random_pos):
					Selected_x[i] = int(Coordinates_x[Selected_pos[i]] * float(Num_cols))
					Selected_y[i] = int(Coordinates_y[Selected_pos[i]] * float(Num_rows))

				File_name = Output_folder + "/" + Test_folder + "/" + Mode + "/result_images_interpolation/" + "{0:03d}".format(
					Num_random_pos) + "_samples/" + Element + "_" + Test_method + "_" + Mode + "_{0:03d}".format(
					Num_random_pos) + File_type

				print("Saving " + File_name)
				draw_result(File_name, Test_method + " " + Mode, Element, Num_random_pos, Mat_intensities1,Selected_x, Selected_y)

if __name__=="__main__":
	main()
