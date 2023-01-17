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

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
import numpy as np
from skimage import io, color
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os

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

# colormap
# Map='rainbow'
Map=Blindcmp

# Parameters
Save=True
Num_points=165
Output_folder="output_data/auxiliar_images"

# Input image: 'ORIGINAL','ORIGINAL_UNSATURATED','MAP'
Input_image='MAP'

# draw mode: "ORIGINAL","POSITIONS","SELECTED_POSITIONS","UNSELECTED_POSITIONS_WITH_ERROR"
Draw_mode="UNSELECTED_POSITIONS_WITH_ERROR"

# draw ticks in graph
Draw_ticks_labels=False

# output type
File_type='.png'

# output density
dpi=100
# pixels
Output_max_size=2000

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


def get_random_positions(Num_positions, Num_total_positions):
    if Num_positions == Num_total_positions:
        Valid = [True] * Num_total_positions
        Selected_pos = [i for i in range(Num_total_positions)]
        return Valid, Selected_pos

    else:
        Valid = [False] * Num_total_positions

        Selected_pos = [0] * Num_positions

        Num_pos = 0
        while Num_pos < Num_positions:
            Pos = random.randint(0, Num_total_positions - 1)
            if Valid[Pos] == False:
                Valid[Pos] = True
                Selected_pos[Num_pos] = Pos
                Num_pos = Num_pos + 1

        return Valid, Selected_pos


def draw_result(File_name,Image_rgb,Num_selected_points,Valid,Coordinates_x,Coordinates_y):
    Height1, Width1, Depth = Image_rgb.shape

    if (Num_selected_points==Num_points):
        Selected_x=[Coordinates_x[i]*float(Width1) for i in range(Num_points)]
        Selected_y = [Coordinates_y[i] * float(Height1) for i in range(Num_points)]
    else:
        Selected_x = [0]*Num_selected_points
        Selected_y = [0]*Num_selected_points
        Unselected_x = [0] * (Num_points-Num_selected_points)
        Unselected_y = [0] * (Num_points-Num_selected_points)
        Error = [0.0] * (Num_points-Num_selected_points)

        Pos_selected=0
        Pos_unselected=0
        for i in range(Num_points):
            if Valid[i]==True:
                Selected_x[Pos_selected]=int(round(Coordinates_x[i]*float(Width1-1)))
                Selected_y[Pos_selected]=int(round(Coordinates_y[i]*float(Height1-1)))
                Pos_selected=Pos_selected+1
            else:
                Unselected_x[Pos_unselected] = int(round(Coordinates_x[i] * float(Width1 - 1)))
                Unselected_y[Pos_unselected] = int(round(Coordinates_y[i] * float(Height1 - 1)))
                Error[Pos_unselected]=random.random()*0.5
                Pos_unselected = Pos_unselected + 1

    if Height1>Width1:
        Height_aux=Output_max_size/dpi;
        Width_aux=Height_aux*(float(Height1) / float(Width1))
    else:
        Width_aux = Output_max_size / dpi;
        Height_aux = Width_aux * (float(Height1) / float(Width1))

    fig, ax = plt.subplots(1, 1, figsize=(Width_aux,Height_aux))

    img_plot = plt.imshow(Image_rgb)

    # remove ticks and labels
    if Draw_ticks_labels==False:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    if Draw_mode == "POSITIONS":
        plt.scatter(Selected_x, Selected_y, c='red', marker='x')
    elif Draw_mode == "SELECTED_POSITIONS":
        plt.scatter(Selected_x, Selected_y, c='black', marker='o')
    elif Draw_mode == "UNSELECTED_POSITIONS_WITH_ERROR":
        plt.scatter(Unselected_x, Unselected_y, c=Error, marker='s', vmin=0, vmax=1, cmap=Map)

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.25)
        plt.colorbar(cm.ScalarMappable(cmap=Map), ax=ax, cax=cax)

    if Save == False:
        plt.show()
    else:
        plt.savefig(File_name,bbox_inches='tight',transparent=True,pad_inches = 0,dpi=dpi)
        plt.close()


def main():
    print("Working")

    if os.path.isdir(Output_folder) == False:
        os.mkdir(Output_folder)

    if Input_image == 'ORIGINAL':
        Image_rgb = io.imread("input_data/vis_visible.png")
    elif Input_image == 'ORIGINAL_UNSATURATED':
        Image_rgb = io.imread("input_data/vis_visible_small_desaturated.png")
    elif Input_image == 'MAP':
        Image_rgb = io.imread("input_data/interpolation.png")

    # positions
    Width, Height, Vec_pos, Coordinates_x, Coordinates_y = read_positions("input_data/positions.txt")

    random.seed(2022)

    if Draw_mode=='ORIGINAL':
        Num_selected_points = Num_points
        File_name = "original"
    elif Draw_mode == 'POSITIONS':
        Num_selected_points = Num_points
        File_name = "original_plus_positions"
    elif Draw_mode == 'SELECTED_POSITIONS':
        Num_selected_points = 50
        File_name = "selected_positions"
    elif Draw_mode == 'UNSELECTED_POSITIONS_WITH_ERROR':
        Num_selected_points = 50
        File_name = "unselected_positions"

    File_name = Output_folder + '/' + File_name + File_type

    print('File name=',File_name)

    Valid, Selected_pos=get_random_positions(Num_selected_points, Num_points)

    draw_result(File_name,Image_rgb,Num_selected_points,Valid,Coordinates_x,Coordinates_y)

if __name__ == "__main__":
    main()
