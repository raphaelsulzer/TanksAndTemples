# ----------------------------------------------------------------------------
# -                   TanksAndTemples Website Toolbox                        -
# -                    http://www.tanksandtemples.org                        -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2017
# Arno Knapitsch <arno.knapitsch@gmail.com >
# Jaesik Park <syncle@gmail.com>
# Qian-Yi Zhou <Qianyi.Zhou@gmail.com>
# Vladlen Koltun <vkoltun@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ----------------------------------------------------------------------------
#
# This python script is for downloading dataset from www.tanksandtemples.org
# The dataset has a different license, please refer to
# https://tanksandtemples.org/license/

# this script requires Open3D python binding
# please follow the intructions in setup.py before running this script.
import numpy as np
import sys

from setup import *
from registration import *
from evaluation import *
from util import *
from plot import *
import open3d as o3d
import argparse

# DATASET_DIR = "/Users/Raphael/Downloads/"

# OPEN3D_BUILD_PATH = "/home/raphael/PhD/cpp/Open3D/build/"
# OPEN3D_PYTHON_LIBRARY_PATH = "/home/raphael/PhD/cpp/Open3D/build/lib/Pyhton/"
# OPEN3D_EXPERIMENTAL_BIN_PATH = "/home/raphael/PhD/cpp/Open3D/build/bin/examples/"
# import sys
# sys.path.append(OPEN3D_PYTHON_LIBRARY_PATH)


def run_evaluation(args):

	DATASET_DIR = args.DATASET_DIR + args.directory

	# DATASET_DIR = "/mnt/a53b45cf-0ac9-41e5-b312-664d1219ca09/raphael/tanksAndTemples/"
	# DATASET_DIR = "/home/rsulzer/PhD/data/tanksAndTemples/"
	DATASET_DIR = "/Users/Raphael/Library/Mobile Documents/com~apple~CloudDocs/Studium/PhD/Paris/data/learningData/"
	DATASET_DIR = "/Users/Raphael/Library/Mobile Documents/com~apple~CloudDocs/Studium/PhD/Paris/data/tanksAndTemples/"


	scenes_tau_dict = {
		"Barn": 0.01,
		"Caterpillar": 0.005,
		"Church": 0.025,
		"Courthouse": 0.025,
		"Ignatius": 0.003,
		"Ignatius30": 0.003,
		"Meetingroom": 0.01,
		"Truck": 0.005,
	}
	scene = args.filename

	print("")
	print("===========================")
	print("Evaluating %s" % scene)
	print("===========================")
	dTau = scenes_tau_dict[scene]

	# put the crop-file, the GT file, the COLMAP SfM log file and
	# the alignment of the according scene in a folder of
	# the same scene name in the DATASET_DIR
	dirname = DATASET_DIR + scene + "/"
	# gt_filen = dirname + scene + "_" + args.ground_truth +  '_gt' +'.ply'
	gt_filen = dirname + scene + '.ply'

	colmap_ref_logfile = dirname + scene + '_COLMAP_SfM.log'
	alignment = dirname + scene + '_trans.txt'
	cropfile = dirname + scene + '.json'

	mvs_outpath = DATASET_DIR + scene + '/evaluation'
	# make_dir(mvs_outpath)

	###############################################################
	# User input files:
	# SfM log file and pointcoud of your reconstruction comes here.
	# as an example the COLMAP data will be used, but the script
	# should work with any other method as well
	###############################################################
	if(args.mine):
		new_logfile = dirname + scene + "_COLMAP_SfM_mine.log"
	else:
		new_logfile = dirname + scene + "_COLMAP_SfM.log"

	if(args.ground_truth == 'poisson'):
		reconstruction = DATASET_DIR + scene + '/' + scene + "_" + args.ground_truth + "_" + args.reconstruction + "_" + args.rw_string + "_sampled.ply"
		print(reconstruction)
	elif(args.ground_truth == 'lidar'):
		reconstruction = DATASET_DIR + scene + '/' + scene + "_" + args.reconstruction + ".ply"


	#Load reconstruction and according GT
	print("Reconstruction: ", reconstruction)
	pcd = o3d.io.read_point_cloud(reconstruction)
	print("Ground truth: ", gt_filen)
	gt_pcd = o3d.io.read_point_cloud(gt_filen)

	# check and stop if one of the ply input files is empty
	if(len(pcd.points) < 1):
		print("\nempty reconstruction file!")
		return

	if(len(gt_pcd.points) < 1):
		print("\nempty ground truth file!")
		return

	if(args.translate):
		gt_trans = np.loadtxt(alignment)
		traj_to_register = read_trajectory(new_logfile)
		gt_traj_col = read_trajectory(colmap_ref_logfile)

		trajectory_transform = trajectory_alignment(
				traj_to_register, gt_traj_col, gt_trans, scene)
	else:
		trajectory_transform = np.identity(4)

	# load crop file
	if(args.crop):
		crop_vol = o3d.visualization.read_selection_polygon_volume(cropfile)
	else:
		crop_vol = None

	### Refine alignment by using the actual GT and MVS pointclouds
	# big pointclouds will be downlsampled to this number to speed up alignment
	dist_threshold = dTau

	# Registration refinment in 3 iterations
	# if(args.register_and_crop):
	r2  = registration_vol_ds(pcd, gt_pcd,
			trajectory_transform, crop_vol, dTau, dTau*80, 20)
	r3  = registration_vol_ds(pcd, gt_pcd,
			r2.transformation, crop_vol, dTau/2.0, dTau*20, 20)
	r  = registration_unif(pcd, gt_pcd,
			r3.transformation, crop_vol, 2*dTau, 20)

	# Histogramms and P/R/F1
	plot_stretch = 5

	[precision, recall, fscore, edges_source, cum_source,
			edges_target, cum_target] = EvaluateHisto(
			pcd, gt_pcd, r.transformation, crop_vol, dTau/2.0, dTau,
			mvs_outpath, plot_stretch, scene, args)

	eva = [precision, recall, fscore]
	print("\n==============================")
	print("evaluation result : %s" % scene)
	print("==============================")
	print("distance tau : %.3f" % dTau)
	print("precision : %.4f" % eva[0])
	print("recall : %.4f" % eva[1])
	print("f-score : %.4f" % eva[2])
	print("==============================\n")

	# Plotting
	plot_graph(scene, fscore, dist_threshold, edges_source, cum_source,
			edges_target, cum_target, plot_stretch, mvs_outpath, args.reconstruction, args.rw_string)





if __name__ == "__main__":


	print("\nDATASET_DIR SET TO: ", DATASET_DIR)
	print("OPEN3D_EXPERIMENTAL_BIN_PATH: ", OPEN3D_EXPERIMENTAL_BIN_PATH)
	print("OPEN3D_PYTHON_LIBRARY_PATH", OPEN3D_PYTHON_LIBRARY_PATH)
	print("OPEN3D_BUILD_PATH", OPEN3D_BUILD_PATH)


	print("\n\nExample usages:")
	print("\n\tpython3 run.py Barn poisson cl -o 50")
	print("\n\tpython3 run.py Ignatius lidar colmap_mesh_sampled -r True")



	parser = argparse.ArgumentParser(description='Evaluate reconstruction.')
	parser.add_argument('directory')

	parser.add_argument('filename')

	parser.add_argument('ground_truth', type=str,
						help='the ground truth, e.g. lidar or poisson')
	parser.add_argument('reconstruction', type=str,
						help='the reconstruction, e.g. lrtcs, rt or colmap')
	# parser.add_argument('--scoring_type', type=str,
	# 					help='the scoring type')
	parser.add_argument('-m', '--mine', type=bool, default=False,
						help='my reconstruction')
	parser.add_argument('-o','--rw_string', type=str,
						help='the regularization weight')

	parser.add_argument('-t', '--translate', type=bool, default=False,
						help='apply transloation to reconstruction')
	parser.add_argument('-c', '--crop', type=bool, default=True,
						help='apply cropping to reconstruction')

	args = parser.parse_args()

	if(args.rw_string):
		print(args.rw_string)

	args.DATASET_DIR = DATASET_DIR

	run_evaluation(args)
