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
import sys, os

from setup import *
from registration import *
from evaluation import *
from util import *
from plot import *
import open3d as o3d
import argparse


def run_evaluation(args):

	scenes_tau_dict = {
		# "Barn": 0.01,
		"Barn": 0.05,
		"Caterpillar": 0.005,
		"Church": 0.025,
		"Courthouse": 0.025,
		"Ignatius": 0.003,
		"Ignatius30": 0.003,
		"Meetingroom": 0.01,
		"Truck": 0.005,
	}

	print("")
	print("===========================")
	print("Evaluating %s at %.3f" % (args.scene, args.dTau))
	print("===========================")

	# put the crop-file, the GT file, the COLMAP SfM log file and
	# the alignment of the according scene in a folder of
	# the same scene name in the DATASET_DIR
	dir = os.path.join(args.user_dir, args.data_dir, args.scene, '')


	mvs_outpath = dir + 'evaluation'
	if(not os.path.isdir(mvs_outpath)):
		os.mkdir(mvs_outpath)

	###############################################################
	# User input files:
	# SfM log file and pointcoud of your reconstruction comes here.
	# as an example the COLMAP data will be used, but the script
	# should work with any other method as well
	###############################################################


	if(args.ground_truth == 'poisson'):
		reconstruction = dir + args.scene + "_" + args.ground_truth + "_" + args.perc_outliers + "_" + args.reconstruction + "_" + args.rw_string + "_sampled"
		gt_filen = dir + args.scene + '_poisson_sampled.ply'
	elif(args.ground_truth == 'lidar'):
		reconstruction = dir + args.scene + "_" + args.reconstruction
		reconstruction = dir + "densify_file"
		# gt_filen = dirname + scene + "_" + args.ground_truth + '_gt' + '.ply'
		gt_filen = dir + args.scene + '.ply'

	if(args.sampled):
		reconstruction += "_sampled"
	reconstruction += ".ply"


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
		colmap_ref_logfile = dir + args.scene + '_COLMAP_SfM.log'
		print("\nLoad ground truth trajectory: ", colmap_ref_logfile)
		gt_traj_col = read_trajectory(colmap_ref_logfile)

		new_logfile = dir + args.scene + "_" + args.reconstruction + "_SfM_mine.log"
		print("\nLoad estimated trajectory: ", new_logfile)
		traj_to_register = read_trajectory(new_logfile)

		alignment = dir + args.scene + '_trans.txt'
		print("Load trajectory alignment....\n")
		gt_trans = np.loadtxt(alignment)

		print("\nAlign trajectories: ")
		trajectory_transform = trajectory_alignment(
				traj_to_register, gt_traj_col, gt_trans, args.scene)
	else:
		trajectory_transform = np.identity(4)

	# load crop file
	if(args.crop):
		cropfile = dir + args.scene + '.json'
		print("\nLoad crop file: ", cropfile)
		crop_vol = o3d.visualization.read_selection_polygon_volume(cropfile)
	else:
		crop_vol = None

	### Refine alignment by using the actual GT and MVS pointclouds
	# big pointclouds will be downlsampled to dTau to speed up alignment

	# Registration refinment in 3 iterations
	# if(args.register_and_crop):
	print("\nRegistration refinment in 3 iterations...\n")
	r2  = registration_vol_ds(pcd, gt_pcd,
			trajectory_transform, crop_vol, args.dTau, args.dTau*80, 20)
	r3  = registration_vol_ds(pcd, gt_pcd,
			r2.transformation, crop_vol, args.dTau/2.0, args.dTau*20, 20)
	r  = registration_unif(pcd, gt_pcd,
			r3.transformation, crop_vol, 2*args.dTau, 20)


	# write final alignment matrix to file:
	new_alignment = dir + args.scene + '_new_trans.txt'
	np.savetxt(new_alignment, r.transformation)


	# Histogramms and P/R/F1
	plot_stretch = 5

	[precision, recall, fscore, edges_source, cum_source,
			edges_target, cum_target] = EvaluateHisto(
			pcd, gt_pcd, r.transformation, crop_vol, args.dTau/2.0, args.dTau,
			mvs_outpath, plot_stretch, args.scene, args)

	eva = [precision, recall, fscore]
	print("\n==============================")
	print("evaluation result : %s" % args.scene)
	print("==============================")
	print("distance tau : %.3f" % args.dTau)
	print("precision : %.4f" % eva[0])
	print("recall : %.4f" % eva[1])
	print("f-score : %.4f" % eva[2])
	print("==============================\n")

	# Plotting
	plot_graph(args.scene, fscore, args.dTau, edges_source, cum_source,
			edges_target, cum_target, plot_stretch, mvs_outpath, args.reconstruction, args.rw_string)





if __name__ == "__main__":

	# DATASET_DIR = "/home/adminlocal/PhD/data/TanksAndTemples/"
	# print("\nDATASET_DIR SET TO: ", DATASET_DIR)
	print("OPEN3D_EXPERIMENTAL_BIN_PATH: ", OPEN3D_EXPERIMENTAL_BIN_PATH)
	print("OPEN3D_PYTHON_LIBRARY_PATH", OPEN3D_PYTHON_LIBRARY_PATH)
	print("OPEN3D_BUILD_PATH", OPEN3D_BUILD_PATH)

	if(len(sys.argv)< 4):
		print("\n\nExample usages:")
		print("\n\tpython3 run.py -f Barn -g poisson -r cl -o 50")
		print("\n\tpython3 run.py -f Ignatius -g lidar -r colmap_mesh_sampled -t True")
		print("\n")



	parser = argparse.ArgumentParser(description='Evaluate reconstruction.')

	parser.add_argument('--user_dir', type=str, default="/home/adminlocal/PhD/",
						help='the user folder, or PhD folder.')
	parser.add_argument('-d', '--data_dir', type=str, default="data/benchmark/scan_example/tat",
						help='working directory which should include the different scene folders.')
	parser.add_argument('-s', '--scene', type=str, default="Ignatius",
						help='on which scene to execute pipeline.')

	parser.add_argument('-g','--ground_truth', type=str, default="lidar",
						help='the ground truth, e.g. lidar or poisson')
	parser.add_argument('-r','--reconstruction', type=str, default="COLMAP",
						help='the reconstruction, e.g. lrtcs, rt or colmap')

	parser.add_argument('-m', '--mine', type=int, default=1,
						help='my reconstruction')
	parser.add_argument('--sampled', type=bool, default=False,
						help='reconstruction sampled from mesh')
	parser.add_argument('-o','--rw_string', type=str, default="0",
						help='the regularization weight')
	parser.add_argument('-out','--perc_outliers', type=str, default="3.0",
						help='the regularization weight')

	parser.add_argument('-t', '--translate', type=int, default=1,
						help='apply transloation to reconstruction')
	parser.add_argument('-c', '--crop', type=int, default=1,
						help='apply cropping to reconstruction')

	parser.add_argument('--dTau', type=float, default=0.003,
						help='show f-score at dTau')

	args = parser.parse_args()



	run_evaluation(args)
