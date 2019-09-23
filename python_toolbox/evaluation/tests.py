from setup import *
import numpy as np
import open3d as o3d


scene = "Ignatius"

cropfile = DATASET_DIR + scene + '/' + scene + '.json'
vol = o3d.visualization.read_selection_polygon_volume(cropfile)

o3d.visualization.draw_geometries([vol])