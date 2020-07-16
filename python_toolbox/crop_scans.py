import open3d as o3d
import sys, os
import argparse

def crop(args):

    cropfile = os.path.join(args.user_dir,args.data_dir,args.scene,args.scene+".json")

    crop_vol = o3d.visualization.read_selection_polygon_volume(cropfile)

    for scan in os.listdir(os.path.join(args.user_dir,args.data_dir,args.scene,"is_ori")):
        if(scan=="scanner_pos.txt"):
            continue
        pcd = o3d.io.read_point_cloud(os.path.join(args.user_dir,args.data_dir,args.scene,"is_ori",scan))
        pcd_crop = crop_vol.crop_point_cloud(pcd)
        o3d.io.write_point_cloud(os.path.join(args.user_dir,args.data_dir,args.scene,"is",scan), pcd_crop)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate reconstruction.')

    parser.add_argument('--user_dir', type=str, default="/home/adminlocal/PhD/",
                        help='the user folder, or PhD folder.')
    parser.add_argument('-d', '--data_dir', type=str, default="data/TanksAndTemples/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('-s', '--scene', type=str, default="Ignatius",
                        help='on which scene to execute pipeline.')

    args = parser.parse_args()

    crop(args)