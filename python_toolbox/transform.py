from read_model import read_images_text, qvec2rotmat, rotmat2qvec
import sys, os, argparse
import numpy as np

def trans(args):

    image_file=os.path.join(args.user_dir,args.data_dir,args.scene,"dslr_calibration_undistorted_modified","images.txt")
    print("Loading: ", image_file)
    images = read_images_text(image_file)

    trans_file=os.path.join(args.user_dir,args.data_dir,args.scene,args.scene+"_trans.txt")
    trans_mat = np.loadtxt(trans_file)

    for key, im in images.items():
        qvec = im[1]
        r = qvec2rotmat(qvec)
        translation = im[2]
        w = np.zeros((4, 4))
        w[3, 3] = 1
        w[0:3, 0:3] = r
        w[0:3, 3] = translation
        A = np.matrix(w)*trans_mat
        new_qvec=rotmat2qvec(A[0:3,0:3])
        new_tvec=A[0:3,3]

        # now simply write back the new qvec and tvec


        f=5





    end=5


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Colmap SfM to openMVS project')

    parser.add_argument('-m','--machine', type=str, default="ign-laptop",
                        help='choose the machine, ign-laptop, cnes or enpc.')

    parser.add_argument('--user_dir', type=str, default="/home/adminlocal/PhD/",
                        help='the user folder, or PhD folder. default: /home/adminlocal/PhD/')
    parser.add_argument('-d', '--data_dir', type=str, default="data/TanksAndTemples/",
                        help='working directory which should include the different scene folders. default: data/ETH3D/')
    parser.add_argument('-s', '--scene', type=str, default="Ignatius",
                        help='on which scene to execute pipeline.')

    parser.add_argument('--openMVS_dir', type=str, default="cpp/openMVS_release/bin",
                        help='Indicate the openMVS binary directory, pointing to .../bin folder starting from user_dir')

    args = parser.parse_args()

    trans(args)