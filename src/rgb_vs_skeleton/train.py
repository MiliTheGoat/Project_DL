from rgb_vs_skeleton.model_CTR_GCN import CTRGCN
from rgb_vs_skeleton.model_X3D_M import build_x3d_m

from rgb_vs_skeleton.loaders import rgb_loader, skeleton_loader



def train():
    dataset = MyDataset("data/raw")
    model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()
