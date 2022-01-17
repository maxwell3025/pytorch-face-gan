import torch
from torch import nn
import torch.utils.data
import PIL
from PIL import Image
import glob
import numpy as np
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
#compile data into numpy file
if not os.path.exists("data/database.npy"):
    print("generating database...")
    img_list = []
    for filepath in glob.glob("data/raw/*.png"):
        image = Image.open(filepath).convert("RGB").resize((64,64))
        img_list.append(np.array(image))
    np.save("data/database.npy", np.stack(img_list))
print("loading data...")
numpy_data = np.load("data/database.npy")
dataset = torch.utils.data.TensorDataset(torch.Tensor(numpy_data))
dataloader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( 100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( 64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( 64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
    def forward(self, input):
        return self.main(input)
gen_net = Generator().to(device)

print(gen_net)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)
dis_net = Discriminator().to(device)
print(dis_net)

example_noise = torch.randn(64, 100, 1, 1, device=device)

print(example_noise.shape)
def save_example(folder, number):
    with torch.no_grad():
        os.makedirs(f"out/{folder}", exist_ok=True)
        sample = gen_net(example_noise).detach().cpu().numpy()
        #convert from (B, C, W, H) to (B, W, H, C)
        sample = np.moveaxis(sample, 1,3)
        print(f"shape is {sample.shape}\ndata is {sample}")
        patch = Image.new("RGB", (512, 512))
        for x in range(8):
            for y in range(8):
                im = Image.fromarray((sample[x+y*8]*128+128).astype("uint8"), "RGB")
                patch.paste(im=im, box = (x*64, y*64))
        patch.save(f"out/{folder}/{number:04d}.png")

save_example("test", 1)