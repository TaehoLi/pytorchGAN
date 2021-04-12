import os
import random
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from network import nz, weights_init, Generator, Discriminator

# manualSeed = random.randint(1, 10000) # use if you want new results
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
data_root = "../data/CelebA"
batch_size = 1024
image_size = 64  # Spatial size of training images. All images will be resized to this size using a transformer.
num_epochs = 5

dataset = dset.ImageFolder(root=data_root,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create network
netG = Generator().to(device)
netD = Discriminator().to(device)
# Apply the weights_init function to randomly initialize all weights to mean=0, std=0.2.
netG.apply(weights_init)
netD.apply(weights_init)

# Initialize BCELoss function
criterion = torch.nn.BCELoss()

# Create batch of latent vectors that we will use to visualize the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Lists to keep track of progress
G_losses = []
D_losses = []

folder_name = os.path.basename(os.path.splitext(__file__)[0])
path = "../results/" + folder_name
if not os.path.isdir(path):
    os.mkdir(path)
if __name__ == '__main__':
    print("Start Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            ###########################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            # Train with all-real batch
            netD.zero_grad()
            b_size = data[0].size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(data[0].to(device)).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################

            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Output training stats
            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 30 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img = vutils.make_grid(fake, padding=2, normalize=True)
                plt.figure(figsize=(8, 8))
                plt.axis("off")
                plt.imshow(np.transpose(img, (1, 2, 0)))
                plt.savefig('{0}/{1}_{2}.png'.format(path, str(epoch), str(i)))
                plt.clf()

    # Visualization
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("{0}/network loss.png".format(path))
    plt.clf()
