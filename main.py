import argparse
import os
import numpy as np
import pandas as pd

import torch
from tqdm import tqdm
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import matplotlib.pyplot as plt
from torch.utils.tensorboard import writer, SummaryWriter
from torch.utils.data import DataLoader
from math import pi
from torchsummary import summary
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from datasets import sample_dataset
from models import Generator, Discriminator

import visualize
from stats import acf
import stylized_facts as sf
from fin_data import data_generator

class Trainer:
    NOISE_LENGTH = 50

    def __init__(self, generator, discriminator, gen_optimizer, discriminator_optimizer,
                 gp_weight=10, discriminator_iterations=5, print_every=200, use_cuda=True, checkpoint_frequency=200):
        self.g = generator
        self.d = discriminator
        self.g_opt = gen_optimizer
        self.d_opt = discriminator_optimizer
        self.gp_weight = gp_weight
        self.discriminator_iterations = discriminator_iterations
        self.print_every = print_every
        self.use_cuda = use_cuda
        self.checkpoint_frequency = checkpoint_frequency

        if self.use_cuda:
            self.g.cuda()
            self.d.cuda()

        self.losses = {'g': [], 'd': [], 'GP': [], 'gradient_norm': []}

        self.num_steps = 0

    def _discriminator_train_iteration(self, real_data):

        batch_size = real_data.size()[0]
        noise_shape = (batch_size, self.NOISE_LENGTH, 1) 
        generated_data = self.sample_generator(noise_shape) 

        real_data = Variable(real_data)

        if self.use_cuda:
            real_data = real_data.cuda()

        #data through the Discriminator
        d_real = self.d(real_data)
        d_generated = self.d(generated_data)

        #gradient penalty
        gradient_penalty = self._gradient_penalty(real_data, generated_data)
        self.losses['GP'].append(gradient_penalty.data.item())

        #total loss and optimize
        self.d_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()
        self.d_opt.step()

        self.losses['d'].append(d_loss.data.item())

    def _generator_train_iteration(self, data):
        self.g_opt.zero_grad()
        batch_size = data.size()[0]
        latent_shape = (batch_size, self.NOISE_LENGTH, 1)
        generated_data = self.sample_generator(latent_shape)


        #calculate loss and optimize
        d_generated = self.d(generated_data)
        g_loss = - d_generated.mean()
        g_loss.backward()
        self.g_opt.step()
        self.losses['g'].append(g_loss.data.item())

    def _gradient_penalty(self, real_data, generated_data):

        batch_size = real_data.size()[0]

        #calculate interpolation
        alpha = torch.rand(batch_size, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.use_cuda:
            interpolated = interpolated.cuda()

        #interpolated data through discriminator
        prob_interpolated = self.d(interpolated)

        #gradients of probabilities with respect to examples
        #gradients have shape (batch_size, num_channels, series length)
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.use_cuda
                               else torch.ones(prob_interpolated.size()), create_graph=True,
                               retain_graph=True)[0]
        

        #flatten to take the norm per example for every batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data.item())

        # add epsilon for stability
        eps = 1e-12

        #derivatives of the gradient close to 0 can cause problems because of square root
        #so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + eps)

        #return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader, epoch):
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            self._discriminator_train_iteration(data.float())
            #update generator once every discriminator_iterations
            if self.num_steps % self.discriminator_iterations == 0:
                self._generator_train_iteration(data)

            if i % self.print_every == 0:
                global_step = i + epoch * len(data_loader.dataset)
                writer.add_scalar('Losses/Discriminator', self.losses['d'][-1], global_step)
                writer.add_scalar('Losses/Gradient Penalty', self.losses['GP'][-1], global_step)
                writer.add_scalar('Gradient Norm', self.losses['gradient_norm'][-1], global_step)

                if self.num_steps > self.discriminator_iterations:
                    writer.add_scalar('Losses/Generator', self.losses['g'][-1], global_step)

    def train(self, data_loader, epochs, plot_training_samples=True, checkpoint=True):

        if checkpoint:
            path = os.path.join('checkpoints', checkpoint)
            state_dicts = torch.load(path, map_location=torch.device('cpu'))
            self.g.load_state_dict(state_dicts['g_state_dict'])
            self.d.load_state_dict(state_dicts['d_state_dict'])
            self.g_opt.load_state_dict(state_dicts['g_opt_state_dict'])
            self.d_opt.load_state_dict(state_dicts['d_opt_state_dict'])

        #define noise_shape
        #third dimension added for LSTM
        noise_shape = (1, self.NOISE_LENGTH, 1)
        noise_shape_1 = (1, 4000, 1) 

        if plot_training_samples==True:
            #fixed latents to see generation improvement during training
            fixed_latents = Variable(self.sample_latent(noise_shape_1))
            if self.use_cuda:
                fixed_latents = fixed_latents.cuda()

        for epoch in tqdm(range(epochs)):

            #sample different region of the latent distribution to check for mode collapse
            #if mode collapse keeps happening change lr
            dynamic_latents = Variable(self.sample_latent(noise_shape_1))
            if self.use_cuda:
                dynamic_latents = dynamic_latents.cuda()

            self._train_epoch(data_loader, epoch + 1)

            #checkpoint save
            if epoch % self.checkpoint_frequency == 0:
                torch.save({
                    'epoch': epoch,
                    'd_state_dict': self.d.state_dict(),
                    'g_state_dict': self.g.state_dict(),
                    'd_opt_state_dict': self.d_opt.state_dict(),
                    'g_opt_state_dict': self.g_opt.state_dict(),
                }, 'checkpoints/epoch_{}.pkl'.format(epoch))

            if plot_training_samples and (epoch % self.print_every == 0):
                self.g.eval()
                # Generate fake data using both fixed and dynamic latents
                fake_data_fixed_latents = self.g(fixed_latents).cpu().data
                fake_data_dynamic_latents = self.g(dynamic_latents).cpu().data

                plt.figure()
                plt.plot(fake_data_fixed_latents.numpy()[0])
                plt.savefig('training_samples/fixed_latents/series_epoch_{}.png'.format(epoch))
                plt.close()

                plt.figure()
                plt.plot(fake_data_dynamic_latents.numpy()[0])
                plt.savefig('training_samples/dynamic_latents/series_epoch_{}.png'.format(epoch))
                plt.close()

                self.g.train()
                np.save('./npy/%i_fixed_latents_generated_time_series.npy'%(epoch),fake_data_fixed_latents.numpy()[0].T)
                np.save('./npy/%i_dynamic_latents_generated_time_series.npy'%(epoch),fake_data_dynamic_latents.numpy()[0].T)
        torch.save(g, 'model_g')
        torch.save(d, 'model_d')

    def sample_generator(self, latent_shape):
        latent_samples = Variable(self.sample_latent(latent_shape))
        if self.use_cuda:
            latent_samples = latent_samples.cuda()

        return self.g(latent_samples)

    @staticmethod
    def sample_latent(shape):
        return torch.randn(shape)

    def sample(self, num_samples):
        generated_data = self.sample_generator(num_samples)
        return generated_data.data.cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='WGAN-FTS', usage='%(prog)s [options]')
    parser.add_argument('-e', '--epochs', type=int, dest='epochs', default=2001,
                        help='number of training epochs')
    parser.add_argument('-bs', '--batches', type=int, dest='batches', default=16,
                        help='number of batches per training iteration')
    parser.add_argument('-cp', '--checkpoint', type=str, dest='checkpoint', default=None,
                        help='checkpoint to use for a warm start')

    args = parser.parse_args()

    #instantiate Generator and Discriminator and initialize weights
    g = Generator()
    g_opt = torch.optim.RMSprop(g.parameters(), lr=0.00001)
    d = Discriminator()
    d_opt = torch.optim.RMSprop(d.parameters(), lr=0.00001)

    #extract data and create dataloader
    dataset = sample_dataset()
    dataloader = DataLoader(dataset, batch_size=args.batches, shuffle=False, num_workers=os.cpu_count())

    #instantiate Trainer
    trainer = Trainer(g, d, g_opt, d_opt, use_cuda=torch.cuda.is_available())

    #train model
    #Tensorboard writer
    writer = SummaryWriter(log_dir='/mnt/c/Users/Utente/Desktop/GAN-FTS/exp')

    trainer.train(dataloader, epochs=args.epochs, plot_training_samples=True, checkpoint=args.checkpoint)

    