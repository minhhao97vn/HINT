import copy
import pickle

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

mean_cifar10 = torch.tensor((0.4914, 0.4822, 0.4465))[:, None, None]
std_cifar10 = torch.tensor((0.2023, 0.1994, 0.2010))[:, None, None]


class CustomCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, split_train=True, download=True, augmentation=None, selected_indices=None,
                 transform=transforms.ToTensor()):
        super(CustomCIFAR10, self).__init__(root=root, train=train, download=download, transform=transform)
        self.poison_samples = None
        self.selected_indices = None
        self.gamma = None
        self.noise_indices = None
        self.noise_deltas = None
        self.poison_indices = None
        self.poison_deltas = None
        self.augmentation = augmentation
        self.to_tensor = transforms.ToTensor()
        if train:
            torch.manual_seed(1)
            indices = torch.randperm(len(self.data), generator=torch.default_generator).tolist()

            if split_train:
                train_data = self.data[indices[:49000]]
                train_targets = np.array(self.targets)[indices[:49000]]
                self.data = train_data
                self.targets = train_targets
            else:
                val_data = self.data[indices[49000:]]
                val_targets = np.array(self.targets)[indices[49000:]]
                self.data = val_data
                self.targets = val_targets

    def set_subset(self, selected_indices):
        self.selected_indices = selected_indices
        self.data = self.data[selected_indices]
        self.targets = self.targets[selected_indices]

    def set_poison_delta(self, poison_indices, poison_deltas):
        self.poison_deltas = poison_deltas
        self.poison_indices = poison_indices

        for idx, img in enumerate(self.data):
            if idx in self.poison_indices:
                img = self.to_tensor(img)
                img = self.transform(img)
                # Add poison delta
                img = img + self.poison_deltas[self.poison_indices.index(idx)]

                # Denormalize
                img = torch.clamp(img * std_cifar10 + mean_cifar10, 0, 1)
                # Convert to [0,255]
                img = img.mul(255).permute(1, 2, 0).to('cpu', torch.uint8)
                self.data[idx] = img.numpy()

    def set_poison_sample(self, poison_indices, poison_samples):
        self.poison_samples = poison_samples
        self.poison_indices = poison_indices

        for idx, img in enumerate(self.data):
            if idx in self.poison_indices:
                new_img = self.poison_samples[self.poison_indices.index(idx)]
                self.data[idx] = np.transpose(np.clip(new_img * 255, 0, 255), (1, 2, 0))

    def update_poison_data_ver1(self, poison_data):
        self.data = poison_data['poisoned_images'].mul(255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        self.targets = poison_data['poison_targets'].numpy()

        print("training max: ", poison_data['poisoned_images'].max())
        print("training min: ", poison_data['poisoned_images'].min())

        # self.data = poison_data[0].mul(255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        # self.targets = poison_data[1].numpy()

        print(self.data.shape)

    def update_poison_data_ver2(self, poison_data):
        # self.data = poison_data['poison_images'].mul(255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        self.data = np.transpose(poison_data['poison_images'] * 255, (0, 2, 3, 1)).astype(np.uint8)
        print(self.data.shape)
        self.targets = poison_data['poison_targets']

    def set_healthy_noise(self, noise_indices, noise_deltas, gamma):
        self.noise_deltas = noise_deltas
        self.noise_indices = noise_indices
        self.gamma = gamma
        print(f'gamma: {self.gamma}')
        print('Use sign function')

        for idx, img in enumerate(self.data):
            if idx in self.noise_indices:
                img = self.to_tensor(img)
                img = img - self.gamma * torch.sign(self.noise_deltas[self.noise_indices.index(idx)][0])
                img = torch.clamp(img, 0, 1)
                img = img.mul(255).permute(1, 2, 0).to('cpu', torch.uint8)
                self.data[idx] = img.numpy()

    def set_healthy_noise_no_sign(self, noise_indices, noise_deltas, gamma):
        self.noise_deltas = noise_deltas
        self.noise_indices = noise_indices
        self.gamma = gamma
        print(f'gamma: {self.gamma}')
        print('Do not use sign function')

        for idx, img in enumerate(self.data):
            if idx in self.noise_indices:
                img = self.to_tensor(img)

                # img = self.transform(img)

                img = img - self.gamma * self.noise_deltas[self.noise_indices.index(idx)][0]
                img = torch.clamp(img, 0, 1)
                img = img.mul(255).permute(1, 2, 0).to('cpu', torch.uint8)
                self.data[idx] = img.numpy()

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        img = self.to_tensor(img)

        # only doing to_tensor and normalization, augmentation is separated
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            img = self.transform(img)

        if self.augmentation is not None:
            img = self.augmentation(img)

        return img, target

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index


class CustomCIFAR10V2(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, split_train=True, download=True, augmentation=None, selected_indices=None,
                 transform=transforms.ToTensor()):
        super(CustomCIFAR10V2, self).__init__(root=root, train=train, download=download, transform=transform)
        self.poison_samples = None
        self.selected_indices = None
        self.gamma = None
        self.noise_indices = None
        self.noise_deltas = None
        self.poison_indices = None
        self.poison_deltas = None
        self.augmentation = augmentation
        self.to_tensor = transforms.ToTensor()
        if train:
            torch.manual_seed(1)
            indices = torch.randperm(len(self.data), generator=torch.default_generator).tolist()

            if split_train:
                train_data = self.data[indices[:49000]]
                train_targets = np.array(self.targets)[indices[:49000]]
                self.data = train_data
                self.targets = train_targets
            else:
                val_data = self.data[indices[49000:]]
                val_targets = np.array(self.targets)[indices[49000:]]
                self.data = val_data
                self.targets = val_targets
            self.original_data = copy.deepcopy(self.data)

    def set_subset(self, selected_indices):
        self.selected_indices = selected_indices
        self.data = self.data[selected_indices]
        self.original_data = copy.deepcopy(self.data)
        self.targets = self.targets[selected_indices]

    def set_poison_delta(self, poison_indices, poison_deltas):
        self.poison_deltas = poison_deltas
        self.poison_indices = poison_indices

        for idx, img in enumerate(self.data):
            if idx in self.poison_indices:
                ori_img = copy.deepcopy(img)
                img = self.to_tensor(img)
                img = img.mul(255) + self.poison_deltas[self.poison_indices.index(idx)].mul(255) * std_cifar10

                # Denormalize
                img = torch.clamp(img, 0, 255)
                self.data[idx] = img.permute(1, 2, 0).numpy()
                # print('max GM noise', torch.max(img.permute(1,2,0) - ori_img))
                # print('min GM noise', torch.max(img.permute(1,2,0) - ori_img))

        self.original_data = copy.deepcopy(self.data)

    def set_poison_sample(self, poison_indices, poison_samples):
        self.poison_samples = poison_samples
        self.poison_indices = poison_indices

        for idx, img in enumerate(self.data):
            if idx in self.poison_indices:
                new_img = self.poison_samples[self.poison_indices.index(idx)]
                self.data[idx] = np.transpose(np.clip(new_img * 255, 0, 255), (1, 2, 0))
        self.original_data = copy.deepcopy(self.data)

    def update_poison_data_ver1(self, poison_data):
        self.data = poison_data['poisoned_images'].mul(255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        self.targets = poison_data['poison_targets'].numpy()

        # print("training max: ", poison_data['poisoned_images'].max())
        # print("training min: ", poison_data['poisoned_images'].min())

        # self.data = poison_data[0].mul(255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        # self.targets = poison_data[1].numpy()
        self.original_data = copy.deepcopy(self.data)

    def update_poison_data_ver2(self, poison_data):
        # self.data = poison_data['poison_images'].mul(255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        self.data = np.transpose(poison_data['poison_images'] * 255, (0, 2, 3, 1)).astype(np.uint8)
        print(self.data.shape)
        self.targets = poison_data['poison_targets']
        self.original_data = copy.deepcopy(self.data)

    def set_healthy_noise(self, noise_indices, noise_deltas, gamma, beta):
        self.noise_deltas = noise_deltas
        self.noise_indices = noise_indices
        self.gamma = gamma
        # print(f'gamma: {self.gamma}')
        # print('Use sign function')

        max_hin_noise = -1
        min_hin_noise = 1000

        for idx, img in enumerate(self.data):
            if idx in self.noise_indices:
                img = self.to_tensor(img)
                original_img = self.to_tensor(self.original_data[idx])
                noise = original_img - img
                noise = noise + self.gamma * torch.sign(self.noise_deltas[self.noise_indices.index(idx)][0])
                noise = torch.clamp(noise, -beta / 255, beta / 255)

                if max_hin_noise < noise.max():
                    max_hin_noise = noise.max()
                if min_hin_noise > noise.min():
                    min_hin_noise = noise.min()

                img = torch.clamp(original_img - noise, 0, 1)
                img = img.mul(255).permute(1, 2, 0).to('cpu', torch.uint8)
                self.data[idx] = img.numpy()
        # print("Max HIN noise this epoch: ", max_hin_noise)
        # print("Min HIN noise this epoch: ", min_hin_noise)

    def set_healthy_noise_no_sign(self, noise_indices, noise_deltas, gamma, beta, epoch):
        self.noise_deltas = noise_deltas
        self.noise_indices = noise_indices
        self.gamma = gamma
        # print(f'gamma: {self.gamma}')
        # print('No use sign function')

        max_hin_noise = -1
        min_hin_noise = 1000

        selected_images = []
        healthy_noise = []
        images_idx = []

        for idx, img in enumerate(self.data):
            if idx in self.noise_indices:
                img = self.to_tensor(img)
                original_img = self.to_tensor(self.original_data[idx])

                noise = original_img - img
                noise = noise + self.gamma * self.noise_deltas[self.noise_indices.index(idx)][0] #* std_cifar10
                noise = torch.clamp(noise, -beta / 255, beta / 255)

                if max_hin_noise < noise.max():
                    max_hin_noise = noise.max()
                if min_hin_noise > noise.min():
                    min_hin_noise = noise.min()

                img = torch.clamp(original_img - noise, 0, 1)
                img = img.mul(255).permute(1, 2, 0).to('cpu', torch.uint8)

                selected_images.append(self.original_data[idx])
                healthy_noise.append(noise.numpy())
                images_idx.append(idx)

                self.data[idx] = img.numpy()
        # print("Max HIN noise this epoch: ", max_hin_noise)
        # print("Min HIN noise this epoch: ", min_hin_noise)
        save_data = {"images": selected_images, "noise": healthy_noise, "ids": images_idx}
        # with open(f'GM_trial_9_new_05_data_at_epoch_{epoch}.pkl', 'wb') as handle:
        #     pickle.dump(save_data, handle, pickle.HIGHEST_PROTOCOL)
        # print(f"Save selected images and noise to data_at_epoch_{epoch}.pkl")

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        img = self.to_tensor(img)

        # only doing to_tensor and normalization, augmentation is separated
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            img = self.transform(img)

        if self.augmentation is not None:
            img = self.augmentation(img)

        return img, target

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index


class CustomMNIST(torchvision.datasets.MNIST):
    def __init__(self, root, train=True, split_train=True, download=True, selected_indices=None,
                 transform=transforms.ToTensor()):
        super(CustomMNIST, self).__init__(root=root, train=train, download=download, transform=None)
        self.poison_samples = None
        self.selected_indices = None
        self.gamma = None
        self.noise_indices = None
        self.noise_deltas = None
        self.poison_indices = None
        self.poison_deltas = None
        self.to_tensor = transforms.ToTensor()
        if train:
            torch.manual_seed(1)
            indices = torch.randperm(len(self.data), generator=torch.default_generator).tolist()

            if split_train:
                train_data = self.data[indices[:59000]]
                train_targets = np.array(self.targets)[indices[:59000]]
                self.data = train_data.numpy()
                self.targets = train_targets
            else:
                val_data = self.data[indices[59000:]]
                val_targets = np.array(self.targets)[indices[59000:]]
                self.data = val_data.numpy()
                self.targets = val_targets
            self.original_data = copy.deepcopy(self.data)

    def set_subset(self, selected_indices):
        self.selected_indices = selected_indices
        self.data = self.data[selected_indices]
        self.targets = self.targets[selected_indices]

    def set_poison_delta(self, poison_indices, poison_deltas):
        self.poison_deltas = poison_deltas
        self.poison_indices = poison_indices

        for idx, img in enumerate(self.data):
            if idx in self.poison_indices:
                img = self.to_tensor(img)
                img = self.transform(img)
                # Add poison delta
                img = img + self.poison_deltas[self.poison_indices.index(idx)]
                # Denormalize
                img = torch.clamp(img * std_cifar10 + mean_cifar10, 0, 1)
                # Convert to [0,255]
                img = img.mul(255).permute(1, 2, 0).to('cpu', torch.uint8)
                self.data[idx] = img.numpy()

    def set_poison_sample(self, poison_indices, poison_samples):
        self.poison_samples = poison_samples
        self.poison_indices = poison_indices

        for idx, img in enumerate(self.data):
            if idx in self.poison_indices:
                new_img = self.poison_samples[self.poison_indices.index(idx)]
                self.data[idx] = np.transpose(np.clip(new_img * 255, 0, 255), (1, 2, 0))

    def update_poison_data_ver1(self, poison_data):
        # self.data = poison_data['poisoned_images'].mul(255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        self.data = poison_data['poisoned_images'].mul(255).to('cpu', torch.uint8).numpy()
        self.targets = poison_data['poison_targets'].numpy()

        # print("training max: ", poison_data['poisoned_images'].max())
        # print("training min: ", poison_data['poisoned_images'].min())

        # self.data = poison_data[0].mul(255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        # self.targets = poison_data[1].numpy()

        self.original_data = copy.deepcopy(self.data)

    def set_healthy_noise(self, noise_indices, noise_deltas, gamma):
        self.noise_deltas = noise_deltas
        self.noise_indices = noise_indices
        self.gamma = gamma
        print(f'gamma: {self.gamma}')

        # Need to update later
        for idx, img in enumerate(self.data):
            if idx in self.noise_indices:
                img = self.to_tensor(img)
                img = img - self.gamma * torch.sign(self.noise_deltas[self.noise_indices.index(idx)][0])
                img = torch.clamp(img, 0, 1)
                img = img.mul(255).permute(1, 2, 0).to('cpu', torch.uint8)
                self.data[idx] = img.numpy()

    # Need edit
    def set_healthy_noise_no_sign(self, noise_indices, noise_deltas, gamma, beta, epoch):
        self.noise_deltas = noise_deltas
        self.noise_indices = noise_indices
        self.gamma = gamma
        print(f'gamma: {self.gamma}')
        print('Do not use sign function')

        max_hin_noise = -1
        min_hin_noise = 1000

        selected_images = []
        healthy_noise = []
        images_idx = []

        for idx, img in enumerate(self.data):
            if idx in self.noise_indices:
                img = self.to_tensor(img)
                original_img = self.to_tensor(self.original_data[idx])

                noise = original_img - img
                noise = noise + self.gamma * self.noise_deltas[self.noise_indices.index(idx)][0]
                noise = torch.clamp(noise, -beta / 255, beta / 255)

                if max_hin_noise < noise.max():
                    max_hin_noise = noise.max()
                if min_hin_noise > noise.min():
                    min_hin_noise = noise.min()

                img = torch.clamp(original_img - noise, 0, 1)
                img = img.mul(255).to('cpu', torch.uint8)

                selected_images.append(self.original_data[idx])
                healthy_noise.append(noise.numpy())
                images_idx.append(idx)

                self.data[idx] = img.numpy()
        print("Max HIN noise this epoch: ", max_hin_noise)
        print("Min HIN noise this epoch: ", min_hin_noise)
        save_data = {"images": selected_images, "noise": healthy_noise, "ids": images_idx}
        with open(f'Clean_MNIST_untgt_healthy_data_at_epoch_{epoch}_eps_{beta}_v2.pkl', 'wb') as handle:
            pickle.dump(save_data, handle, pickle.HIGHEST_PROTOCOL)
        print(f"Save selected images and noise to data_at_epoch_{epoch}.pkl")

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        img = self.to_tensor(img)

        # only doing to_tensor and normalization, augmentation is separated
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index
