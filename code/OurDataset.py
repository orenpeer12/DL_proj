import random
import sys
from glob import glob
import cv2
from PIL import Image
import PIL
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
import numpy as np


class OurDataset(Dataset):
    """
    All datasets are subclasses of torch.utils.data.Dataset i.e, they have __getitem__ and __len__ methods implemented.
    Hence, they can all be passed to a torch.utils.data.DataLoader which can load multiple samples
    parallelly using torch.multiprocessing workers
    """
    def __init__(self, imageFolderDataset, relationships, familise_trees, ppl, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.relationships = relationships  # choose either train or val dataset to use
        self.transform = transform
        self.delim = '\\' if sys.platform.startswith('win') else "/"
        self.familise_trees = familise_trees
        self.ppl = ppl

    def __getitem__(self, index):
        if not isinstance(index, int):
            p0 = index[0]
            p1 = index[1]

            family0, person0 = self.ppl[p0].split(self.delim)
            family1, person1 = self.ppl[p1].split(self.delim)

            relative_label = (self.ppl[p0], self.ppl[p1]) in self.relationships

            img0_path = random.choice(self.familise_trees[family0][person0])
            img1_path = random.choice(self.familise_trees[family1][person1])
        else:
            # Returns two images and whether they are related.
            # for each relationship in train_relationships.csv,
            # the first img comes from first row, and the second is either specially chosen related person or randomly chosen non-related person
            img0_info = random.choice(self.relationships[index])
            family0, person0 = img0_info.split(self.delim)
            # img0_path = glob(str(self.imageFolderDataset.root / img0_info) + self.delim + "*.jpg")
            img0_path = random.choice(self.familise_trees[family0][person0])

            # cand_relationships = self.familise_trees[img0_info]
            cand_relationships = [x for x in self.relationships if
                                  x[0] == img0_info or x[1] == img0_info]  # found all candidates related to person in img0
            if cand_relationships == []:  # in case no relationship is mentioned. But it is useless here because I choose the first person line by line.
                relative_label = 0  # no matter what image we feed in, it is not related...
            else:
                relative_label = random.randint(0, 1)    # 50% of sampling a relative

            if relative_label == 1:  # 1 means related, and 0 means non-related.
                img1_info = random.choice(cand_relationships)  # choose the second person from related relationships
                if img1_info[0] != img0_info:
                    img1_info = img1_info[0]
                else:
                    img1_info = img1_info[1]
                family1, person1 = img1_info.split(self.delim)
                assert family0 == family1
                img1_path = random.choice(self.familise_trees[family1][person1])
            else:  # 0 means non-related
                randChoose = True  # in case the chosen person is related to first person
                while randChoose:
                    img1_path = random.choice(self.imageFolderDataset.imgs)[0]
                    img1_info = img1_path.split(self.delim)[-3] + self.delim + img1_path.split(self.delim)[-2]
                    randChoose = False
                    for x in cand_relationships:  # if so, randomly choose another person
                        if x[0] == img1_info or x[1] == img1_info:
                            randChoose = True
                            break

        img0 = Image.open(img0_path)
        img1 = Image.open(img1_path)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0.float(), img1.float(), int(relative_label)  # the returned data from dataloader is img=[batch_size,channels,width,length], relative_label=[batch_size,label]

    def __len__(self):
        return len(self.relationships)  # essential for choose the num of data in one epoch


class TestDataset(Dataset):

    def __init__(self, df, root_dir, transform=None):
        self.relations = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.relations)

    def __getpair__(self, idx):
        # pair = self.root_dir + self.relations.iloc[idx, 2], \
        #        self.root_dir + self.relations.iloc[idx, 3]
        pair = self.relations.iloc[idx].img_pair.split('-')
        pair = [self.root_dir / p for p in pair]
        return pair

    def __getlabel__(self, idx):
        return self.relations.iloc[idx, 4]

    def __getitem__(self, idx):
        pair = self.__getpair__(idx)

        img0 = Image.open(pair[0])
        img1 = Image.open(pair[1])
        #         img0 = img0.convert("L")
        #         img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return idx, img0, img1


class OurSampler(BatchSampler):
    def __init__(self, data, batch_size):
        self.dataset = data
        self.num_samples = len(self.dataset)
        self.batch_size = batch_size

    def __iter__(self):
        rand_relat = random.sample(self.dataset.relationships, self.batch_size//2)

        # get the indices of ppl
        batch = []
        for rel in rand_relat:
            batch.append((self.dataset.ppl.index(rel[0]), self.dataset.ppl.index(rel[1])))

        while len(batch) < self.batch_size:
            p1_idx = random.randint(0, len(self.dataset.ppl) - 1)
            p1_ppl = self.dataset.ppl[p1_idx]
            p2_idx = random.randint(0, len(self.dataset.ppl) - 1)
            p2_ppl = self.dataset.ppl[p2_idx]

            if p1_idx != p2_idx and (p1_ppl, p2_ppl) not in self.dataset.relationships \
                    and (p2_ppl, p1_ppl) not in self.dataset.relationships:
                batch.append((p1_idx, p2_idx))

        # print(batch)
        yield batch

    def __len__(self):
        return self.num_samples


def create_datasets(folder_dataset, train_pairs, val_pairs, image_transforms, train_family_persons_tree,
                    val_family_persons_tree, train_ppl, val_ppl, hyper_params, NUM_WORKERS):

    trainset = OurDataset(imageFolderDataset=folder_dataset,
                          relationships=train_pairs,
                          transform=image_transforms["train"],
                          familise_trees=train_family_persons_tree,
                          ppl=train_ppl)


    train_batch_sampler = OurSampler(trainset, hyper_params["BATCH_SIZE"])
    trainloader = DataLoader(trainset,
                             shuffle=False,
                             batch_sampler=train_batch_sampler,
                             num_workers=NUM_WORKERS,
                             drop_last=False,
                             batch_size=1)

    valset = OurDataset(imageFolderDataset=folder_dataset,
                        relationships=val_pairs,
                        transform=image_transforms["valid"],
                        familise_trees=val_family_persons_tree,
                        ppl=val_ppl)

    valloader = DataLoader(valset,
                           shuffle=True,
                           num_workers=NUM_WORKERS,
                           batch_size=hyper_params["BATCH_SIZE"])

    return trainloader, valloader