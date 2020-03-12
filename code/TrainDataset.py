import random
import sys
from glob import glob

from PIL import Image
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """
    All datasets are subclasses of torch.utils.data.Dataset i.e, they have __getitem__ and __len__ methods implemented.
    Hence, they can all be passed to a torch.utils.data.DataLoader which can load multiple samples
    parallelly using torch.multiprocessing workers
    """
    def __init__(self, imageFolderDataset, relationships, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.relationships = relationships  # choose either train or val dataset to use
        self.transform = transform
        self.delim = '\\' if sys.platform.startswith('win') else "/"

    def __getitem__(self, index):
        # Returns two images and whether they are related.
        # for each relationship in train_relationships.csv,
        # the first img comes from first row, and the second is either specially chosen related person or randomly chosen non-related person
        img0_info = self.relationships[index][0]
        img0_path = glob(str(self.imageFolderDataset.root / img0_info) + self.delim + "*.jpg")
        img0_path = random.choice(img0_path)

# is it better to do that in advance?
        cand_relationships = [x for x in self.relationships if
                              x[0] == img0_info or x[1] == img0_info]  # found all candidates related to person in img0
        if cand_relationships == []:  # in case no relationship is mentioned. But it is useless here because I choose the first person line by line.
            relative_label = 0 # no matter what image we feed in, it is not related...
        else:
            relative_label = random.randint(0, 1)    # 50% of sampling a relative

        if relative_label == 1:  # 1 means related, and 0 means non-related.
            img1_info = random.choice(cand_relationships)  # choose the second person from related relationships
            if img1_info[0] != img0_info:
                img1_info = img1_info[0]
            else:
                img1_info = img1_info[1]
            img1_path = glob(str(self.imageFolderDataset.root / img1_info) + self.delim + "*.jpg")  # randomly choose a img of this person
            img1_path = random.choice(img1_path)

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

        if self.transform is not None:  # I think the transform is essential if you want to use GPU, because you have to trans data to tensor first.
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, relative_label  # the returned data from dataloader is img=[batch_size,channels,width,length], relative_label=[batch_size,label]

    def __len__(self):
        return len(self.relationships)  # essential for choose the num of data in one epoch