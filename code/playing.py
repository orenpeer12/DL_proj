from itertools import islice
from torch.utils.data import IterableDataset, Dataset, DataLoader

data = [0,1,2,3,4,5,6,7,8,9,10,11]

class MyIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)

iterable_dataset = MyIterableDataset(data)

loader = DataLoader(iterable_dataset, batch_size=4)
while True:
    for batch in loader:
        print(batch)
