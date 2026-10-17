class Dataset(object):
    def __init__(self):
        pass

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return len(self.idx_list)

    def __repr__(self):
        return f'{str(self.__class__.__name__)}()'