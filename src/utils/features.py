import gc, time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from contextlib import contextmanager

# Feature Basis Class
@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


class Feature(metaclass=ABCMeta):

    def __init__(self, df, dir):
        self.name = self.__class__.__name__
        self.df = df
        self.dir = dir
        self.save_path = Path(self.dir) / f'{self.name}.ftr'
        self.new_colname = None

    def run(self):
        with timer(self.name):
            self.create_features()
        return self

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        cols = ['id', 'date', 'part', 'demand'] + self.new_colname
        self.df[cols].to_feather(str(self.save_path))
        self.df.drop(self.new_colname, axis=1, inplace=True)
        gc.collect()
