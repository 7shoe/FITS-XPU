from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, Sampler

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


class ExactDistributedSampler(Sampler):
    """Disjoint, non-padding sampler for exact validation metrics."""

    def __init__(self, dataset, num_replicas, rank):
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("invalid distributed sampler topology")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        size = len(self.dataset)
        if self.rank >= size:
            return 0
        return (size - 1 - self.rank) // self.num_replicas + 1


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag in ('test', 'val'):
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        config=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    sampler = None
    if getattr(args, 'distributed', False):
        if flag == 'train':
            sampler = DistributedSampler(
                data_set,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=True,
                seed=args.seed,
                drop_last=True,
            )
            # DistributedSampler gives every rank equally many samples. FITS can
            # safely consume the equally sized final partial batch.
            drop_last = False
        elif flag in ('val', 'test'):
            sampler = ExactDistributedSampler(
                data_set, num_replicas=args.world_size, rank=args.rank
            )
        shuffle_flag = False

    if getattr(args, 'rank', 0) == 0:
        print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
