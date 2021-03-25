class AverageMeter(object):
    def __init__(self, name=''):
        self._name = name
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def reset(self):
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def __str__(self):
        return "%s: %.5f" % (self._name, self.avg)

    def get_avg(self):
        return self.avg

    def __repr__(self):
        return self.__str__()

def set_random_seed(seed):
    import random
    logging.info("Set seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_logger(log_dir=None):
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    log_format = "%(asctime)s | %(message)s"

    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format=log_format,
                        datefmt="%m/%d %I:%M:%S %p")

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "logger"))
        file_handler.setFormatter(logging.Formatter(log_format))

    logging.getLogger().addHandler(file_handler)
