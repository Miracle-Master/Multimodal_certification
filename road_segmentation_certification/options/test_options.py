from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='test', help='train, val, test')
        parser.add_argument('--certification_method', type=str, default="MMCert")
        parser.add_argument('--nepoch', type=int, default=100, help='maximum epochs')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for optimizer')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum factor for SGD')
        parser.add_argument('--weight_decay', type=float, default=0.0005, help='momentum factor for optimizer')
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
        parser.add_argument('--lr_decay_iters', type=int, default=5000000, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--lr_decay_epochs', type=int, default=25, help='multiply by a gamma every lr_decay_epoch epochs')
        parser.add_argument('--lr_gamma', type=float, default=0.9, help='gamma factor for lr_scheduler')
        parser.add_argument('--save_dir', type=str, default='checkpoints\\kitti')
        parser.add_argument('--ablation_ratio_test1', type=float, default=0.00322)
        parser.add_argument('--ablation_ratio_test2', type=float, default=0.001074)
        parser.add_argument('--ablation_ratio_test', type=float, default=0.00215)

        self.isTrain = True
        return parser


#2000 pixels in total: (0.00215)
#500 pixels for an single image: (0.001074)
#1500 pixels for an single image: (0.00322)