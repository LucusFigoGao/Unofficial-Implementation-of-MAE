## timm库支持的transformer结构

```
'vit_base_patch16_224',
'vit_base_patch16_224_in21k',
'vit_base_patch16_384',
'vit_base_patch32_224',
'vit_base_patch32_224_in21k',
'vit_base_patch32_384',
'vit_base_resnet26d_224',
'vit_base_resnet50_224_in21k',
'vit_base_resnet50_384',
'vit_base_resnet50d_224',
'vit_deit_base_distilled_patch16_224',
'vit_deit_base_distilled_patch16_384',
'vit_deit_base_patch16_224',
'vit_deit_base_patch16_384',
'vit_deit_small_distilled_patch16_224',
'vit_deit_small_patch16_224',
'vit_deit_tiny_distilled_patch16_224',
'vit_deit_tiny_patch16_224',
'vit_huge_patch14_224_in21k',
'vit_large_patch16_224',
'vit_large_patch16_224_in21k',
'vit_large_patch16_384',
'vit_large_patch32_224',
'vit_large_patch32_224_in21k',
'vit_large_patch32_384',
'vit_small_patch16_224',
'vit_small_resnet26d_224',
'vit_small_resnet50d_s3_224'
```

## timm库支持的swin-transformer结构

```
'swin_base_patch4_window12_384'
'swin_base_patch4_window7_224'
'swin_large_patch4_window12_384'
'swin_large_patch4_window7_224'
'swin_small_patch4_window7_224'
'swin_tiny_patch4_window7_224'
'swin_base_patch4_window12_384_in22k'
'swin_base_patch4_window7_224_in22k'
'swin_large_patch4_window12_384_in22k'
'swin_large_patch4_window7_224_in22k'
'swin_s3_tiny_224'
'swin_s3_small_224'
'swin_s3_base_224'
```

```
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--seed', default=888, type=int, help='random seed')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=30, type=int, help='total epochs to run')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--num_classes', default=10, type=int, help='The number of classes in dataset')
parser.add_argument('--pretrained', default=1, type=int, help='If pretrain the model, default not')
parser.add_argument('--dataset', default='imagenet', type=str, help="dataset")
parser.add_argument('--criterion', default='CE', type=str, help="Loss function")
parser.add_argument('--scheduler', default='Cosine', type=str, help="The scheduler of optimizer")
parser.add_argument('--optimizer', default='Adam', type=str, help="The optimizer of training")
parser.add_argument('--model', default="mae", type=str, help='The type of loaded models, mae, simmim')
parser.add_argument('--encoder', default='vit_base_patch16_384', type=str, help="The type of encoder, please read README.md")
```