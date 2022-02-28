import numpy as np
import torch
from torchvision.models import vgg16
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from torchvision import models
import torch.nn as nn
#import kornia

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg= self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            #if (m_vgg[i]!=0).any():
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'fake_B_m', 'real_B_op', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.seg=models.segmentation.fcn_resnet50(pretrained=True)
        self.seg.eval().to("cuda")
        if self.isTrain:
            self.netD = networks.define_D1(opt.input_nc, opt.ndf, 3, 'instance', False, 2, False, gpu_ids=self.gpu_ids)

            self.criterionGAN = networks.GANLoss1().to(self.device)
            #self.criterionL1 = torch.nn.L1Loss()
            self.criterionL1 = self.L1_nn
            self.kh, self.kw = 3, 3
            #self.similoss = nn.CosineSimilarity(dim=0, eps=1e-6)
            # define loss functions
            #self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            self.criterionVGG = VGGLoss(self.gpu_ids)

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.real_A_m = self.real_A_m[:bs_per_gpu]
        self.real_B_m = self.real_B_m[:bs_per_gpu]
        self.real_B1_m = self.real_B1_m[:bs_per_gpu]
        self.real_B_op = self.real_B_op[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def mask_img(im1,im2):
        for i in range(len(im1)):
            for j in range(len(im1[i])):
                if np.all(im1[i,j]==[0,0,0]):
                    im2[i,j]=[0,0,0]
        return im2

    def L1_nn(self, inputs, targets):
        # IMPORTANT: Unlike the normal L1 loss, it is IMPORTANT which one is inputs & targets
        # inputs, targets: (B, nc, h, w)
        # kh, kw: int -- add consider +- 2kh and +- 2kw
        kh, kw = self.kh, self.kw
        B, nc, h, w = inputs.shape

        # Pad target w/ large number, padding: left,right,top,bottom
        targets_padded = F.pad(targets, (kw,kw,kh,kh), "constant", 10000)

        loss = torch.zeros(B, (2*kh+1)*(2*kw+1), h, w, device=self.device)
        count = 0
        for i in range(2*kh+1):
            for j in range(2*kw+1):
                filters = torch.zeros(nc, 1, 2*kh+1, 2*kw+1, device=self.device) # nc_out=no of neighbors, nc_in/groups, K, K--- for depthwise groups=nc_in
                filters[:,0,i,j] = 1
                outp = F.conv2d(targets_padded, filters, groups=nc) # B, nc_out=no of neighbors, h, w
                loss[:,count,:, :] = torch.mean(torch.abs(outp-inputs),1)
                count += 1
        loss, _ = torch.min(loss, 1) # B, h, w
        loss = torch.mean(loss)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        #masked A (not used)
        self.real_A_m = input['A' if AtoB else 'B'].to(self.device)
        self.real_B_m = input['B' if AtoB else 'A'].to(self.device)
        #paired B masked (used)
        self.real_B1_m = input['B1'].to(self.device)
        #unmasked A version
        self.real_A = input['AO' if AtoB else 'BO'].to(self.device)
        #paired unmasked B versions
        self.real_B_op = input['BO' if AtoB else 'AO'].to(self.device)
        #unpaired and randmly selected B from the sunny dataset
        self.real_B = input['BU'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        #self.snow_seg=torch.argmax(self.seg(self.real_A)['out'].squeeze(), dim=0)
        #fake_seg=torch.argmax(self.seg(self.fake_B).squeeze(), dim=0)
        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        #self.fake_seg=torch.argmax(self.seg(self.fake_B)['out'].squeeze(), dim=0)
        #self.fake_A=self.fake[self.real_A.size(0):]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]



        self.real_m = torch.cat((self.real_A_m, self.real_B1_m), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real_m = torch.flip(self.real_m, [3])


        self.fake_m = self.netG(self.real_m)
        self.fake_B_m = self.fake[:self.real_A.size(0)]


    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        real_B1_m=self.real_B1_m.detach().cpu().numpy()
        fake_B=self.fake_B.detach().cpu().numpy()
        mask_fake=torch.from_numpy(self.mask_img(real_B1_m,fake_B)).float().cuda()
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0


        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:

            self.loss_NCE_P = self.calculate_NCE_loss(self.real_B1_m, mask_fake)

            #self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = self.loss_NCE +self.loss_NCE_P
        else:
            #NCE loss for the defined functions
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN+loss_NCE_both

        self.loss_G_L1_B = self.criterionL1(mask_fake,self.real_B1_m)

        # VGG feature matching loss
        #snow_seg=torch.argmax(self.seg(self.real_A.detach().).squeeze(), dim=0)
        #fake_seg=torch.argmax(self.seg(self.fake_B.detach().numpy()).squeeze(), dim=0)
        #self.edlos=
        #self.loss_G_L1_obj=self.criterionL1(self.real_A, self.idt_B) *10

        #self.loss_G_VGG_B = self.criterionVGG(self.real_B1_m, self.fake_B_m) * 10
        #L1 loss
        self.loss_G=self.loss_G+(self.loss_G_L1_B)

        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
