import os
import math
import torch
import torch.nn as nn
from model.common import DownBlock
import model.drn
from option import args


def dataparallel(model, gpu_list):
    ngpus = len(gpu_list)
    assert ngpus != 0, "only support gpu mode"
    assert torch.cuda.device_count() >= ngpus, "Invalid Number of GPUs"
    assert isinstance(model, list), "Invalid Type of Dual model"
    for i in range(len(model)):
        if ngpus >= 2:
            model[i] = nn.DataParallel(model[i], gpu_list).cuda()
        else:
            model[i] = model[i].cuda()
    return model


class Model(nn.Module):
    def __init__(self, opt, ckp):
        super(Model, self).__init__()
        print('Making model...')
        self.opt = opt
        self.scale = opt.scale
        self.idx_scale = 0
        self.self_ensemble = opt.self_ensemble
        self.cpu = opt.cpu
        self.device = torch.device('cpu' if opt.cpu else 'cuda')
        self.n_GPUs = opt.n_GPUs

        if self.scale[0] % 2 == 0:
            sf = 2
        else:
            sf = 3

        self.model = drn.make_model(opt).to(self.device)
        self.dual_models = []
        for _ in self.opt.scale:
            dual_model = DownBlock(opt, sf).to(self.device)
            self.dual_models.append(dual_model)
        
        if not opt.cpu and opt.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(opt.n_GPUs))
            self.dual_models = dataparallel(self.dual_models, range(opt.n_GPUs))

        self.load(opt.pre_train, opt.pre_train_dual, opt.pre_train_metasr, cpu=opt.cpu)

        if not opt.test_only:
            print(self.model, file=ckp.log_file)
            print(self.dual_models, file=ckp.log_file)
        
        # compute parameter
        num_parameter = self.count_parameters(self.model)
        ckp.write_log(f"The number of parameters is {num_parameter / 1000 ** 2:.2f}M")

    def forward(self, x, idx_scale=0):
        self.idx_scale = idx_scale
        target = self.get_model()
        if hasattr(target, 'set_scale'):
            target.set_scale(idx_scale)
        return self.model(x)

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module
    
    def get_dual_model(self, idx):
        if self.n_GPUs == 1:
            return self.dual_models[idx]
        else:
            return self.dual_models[idx].module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)
    
    def count_parameters(self, model):
        if self.opt.n_GPUs > 1:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def save(self, path, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(), 
            os.path.join(path, 'model', args.data_train +'_latest_x'+str(args.scale[len(args.scale)-1])+'.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(path, 'model', args.data_train +'_best_x'+str(args.scale[len(args.scale)-1])+'.pt')
            )
        #### save dual models ####
        dual_models = []
        for i in range(len(self.dual_models)):
            dual_models.append(self.get_dual_model(i).state_dict())
        torch.save(
            dual_models,
            os.path.join(path, 'model', args.data_train +'_dual_latest_x'+str(args.scale[len(args.scale)-1])+'.pt')
        )
        if is_best:
            torch.save(
                dual_models,
                os.path.join(path, 'model',args.data_train +'_dual_best_x'+str(args.scale[len(args.scale)-1])+'.pt')
            )

    def load(self, pre_train='.', pre_train_dual='.', pre_train_metasr='.', cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        #### load primal model ####

        weight4 = torch.load(pre_train, **kwargs)
       
        # for name in dual_models[0]:
        #     dual_models[0][name].requires_grad = False


        # print(weight4['P2W.0.bias'].requires_grad)
        # metasr = torch.load(pre_train_metasr, **kwargs)[0]
        # p2w = []
        # for param in metasr:
        #     paramsplit = param.split('.')
        #     if paramsplit == 'P2W':
        #         p2w.append(param)

        # p2weight={}
        # for p2 in p2w:
        #     p2weight[p2] = metasr[0][p2]
            # p2weight.update(metasr[p2])
        # new.update(p2weight)

        # param.requires_grad = False  

        if pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            self.get_model().load_state_dict(
                weight4,
                strict=False
            )
        #### load dual model ####
        if pre_train_dual != '.':
            print('Loading dual model from {}'.format(pre_train_dual))
            dual_models = torch.load(pre_train_dual, **kwargs)
            for i in range(len(self.dual_models)):
                self.get_dual_model(i).load_state_dict(
                    dual_models[i], strict=False
                )
