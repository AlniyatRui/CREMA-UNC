"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os,random
import numpy as np
import torch
import math
import torch.distributed as dist
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample

EMA_DECAY = 0.98
LAZY_GRADSEL_EPOCH=5

class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"
        
        self.metric_logger = MetricLogger(delimiter="  ")
        self.metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        self.metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        self.metric_logger.add_meter("loss_rec", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        self.es_ticker = {}
        self.post_es_ticker = {'rgb':-1, 'pc': -1, 'norm': -1, 'depth': -1, 'audio':-1, 'flow':-1}
        # print(f'\n\n\n\n\n\n\nself.pred_es_ticker : {self.pred_es_ticker}')
        
        
        self.modality_grad_slope = {'rgb':[],'pc':[],'norm':[],'depth':[],'audio':[], 'flow':[]}

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg
        assert len(datasets_config) > 0, "At least one dataset has to be specified."
        self.availble_data_ratio = 0

        for name in datasets_config:
            dataset_config = datasets_config[name]
            if 'available_data' in dataset_config:
                self.availble_data_ratio = getattr(dataset_config,'available_data',0)
            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):
        modalities = list(os.environ['MODALITIES'].split('_'))

        adv_methods = os.environ['ATTACK_METHOD']
        at_methods = os.environ['AT_FRAMEWORK']
        at_ratio = os.environ['ATTACK_RATIO']

        # Clean input inference
        out = model(samples)
        loss = out["loss"]
        token_mask = out['mask'].bool()

        temp = random.random()

        if os.environ['AT'] == 'True' and temp > (1 - eval(at_ratio)):
            if adv_methods == "MI":
                from lavis.attack_methods.SQA.MI import MI_Attack
                adv_samples = MI_Attack(model, modalities, samples)
            elif adv_methods == "NuAT":
                from lavis.attack_methods.SQA.NuAT import NuAT_Attack
                adv_samples = NuAT_Attack(model, modalities, samples, out)

            if at_methods == "TRADES":
                adv_out = model(adv_samples)
                logits_clean = out['logits']
                logits_adv = adv_out['logits']

                import torch.nn.functional as F
                probs_adv = F.log_softmax(logits_adv, dim=-1)     # [B, T, V]
                probs_clean = F.softmax(logits_clean, dim=-1)         # [B, T, V]

                loss_kl = F.kl_div(probs_adv[token_mask], probs_clean[token_mask], reduction='batchmean') 
                beta = 1.0
                loss = loss + beta * loss_kl

            elif at_methods == "NuAT":
                Nuc_Reg = 4
                adv_out = model(adv_samples)
                loss = out['loss'] + Nuc_Reg * torch.linalg.matrix_norm(adv_out['logits']*token_mask - \
                                                    out['logits']*token_mask, ord='nuc', dim=(1, 2)).mean()
                
        if 'loss_rec' in out:
            return [loss, out['loss_rec']]
        else:
            return [loss]

    def valid_step(self, model, samples):
        raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def get_num_present_modalities(self, mr, num_total_modalities=3):
        """
        根据缺失率 mr 决定当前样本应保留多少个模态。
        mr: 单个模态的缺失概率 (0.0 到 1.0)
        num_total_modalities: 总共的模态数量
        返回: 保留的模态数量 (0 到 num_total_modalities)
        """
        if not (0.0 <= mr <= 1.0):
            raise ValueError("mr (missing rate) 必须在 0.0 和 1.0 之间")

        # 方法1: 每个模态独立决定是否缺失 (更直接反映mr的含义)
        present_count = 0
        for _ in range(num_total_modalities):
            if random.random() >= mr:  # mr 是缺失的概率,所以 1-mr 是存在的概率
                present_count += 1
        return present_count

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []
        RANGES = {
            0: (0, 1/8),                    # vt
            1: (1/8, 1/2),                  # vt + x/y/z
            2: (1/2, 7/8),                  # vt + xy/yz/xz
            3: (7/8, 1.0)                   # vtxyz
        }
        mr = 0.9
        num_modalities = 3
        import random
        for samples in metric_logger.log_every(data_loader, print_freq, header):
            num_modal = self.get_num_present_modalities(mr, num_modalities)
            low, high = RANGES[num_modal]
            samples['comb_random'] = random.uniform(low, high)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=len(data_loader),
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _synchronize_between_processes(self, target):
        """
        Warning: does not synchronize the deque!
        """
        if not dist_utils.is_dist_avail_and_initialized():
            return
        dist.barrier()
        dist.all_reduce(target.data)

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        def check_gradient(module, grad_input, grad_output):
            print(f"Module: {module}")
            print(f"Gradient input: {grad_input}")
            print(f"Gradient output: {grad_output}")
        

        # ES_TEMPERATURE = model.module.es_temperature
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)
        
        if self.availble_data_ratio > 0:
            full_modal_iter_range = math.floor(self.availble_data_ratio * iters_per_epoch / (self.availble_data_ratio + 0.05))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        self.metric_logger.lr.clear(window_size=1)
        self.metric_logger.loss.clear(window_size=1)        
        
        for tm in model.module.modalities:
            if hasattr(self.metric_logger, f'{tm}_grad_scale'):    
                eval(f'self.metric_logger.{tm}_grad_scale.clear(window_size=20)')
                    
        if not hasattr(self, 'initial_trainables'):
            self.initial_trainables = [ww for ww, qq in model.module.named_parameters() if qq.requires_grad==True]   
        
        fusion_list = {ww: qq for ww, qq in model.module.named_parameters() if 'fusion' in ww}

        for i in self.metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            if i >= iters_per_epoch:
                break
            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                    "full_modal": True if self.availble_data_ratio > 0 and i < full_modal_iter_range else False,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)
            # print(model.module.task)
            if 'seq' in model.module.task:                
                tmp_loss = 0.                
                modality_trainables = {}                
                for t_m in model.module.modalities:
                    modality_trainables[t_m] = [ww for ww, qq in model.module.named_parameters() if f'_{t_m}' in ww]
                gs = {}
                for t_m in model.module.modalities:     
                    if 'gradsel' in model.module.task:
                        if not hasattr(self.metric_logger, f'{t_m}_cosim'):    
                            self.metric_logger.add_meter(f"{t_m}_grad_scale", SmoothedValue(window_size=20, fmt="{value:.6f}"))
                    
                    if not t_m in self.es_ticker.keys():
                        print(f'Initialize self.es_ticker[{t_m}] = False')
                        self.es_ticker[t_m] = False
        
                    if self.es_ticker[t_m] == True:
                        print(f'exit the training of [{t_m}]')
                        continue                        
                    else:
                        for ww, qq in model.module.named_parameters():
                            if (f'_{t_m}' in ww and ww in self.initial_trainables) or ww in fusion_list.keys():
                                qq.requires_grad=True
                            elif 'pad_embedding_weight' in ww: # for crema_pad
                                continue
                            # elif ('t2' in ww) or ('v2' in ww) or ('_vtp' in ww): # for crema_gen
                            #     continue
                            else:
                                qq.requires_grad=False

                        with torch.cuda.amp.autocast(enabled=use_amp):
                            losses = self.train_step(model=model, samples=samples)
                        if len(losses) == 1:
                            loss = losses[0]
                            loss_rec = 0.0
                        elif len(losses) == 2:
                            loss, loss_rec = losses
                        
                        if use_amp:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        
                        if 'gradsel' in model.module.task and (i + 1) % accum_grad_iters == 0:
                            scaler.unscale_(optimizer)
                            trainables_list = {ww: qq for ww, qq in model.module.named_parameters() if qq.requires_grad==True and 'attention' in ww and len(qq.shape) > 1}
                            gradsel_ckpt = torch.cat([trainables_list[ww].grad.detach().view(-1) for ww in trainables_list if t_m in ww])

                            gs[t_m] = torch.norm(torch.abs(gradsel_ckpt)).item()                        
                            
                        
                        import torch.nn.utils as utils

                        # update gradients every accum_grad_iters iterations
                        if (i + 1) % accum_grad_iters == 0:
                            if use_amp:
                                # scaler.unscale_(optimizer)  # 首先取消梯度的缩放
                                # utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 进行梯度裁剪
                                scaler.step(optimizer)
                                scaler.update()                     
                            else:    
                                # utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                optimizer.step()
                            optimizer.zero_grad()
                        
                        if 'gradsel' in model.module.task and (i + 1) % accum_grad_iters == 0:
                            if hasattr(self.metric_logger, f'{t_m}_grad_scale'):
                                arguments = {f'{t_m}_grad_scale': gs[t_m]}
                                self.metric_logger.update(**arguments)                 
                                                        
                        tmp_loss += loss.item()
                
                self.metric_logger.update(loss=tmp_loss)
                self.metric_logger.update(loss_rec=loss_rec)
                self.metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            else:
                trainables_list = {ww: qq for ww, qq in model.module.named_parameters() if qq.requires_grad==True and 'attention' in ww and len(qq.shape) > 1}
                gs = {}
                if 'gradsel' in model.module.task:
                    for t_m in model.module.modalities:
                        if not hasattr(self.metric_logger, f'{t_m}_grad_scale'):    
                            self.metric_logger.add_meter(f"{t_m}_grad_scale", SmoothedValue(window_size=20, fmt="{value:.6f}"))
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss = self.train_step(model=model, samples=samples)
                
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if 'gradsel' in model.module.task:
                    scaler.unscale_(optimizer)
                    for t_m in model.module.modalities:
                        gradsel_ckpt = torch.cat([trainables_list[ww].grad.detach().view(-1) for ww in trainables_list if t_m in ww])
                        gs[t_m] = torch.norm(torch.abs(gradsel_ckpt)).item()
                        
                # update gradients every accum_grad_iters iterations
                if (i + 1) % accum_grad_iters == 0:
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()                     
                    else:    
                        optimizer.step()
                    optimizer.zero_grad()

                if 'gradsel' in model.module.task:
                    for t_m in model.module.modalities:
                        if hasattr(self.metric_logger, f'{t_m}_grad_scale'):
                            arguments = {f'{t_m}_grad_scale': gs[t_m]}
                            self.metric_logger.update(**arguments)                 
            
                self.metric_logger.update(loss=loss.item())
                self.metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        dist.barrier()
        if 'gradsel-pred-es' in model.module.task:
            for t_m in model.module.modalities:   
                if self.pred_es_ticker[t_m] == inner_epoch and inner_epoch >= LAZY_GRADSEL_EPOCH:                
                    self.es_ticker[t_m] = True
                    print(f'################## gradsel-pred-es: early stop {t_m}: epoch {inner_epoch}')   
                                        
        elif 'gradsel-es' in model.module.task:
            for t_m in model.module.modalities:                
                if inner_epoch >= LAZY_GRADSEL_EPOCH:
                    ev_tmp = eval(f'self.metric_logger.{t_m}_grad_scale.global_avg')
                    print(f'##############before self.modality_grad_slope[{t_m}]: {self.modality_grad_slope[t_m]}')
                    mean_history = np.mean(self.modality_grad_slope[t_m])
                    print(f"mean_history ({mean_history}) * ES_TEMPERATURE ({ES_TEMPERATURE}) < eval(f'self.metric_logger.{t_m}_grad_scale.global_avg') ({ev_tmp})")                
                    if mean_history * ES_TEMPERATURE < ev_tmp:
                        self.es_ticker[t_m] = True
                        print(f'gradsel-es: early stop {t_m}: epoch {inner_epoch}')                        
                        self.post_es_ticker[t_m] = inner_epoch
                    
                self.modality_grad_slope[t_m].append(eval(f'self.metric_logger.{t_m}_grad_scale.global_avg'))
                print(f'############## self.modality_grad_slope[t_m]: {self.modality_grad_slope[t_m]}')
                
        # gather the stats from all processes        
        print(f'self.post_es_ticker: {self.post_es_ticker}')
        self.metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(self.metric_logger.global_avg()))
        return {
            k: "{:.4f}".format(meter.global_avg)
            for k, meter in self.metric_logger.meters.items()
        }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
