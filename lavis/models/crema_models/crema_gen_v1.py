"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast, BertTokenizer
import torch.distributed as dist

from lavis.utils import safe_breakpoint_rank0
from lavis.common.registry import registry
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from lavis.models.crema_models.resampler import Resampler
from positional_encodings.torch_encodings import PositionalEncoding1D
import numpy as np

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )



class MLPLayer(nn.Module):
    def __init__(self, dim, embed_dim, is_Fusion=False):
        super().__init__()
        if is_Fusion:
            self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        else:
            self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))


@registry.register_model("crema_gen_v1")
class CREMA_GEN_v1(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
        - caption_coco_flant5xl: fintuned image captioning model with FlanT5-XL
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5", "pretrain_flant5xl")
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
        "pretrain_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",
        "caption_coco_flant5xl": "configs/models/blip2/blip2_caption_flant5xl.yaml",
    }

    def __init__(self, img_size=224, drop_path_rate=0,
        use_grad_checkpoint=False, vit_precision="fp16", freeze_vit=True,
        num_query_token=32, t5_model="google/flan-t5-xl", prompt="",
        max_txt_len=32, frame_num=8, answer_num=5, apply_lemmatizer=False, 
        task='concate',
        modalities='rgb',
        downstream_task='mcqa', # caption / oeqa / mcqa
        lora_rank=64,
        lora_layer=None,
        lora_dropout=0.1,
        missing_mode=0,
        gen_loss_weight=0.1):

        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        
        self.task = task #.split('_')
        modalities = modalities.split('_')
        self.modalities = []
        self.skip = []
        for m in modalities:
            if 'skip' in m:
                self.modalities.append(m.split('-')[0])
                self.skip.append('-skip')
            else:
                self.modalities.append(m)
                self.skip.append('')

        num_features = 1408
        # ========= init vision encoder ============
        # init vision backbone for vision experts
        if 'rgb' in self.modalities or 'depth' in self.modalities or 'flow' in self.modalities or 'norm' in self.modalities:
            self.visual_encoder = self.init_vision_encoder_only(
                img_size, drop_path_rate, use_grad_checkpoint, vit_precision)
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False         
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        if 'audio' in self.modalities:
            self.audio_encoder, self.ln_audio = self.init_audio_encoder('beats', cached_audio=False)
            for name, param in self.audio_encoder.named_parameters():
                param.requires_grad = False  
            self.audio_encoder = self.audio_encoder.eval()
            self.audio_encoder.train = disabled_train
            logging.info("freeze audio encoder")

        if 'pc' in self.modalities:
            # pre-extracted features
            pass
        
        # ========= init LLM ============  
        # text backbone
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model, config=t5_config)
        # freeze T5
        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data.bfloat16() 

        # ========= init Qformer ============
        self.Qformer, encoder_config = self.init_Multimodal_Qformer(
            num_query_token, num_features, #self.visual_encoder.num_features,
            modulars=self.modalities, 
            r=lora_rank, lora_layer=lora_layer, lora_dropout=lora_dropout)

        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.num_query_token = num_query_token
        self.mv_embeddings = {}
        
        g_size = sum([ww.numel() for nn, ww in self.named_parameters() if len(ww.shape) > 1 and 'attention' in nn and 'rgb' in nn])
        for md in self.modalities:            
            setattr(self, f'{md}_gradsel_embed', nn.Parameter(torch.zeros((1, g_size))))
            
            new_param = nn.Parameter(torch.zeros((1, g_size)))#.to(self.device)
            self.register_parameter(f'{md}_gradsel_embed', new_param)
        
        # ========= Init Modality Components ============
        # Initialize modality-specific components
        if 'rgb' in self.modalities:
            self.query_tokens_rgb = nn.Parameter(
                torch.zeros(1, num_query_token, encoder_config.hidden_size))
            self.query_tokens_rgb.data.normal_(mean=0.0, std=encoder_config.initializer_range)
            self.t5_proj_rgb = nn.Linear(
                self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)
            self.ln_rgb = nn.LayerNorm(self.visual_encoder.num_features)
            

        if 'flow' in self.modalities:
            self.query_tokens_flow = nn.Parameter(
                torch.zeros(1, num_query_token, encoder_config.hidden_size))
            self.query_tokens_flow.data.normal_(mean=0.0, std=encoder_config.initializer_range)
            self.t5_proj_flow = nn.Linear(
                self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)
            self.ln_flow = nn.LayerNorm(self.visual_encoder.num_features)

        if 'norm' in self.modalities:
            self.query_tokens_norm = nn.Parameter(
                torch.zeros(1, num_query_token, encoder_config.hidden_size))
            self.query_tokens_norm.data.normal_(mean=0.0, std=encoder_config.initializer_range)
            self.t5_proj_norm = nn.Linear(
                self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)
            self.ln_norm = nn.LayerNorm(self.visual_encoder.num_features)

        if 'depth' in self.modalities:
            self.query_tokens_depth = nn.Parameter(
                torch.zeros(1, num_query_token, encoder_config.hidden_size))
            self.query_tokens_depth.data.normal_(mean=0.0, std=encoder_config.initializer_range)
            self.t5_proj_depth = nn.Linear(
                self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)
            self.ln_depth = nn.LayerNorm(self.visual_encoder.num_features)
        
        if 'audio' in self.modalities:
            self.query_tokens_audio = nn.Parameter(
                torch.zeros(1, num_query_token, encoder_config.hidden_size))
            self.query_tokens_audio.data.normal_(mean=0.0, std=encoder_config.initializer_range)
            self.t5_proj_audio = nn.Linear(
                self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)
            
            self.projection_audio = nn.Linear(self.audio_encoder.num_features, num_features)
            self.ln_audio = nn.LayerNorm(num_features)
            
        if 'pc' in self.modalities:
            self.query_tokens_pc = nn.Parameter(
                torch.zeros(1, num_query_token, encoder_config.hidden_size))
            self.query_tokens_pc.data.normal_(mean=0.0, std=encoder_config.initializer_range)
            self.t5_proj_pc = nn.Linear(
                self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)
            self.ln_pc = nn.LayerNorm(num_features)
            pos_model = PositionalEncoding1D(1408 // 3)
            x = torch.zeros(1, 256, 1408 // 3)
            self.pos_embedding = pos_model(x).squeeze().cuda()

        # ========= Init Glow Flow Models for Modality Generation ============
        self.missing_mode = int(missing_mode)
        self.gen_loss_weight = float(gen_loss_weight)
        
        # init compressor
        self.compressor_size= 2
        self.compressor=Resampler(
            grid_size=self.compressor_size,
            embed_dim=1408,
            num_heads=8,
        )
        '''
        missing_mode: 
        0 for all available
        1 for vt + x
        2 for vt + xy
        3 for vt + xyz 
        '''
        self.non_rgb_modalities = [modal for modal in modalities if modal != 'rgb']

        if self.missing_mode != 0:
            self._init_flow_models()

        if 'espresso' in self.task:
            if "rgb" in self.modalities:
                _fusion_input_dim = 2048*(len(self.modalities)-1)
            else:
                _fusion_input_dim = 2048*(len(self.modalities))
            self.fusion = nn.Sequential(
                    nn.Linear(_fusion_input_dim, 2048)
            )
                
            self.sigmoid = nn.Sigmoid()
            
        self.downstream_task = downstream_task 
        self.max_txt_len = 77
        answer_id = [71, 272, 205, 309, 262] # A B C D E
        self.answer_id = answer_id[:answer_num]
        
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
        self.frame_num = frame_num
        self.ANS_MAP = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}

        if frame_num == 1:
            self.vid_prefix = ['Frame: ']
            self.depth_prefix = ['Depth Map: ']
            self.flow_prefix = ['Optical Flow: ']
            self.norm_prefix = ['Surface Normalization: ']
        else:
            self.vid_prefix = ['Frame {}: '.format(str(i+1)) for i in range(frame_num)]
            self.depth_prefix = ['Depth Map {}: '.format(str(i+1)) for i in range(frame_num)]
            self.flow_prefix = ['Optical Flow {}: '.format(str(i+1)) for i in range(frame_num)]
            self.norm_prefix = ['Surface Normalization {}: '.format(str(i+1)) for i in range(frame_num)]

        self.audio_prefix = ['Audio: ']
        self.pc_prefix = ['3D Model: ']

    def _init_flow_models(self):
        """Initialize Glow flow models for modality generation."""
        # Initialize Glow models for each possible modality
        # Each modality can be generated from any other modality
        prompt_dim = 1408
        prompt_length = 16
        self.tlen = 77
        self.vlen = 257
        self.xlen = 257 # other visual
        # self.xlen = 256 # audio
        # self.xlen = 5000 # pc
        self.ylen = 257
        self.zlen = 257
        self.alen = 257
        self.v2x = MLPLayer(1408, 
                        prompt_dim
                    )
        self.t2x = MLPLayer(2048, 
                        prompt_dim
                    )
        if self.missing_mode == 1:
            self.x_vtp = MLPLayer(prompt_length + self.tlen + self.vlen, self.xlen, True)
        elif self.missing_mode == 2:
            self.v2y = MLPLayer(1408, 
                        prompt_dim
                    )
            self.t2y = MLPLayer(2048, 
                        prompt_dim
                        )
            self.x_vtp = MLPLayer(prompt_length + self.tlen + self.vlen, self.xlen, True)
            self.y_vtp = MLPLayer(prompt_length + self.tlen + self.vlen, self.ylen, True)
        elif self.missing_mode == 3:
            self.v2y = MLPLayer(1408, 
                        prompt_dim
                    )
            self.t2y = MLPLayer(2048, 
                        prompt_dim
                        )
            self.v2z = MLPLayer(1408, 
                        prompt_dim
                    )
            self.t2z = MLPLayer(2048, 
                        prompt_dim
                        )
            self.x_vtp = MLPLayer(prompt_length + self.tlen + self.vlen, self.xlen, True)
            self.y_vtp = MLPLayer(prompt_length + self.tlen + self.vlen, self.ylen, True)
            self.z_vtp = MLPLayer(prompt_length + self.tlen + self.vlen, self.zlen, True)
        elif self.missing_mode == 4:
            self.v2y = MLPLayer(1408, 
                        prompt_dim
                    )
            self.t2y = MLPLayer(2048, 
                        prompt_dim
                        )
            self.v2z = MLPLayer(1408, 
                        prompt_dim
                    )
            self.t2z = MLPLayer(2048, 
                        prompt_dim
                        )
            self.v2m = MLPLayer(1408, 
                        prompt_dim
                    )
            self.t2m = MLPLayer(2048, 
                        prompt_dim
                        )
            self.x_vtp = MLPLayer(prompt_length + self.tlen + self.vlen, self.xlen, True)
            self.y_vtp = MLPLayer(prompt_length + self.tlen + self.vlen, self.ylen, True)
            self.z_vtp = MLPLayer(prompt_length + self.tlen + self.vlen, self.zlen, True)
            self.m_vtp = MLPLayer(prompt_length + self.tlen + self.vlen, self.alen, True)
        else:
            raise NotImplementedError
        
        generative_prompt = torch.zeros(1,prompt_dim, prompt_length)
        self.generative_prompt = nn.Parameter(generative_prompt)
        
        # MSE loss for comparing original vs generated embeddings (for training)
        self.MSE = MSE()

    def forward(self, samples):
        if self.missing_mode != 0:
            if self.missing_mode == 1:
                x = self.non_rgb_modalities[0]
            elif self.missing_mode == 2:
                x, y = self.non_rgb_modalities
            elif self.missing_mode == 3:
                x, y, z = self.non_rgb_modalities
            elif self.missing_mode == 4:
                x, y, z, m = self.non_rgb_modalities
            else:
                raise NotImplementedError
            
        # rgb visual embedding
        qa_text, answer = samples['qa_input'], samples['qa_output']
        b = len(qa_text)

        input_embed_dict, input_att_dict = {}, {}
        
        # GEN step1: encode available modalities
        comb_random = random.random()
        if (self.missing_mode == 1 and comb_random < 0.5) or (
            self.missing_mode == 2 and comb_random < 0.25) or (
            self.missing_mode == 3 and comb_random < 1/8) or (
            self.missing_mode == 4 and comb_random < 1/16):
            # Encode only v here
            # case1: vtx, 50% x is missing
            # case2-1: vtxy, 25% xy both missing
            modal = 'rgb'
            input = samples[modal]
            if input.shape[1] == 3:
                input = input.permute(0, 2, 1, 3, 4)
            input_embed_dict[modal], input_att_dict[modal] = self.encode_input(input, modal)
        elif self.missing_mode == 4 and comb_random < 15/16:
            if comb_random < 1/8:
                modalities = ['rgb', x] 
            elif comb_random < 3/16:
                modalities = ['rgb', y] 
            elif comb_random < 1/4: 
                modalities = ['rgb', z] 
            elif comb_random < 5/16:
                modalities = ['rgb', m] 
            elif comb_random < 3/8: 
                modalities = ['rgb', x, y] 
            elif comb_random < 7/16:
                modalities = ['rgb', x, z] 
            elif comb_random < 1/2: 
                modalities = ['rgb', x, m] 
            elif comb_random < 9/16: 
                modalities = ['rgb', y, z] 
            elif comb_random < 5/8: 
                modalities = ['rgb', y, m]
            elif comb_random < 11/16:
                modalities = ['rgb', z, m] 
            elif comb_random < 12/16: 
                modalities = ['rgb', x, y, z]
            elif comb_random < 13/16:
                modalities = ['rgb', x, y, m] 
            elif comb_random < 14/16:
                modalities = ['rgb', x, z, m] 
            else:
                modalities = ['rgb', y, z, m] 

            for modal in modalities:
                input = samples[modal] 
                # fix some loading issue
                if input.shape[1] == 3:
                    input = input.permute(0, 2, 1, 3, 4)
                # following 3D-LLM 
                if modal == 'pc':
                    with torch.cuda.amp.autocast(dtype=torch.float32):
                        pc_embeds = samples["pc_feat"]
                        pc = samples["pc"].long()
                        all_pcs = torch.zeros((pc_embeds.shape))
                        for j in range(pc.shape[0]):
                            pcs = []
                            for i in range(3):
                                pc_i = pc[j][:, i]
                                pcs.append(self.pos_embedding[pc_i])
                            pcs = torch.cat(pcs, -1)
                            all_pcs[j][:, :1407] = pcs
                        all_pcs = all_pcs.cuda()
                    pc_embeds = pc_embeds + 0.01 * all_pcs
                    atts = torch.ones(pc_embeds.size()[:-1], dtype=torch.long).to(pc_embeds.device)
                    input_embed_dict[modal], input_att_dict[modal] = pc_embeds, atts
                else:
                    input_embed_dict[modal], input_att_dict[modal] = self.encode_input(input, modal)
        elif self.missing_mode == 3 and comb_random < 7/8:
            if comb_random < 1/4:
                modalities = ['rgb', x]
            elif comb_random < 3/8:
                modalities = ['rgb', y]
            elif comb_random < 1/2:
                modalities = ['rgb', z]
            elif comb_random < 5/8:
                modalities = ['rgb', x, y]
            elif comb_random < 3/4:
                modalities = ['rgb', x, z]
            else:
                modalities = ['rgb', y, z]
            for modal in modalities:
                input = samples[modal] 
                # fix some loading issue
                if input.shape[1] == 3:
                    input = input.permute(0, 2, 1, 3, 4)
                if modal == 'pc':
                    with torch.cuda.amp.autocast(dtype=torch.float32):
                        pc_embeds = samples["pc_feat"]
                        pc = samples["pc"].long()
                        all_pcs = torch.zeros((pc_embeds.shape))
                        for j in range(pc.shape[0]):
                            pcs = []
                            for i in range(3):
                                pc_i = pc[j][:, i]
                                pcs.append(self.pos_embedding[pc_i])
                            pcs = torch.cat(pcs, -1)
                            all_pcs[j][:, :1407] = pcs
                        all_pcs = all_pcs.cuda()
                    pc_embeds = pc_embeds + 0.01 * all_pcs
                    atts = torch.ones(pc_embeds.size()[:-1], dtype=torch.long).to(pc_embeds.device)
                    input_embed_dict[modal], input_att_dict[modal] = pc_embeds, atts
                else:
                    
                    input_embed_dict[modal], input_att_dict[modal] = self.encode_input(input, modal)
        elif self.missing_mode == 2 and comb_random < 0.75:
            modalities = ['rgb', x] if comb_random < 0.5 else ['rgb', y]
            for modal in modalities:
                input = samples[modal] 
                # fix some loading issue
                if input.shape[1] == 3:
                    input = input.permute(0, 2, 1, 3, 4)
                input_embed_dict[modal], input_att_dict[modal] = self.encode_input(input, modal)
        elif self.missing_mode == 0 or (
            self.missing_mode == 1 and comb_random >= 0.5) or (
            self.missing_mode == 2 and comb_random >= 0.75) or (
            self.missing_mode == 3 and comb_random >= 7/8) or (
            self.missing_mode == 4 and comb_random >= 15/16):
            # all modality in self.modality
            for modal in self.modalities:
                input = samples[modal] # samples: dict_keys(['rgb', 'qa_input', 'qa_output', 'question_id', 'duration', 'epoch', 'num_iters_per_epoch', 'iters'])
                
                if modal in ['rgb', 'depth', 'norm', 'flow']:
                    # fix some loading issue
                    if input.shape[1] == 3:
                        input = input.permute(0, 2, 1, 3, 4)
                # following 3D-LLM 
                if modal == 'pc':
                    with torch.cuda.amp.autocast(dtype=torch.float32):
                        pc_embeds = samples["pc_feat"]
                        pc = samples["pc"].long()
                        all_pcs = torch.zeros((pc_embeds.shape))
                        for j in range(pc.shape[0]):
                            pcs = []
                            for i in range(3):
                                pc_i = pc[j][:, i]
                                pcs.append(self.pos_embedding[pc_i])
                            pcs = torch.cat(pcs, -1)
                            all_pcs[j][:, :1407] = pcs
                        all_pcs = all_pcs.cuda()
                    pc_embeds = pc_embeds + 0.01 * all_pcs
                    atts = torch.ones(pc_embeds.size()[:-1], dtype=torch.long).to(pc_embeds.device)
                    input_embed_dict[modal], input_att_dict[modal] = pc_embeds, atts
                else:
                    input_embed_dict[modal], input_att_dict[modal] = self.encode_input(input, modal)
        else:
            raise NotImplementedError
        
        device = input_embed_dict[list(input_embed_dict.keys())[0]].device

        input_text= self.t5_tokenizer(
            qa_text, padding="longest", truncation=True,
            max_length=self.max_txt_len, return_tensors="pt").to(device)
        if input_text["input_ids"].size(1) < 77: # to match generator param size
            pad_token_id = self.t5_tokenizer.pad_token_id
            padding_length = 77 - input_text["input_ids"].size(1)
            padding = torch.full((input_text["input_ids"].size(0), padding_length), 
                            pad_token_id, 
                            dtype=torch.long, 
                            device=device)
            # Concatenate original input_ids with padding
            input_text["input_ids"] = torch.cat([input_text["input_ids"], padding], dim=1)
                # Also pad attention mask with zeros
            attention_padding = torch.zeros((input_text["attention_mask"].size(0), padding_length), 
                                        dtype=torch.long, 
                                        device=device)
            input_text["attention_mask"] = torch.cat([input_text["attention_mask"], attention_padding], dim=1)
        input_text_embeds = self.t5_model.encoder.embed_tokens(input_text.input_ids) 

        
        if self.missing_mode == 1:
            t = input_text_embeds.repeat(4,1,1).transpose(1, 2)
            v = input_embed_dict['rgb'].transpose(1, 2)
            v_bs = input_embed_dict['rgb'].size(0)
            xx = torch.cat(
                    [self.generative_prompt.repeat(v_bs,1,1), self.t2x(t), self.v2x(v)],
                    dim=2,
                )
            xx = self.x_vtp(xx.transpose(1, 2))
            if comb_random >= 0.5:
                # full modality, learn to reconstruct
                loss_rec = self.gen_loss_weight * self.MSE(xx, input_embed_dict[x].detach())
            else:
                # partial modality, direct reconstruct
                loss_rec = 0.0
                input_embed_dict[x] = xx
            input_att_dict[x] = input_att_dict['rgb'].clone()
        elif self.missing_mode == 2:
            t = input_text_embeds.repeat(4,1,1).transpose(1, 2)
            v = input_embed_dict['rgb'].transpose(1, 2)
            v_bs = input_embed_dict['rgb'].size(0)
            xx = torch.cat(
                    [self.generative_prompt.repeat(v_bs,1,1), self.t2x(t), self.v2x(v)],
                    dim=2,
            )
            xx = self.x_vtp(xx.transpose(1, 2))
            yy = torch.cat(
                [self.generative_prompt.repeat(v_bs,1,1), self.t2y(t), self.v2y(v)],
                dim=2,
            )
            yy = self.y_vtp(yy.transpose(1, 2))
            if comb_random < 0.25: #vt
                input_embed_dict[x] = xx
                input_embed_dict[y] = yy
                loss_rec = 0.0
            elif comb_random < 0.5: #vtx
                input_embed_dict[y] = yy
                loss_rec = self.gen_loss_weight * self.MSE(xx, input_embed_dict[x].detach())
            elif comb_random < 0.75: #vty
                input_embed_dict[x] = xx
                loss_rec = self.gen_loss_weight * self.MSE(yy, input_embed_dict[y].detach())
            else: #vtxy
                loss_rec = self.gen_loss_weight * (self.MSE(xx, input_embed_dict[x].detach()) + self.MSE(yy, input_embed_dict[y].detach()))
            input_att_dict[x] = input_att_dict['rgb'].clone()
            input_att_dict[y] = input_att_dict['rgb'].clone()
        elif self.missing_mode == 3:
            t_bs = input_text_embeds.size(0)
            # x_bs = 16 # pc
            x_bs = 64
            t = input_text_embeds.repeat(4,1,1).transpose(1, 2)
            t_forx = input_text_embeds.repeat(int(x_bs/t_bs),1,1).transpose(1, 2)
            v = input_embed_dict['rgb'].transpose(1, 2)
            v_bs = v.size(0)
            v_forx = v.view(x_bs, int(v_bs/x_bs), v.size(-2), v.size(-1)).mean(dim=1)
            xx = torch.cat(
                    [self.generative_prompt.repeat(x_bs,1,1), self.t2x(t), self.v2x(v)],
                    dim=2,
            )
            xx = self.x_vtp(xx.transpose(1, 2))
            yy = torch.cat(
                [self.generative_prompt.repeat(v_bs,1,1), self.t2y(t), self.v2y(v)],
                dim=2,
            )
            yy = self.y_vtp(yy.transpose(1, 2))
            zz = torch.cat(
                [self.generative_prompt.repeat(v_bs,1,1), self.t2z(t), self.v2z(v)],
                dim=2,
            )
            zz = self.z_vtp(zz.transpose(1, 2))
            if comb_random < 1/8: #vt
                input_embed_dict[x] = xx
                input_embed_dict[y] = yy
                input_embed_dict[z] = zz
                loss_rec = 0.0
            elif comb_random < 1/4: #vtx
                input_embed_dict[y] = yy
                input_embed_dict[z] = zz
                loss_rec = self.gen_loss_weight * self.MSE(xx, input_embed_dict[x].detach())
            elif comb_random < 3/8: #vty
                input_embed_dict[x] = xx
                input_embed_dict[z] = zz
                loss_rec = self.gen_loss_weight * self.MSE(yy, input_embed_dict[y].detach())
            elif comb_random < 1/2: #vtz
                input_embed_dict[x] = xx
                input_embed_dict[y] = yy
                loss_rec = self.gen_loss_weight * self.MSE(zz, input_embed_dict[z].detach())
            elif comb_random < 5/8: #vtxy
                input_embed_dict[z] = zz
                loss_rec = self.gen_loss_weight * (self.MSE(xx, input_embed_dict[x].detach()) + self.MSE(yy, input_embed_dict[y].detach()))
            elif comb_random < 3/4: #vtxz
                input_embed_dict[y] = yy
                loss_rec = self.gen_loss_weight * (self.MSE(xx, input_embed_dict[x].detach()) + self.MSE(zz, input_embed_dict[z].detach()))
            elif comb_random < 7/8: #vtyz
                input_embed_dict[x] = xx
                loss_rec = self.gen_loss_weight * (self.MSE(yy, input_embed_dict[y].detach()) + self.MSE(zz, input_embed_dict[z].detach()))
            else: #vtxyz
                loss_rec = self.gen_loss_weight * (self.MSE(xx, input_embed_dict[x].detach()) + self.MSE(yy, input_embed_dict[y].detach()) + self.MSE(zz, input_embed_dict[z].detach()))
            # input_att_dict[x] = input_att_dict['rgb'].clone()
            input_att_dict[x] = torch.ones((x_bs, self.xlen), dtype=torch.long, device=input_att_dict['rgb'].device)
            input_att_dict[y] = input_att_dict['rgb'].clone()
            input_att_dict[z] = input_att_dict['rgb'].clone()
        elif self.missing_mode == 4:
            t_bs = input_text_embeds.size(0)
            x_bs = 32 # audio
            t = input_text_embeds.repeat(4,1,1).transpose(1, 2)
            t_forx = input_text_embeds.repeat(int(x_bs/t_bs),1,1).transpose(1, 2)
            v = input_embed_dict['rgb'].transpose(1, 2)
            v_bs = v.size(0)
            v_forx = v.view(x_bs, int(v_bs/x_bs), v.size(-2), v.size(-1)).mean(dim=1)
            xx = torch.cat(
                    [self.generative_prompt.repeat(x_bs,1,1), self.t2x(t), self.v2x(v)],
                    dim=2,
            )
            xx = self.x_vtp(xx.transpose(1, 2))
            yy = torch.cat(
                [self.generative_prompt.repeat(v_bs,1,1), self.t2y(t), self.v2y(v)],
                dim=2,
            )
            yy = self.y_vtp(yy.transpose(1, 2))
            zz = torch.cat(
                [self.generative_prompt.repeat(v_bs,1,1), self.t2z(t), self.v2z(v)],
                dim=2,
            )
            zz = self.z_vtp(zz.transpose(1, 2))
            mm = torch.cat(
                [self.generative_prompt.repeat(v_bs,1,1), self.t2m(t), self.v2m(v)],
                dim=2,
            )
            mm = self.m_vtp(mm.transpose(1, 2))
            if comb_random < 1/16: #vt
                
                input_embed_dict[x] = xx
                input_embed_dict[y] = yy
                input_embed_dict[z] = zz
                input_embed_dict[m] = mm
                loss_rec = 0.0
                
            elif comb_random < 1/8: #vtx
                
                input_embed_dict[y] = yy
                input_embed_dict[z] = zz
                input_embed_dict[m] = mm
                loss_rec = self.gen_loss_weight * self.MSE(xx, input_embed_dict[x].detach())
            elif comb_random < 3/16: #vty
                
                input_embed_dict[x] = xx
                input_embed_dict[z] = zz
                input_embed_dict[m] = mm
                loss_rec = self.gen_loss_weight * self.MSE(yy, input_embed_dict[y].detach())
            elif comb_random < 1/4: #vtz
                
                input_embed_dict[x] = xx
                input_embed_dict[y] = yy
                input_embed_dict[m] = mm
                loss_rec = self.gen_loss_weight * self.MSE(zz, input_embed_dict[z].detach())
            elif comb_random < 5/16: #vtm
                
                input_embed_dict[x] = xx
                input_embed_dict[y] = yy
                input_embed_dict[z] = zz
                loss_rec = self.gen_loss_weight * self.MSE(mm, input_embed_dict[m].detach())
            elif comb_random < 3/8: #vtxy
                
                input_embed_dict[z] = zz
                input_embed_dict[m] = mm
                loss_rec = self.gen_loss_weight * (self.MSE(xx, input_embed_dict[x].detach()) 
                                                   + self.MSE(yy, input_embed_dict[y].detach()))
            elif comb_random < 7/16: #vtxz
                
                input_embed_dict[y] = yy
                input_embed_dict[m] = mm
                loss_rec = self.gen_loss_weight * (self.MSE(xx, input_embed_dict[x].detach()) 
                                                   + self.MSE(zz, input_embed_dict[z].detach()))
            elif comb_random < 1/2:  #vtxm
                
                input_embed_dict[y] = yy
                input_embed_dict[z] = zz
                loss_rec = self.gen_loss_weight * (self.MSE(xx, input_embed_dict[x].detach()) 
                                                   + self.MSE(mm, input_embed_dict[m].detach()))
            elif comb_random < 9/16: #vtyz
                
                input_embed_dict[x] = xx
                input_embed_dict[m] = mm
                loss_rec = self.gen_loss_weight * (self.MSE(yy, input_embed_dict[y].detach()) 
                                                   + self.MSE(zz, input_embed_dict[z].detach()))
            elif comb_random < 5/8:  #vtym
                
                input_embed_dict[x] = xx
                input_embed_dict[z] = zz
                loss_rec = self.gen_loss_weight * (self.MSE(yy, input_embed_dict[y].detach()) 
                                                   + self.MSE(mm, input_embed_dict[m].detach()))
            elif comb_random < 11/16: #vtzm
                
                input_embed_dict[x] = xx
                input_embed_dict[y] = yy
                loss_rec = self.gen_loss_weight * (self.MSE(zz, input_embed_dict[z].detach()) 
                                                   + self.MSE(mm, input_embed_dict[m].detach()))
            elif comb_random < 3/4: #vtxyz
                
                input_embed_dict[m] = mm
                loss_rec = self.gen_loss_weight * (self.MSE(xx, input_embed_dict[x].detach()) 
                                                   + self.MSE(yy, input_embed_dict[y].detach()) 
                                                   + self.MSE(zz, input_embed_dict[z].detach()))
            elif comb_random < 13/16:
                
                # modalities = ['rgb', x, y, m] 
                input_embed_dict[z] = zz
                loss_rec = self.gen_loss_weight * (self.MSE(xx, input_embed_dict[x].detach()) 
                                                   + self.MSE(yy, input_embed_dict[y].detach()) 
                                                   + self.MSE(mm, input_embed_dict[m].detach()))
            elif comb_random < 7/8:
                
                # modalities = ['rgb', x, z, m] 
                input_embed_dict[y] = yy
                loss_rec = self.gen_loss_weight * (self.MSE(xx, input_embed_dict[x].detach()) 
                                                   + self.MSE(zz, input_embed_dict[z].detach()) 
                                                   + self.MSE(mm, input_embed_dict[m].detach()))
            elif comb_random < 15/16:
                
                # modalities = ['rgb', y, z, m] 
                input_embed_dict[x] = xx
                loss_rec = self.gen_loss_weight * (self.MSE(yy, input_embed_dict[y].detach()) 
                                                   + self.MSE(zz, input_embed_dict[z].detach()) 
                                                   + self.MSE(mm, input_embed_dict[m].detach()))
            else:
                
                loss_rec = self.gen_loss_weight * (self.MSE(xx, input_embed_dict[x].detach()) 
                                                   + self.MSE(yy, input_embed_dict[y].detach()) 
                                                   + self.MSE(zz, input_embed_dict[z].detach()) 
                                                   + self.MSE(mm, input_embed_dict[m].detach()))
            
            # input_att_dict[x] = input_att_dict['rgb'].clone()
            input_att_dict[x] = torch.ones((x_bs, self.xlen), dtype=torch.long, device=input_att_dict['rgb'].device)
            input_att_dict[y] = input_att_dict['rgb'].clone()
            input_att_dict[z] = input_att_dict['rgb'].clone()
            input_att_dict[m] = input_att_dict['rgb'].clone()
        else:
            raise NotImplementedError

        fusion_modal = []
        t5_inputs, t5_atts, t5_query = {}, {}, {}
        for modal in self.modalities:
            t5_inputs[modal], t5_atts[modal], t5_query[modal] = self.get_qformer_embedding(input_embed_dict[modal], input_att_dict[modal], device, modal, b)

        if 'rgb' in self.modalities:
            inputs_t5_rgb = t5_inputs['rgb'] #[16, 4, 32, 2048] bs, frame, num_query_token, dim
            atts_t5_rgb = t5_atts['rgb']    #torch.Size([16, 4, 32]) bs, frame, num_query_token
            vid_prefix_embed, vid_prefix_mask = self.get_prefix_embedding(self.vid_prefix, b, device ) #[16, 4, 4, 2048]
            inputs_t5 = torch.cat([vid_prefix_embed, inputs_t5_rgb], dim=2) # b, t, n_word + m, c
            atts_t5 = torch.cat([vid_prefix_mask, atts_t5_rgb], dim=2) # b, t, n_word + m 
            
            MI_loss=0.
            for modal in self.modalities:
                if modal == 'rgb':
                    continue
                if modal in ['depth', 'norm', 'flow']:
                    if 'espresso' in self.task:
                        fusion_modal.append(t5_inputs[modal])
                    else:
                        inputs_t5 = torch.cat([inputs_t5, t5_inputs[modal]], dim=2)
                        atts_t5 = torch.cat([atts_t5, t5_atts[modal]], dim=2)
                
                if modal in ['pc']:
                    if 'espresso' in self.task:
                        pc = t5_inputs[modal]
                        pc = pc.unsqueeze(1)
                        pc = torch.repeat_interleave(pc, self.frame_num, 1)
                        fusion_modal.append(pc)

                if modal in ['audio']:
                    if 'espresso' in self.task:
                        audio = t5_inputs[modal]
                        audio = audio.mean(dim=1)
                        audio = audio.unsqueeze(1)
                        audio = torch.repeat_interleave(audio, self.frame_num, 1) 
                        fusion_modal.append(audio)
            
            # visual only input
            if 'audio' not in self.modalities and 'pc' not in self.modalities:
                if 'espresso' in self.task:
                    fusion_modal = torch.cat(fusion_modal, dim=-1) 
                    inputs_t5_extra = self.fusion(fusion_modal) #same dim as fusion modal
                    if 'concat' in self.task:
                        inputs_t5 = inputs_t5.reshape(b, -1, inputs_t5.shape[-1]) #torch.Size([16, 144, 2048]), bs,frame*qformer,dim
                        atts_t5 = atts_t5.reshape(b, -1)
                        
                        inputs_t5_extra = self.sigmoid(inputs_t5_extra) * inputs_t5_extra #activate 
                        vpe = vid_prefix_embed.reshape(b, -1, vid_prefix_embed.shape[-1]) #bs,frame*qformer,dim
                        inputs_t5_extra = inputs_t5_extra.reshape(b, -1, inputs_t5_extra.shape[-1])
                        inputs_t5 = torch.cat([inputs_t5, vpe, inputs_t5_extra], dim=1)
                        atts_t5 = torch.cat([atts_t5, atts_t5], dim=1)
                    else: #将激活后的 inputs_t5_extra 加到 inputs_t5_rgb 上（融合信息），然后在特征维度与 vid_prefix_embed 拼接
                        inputs_t5_rgb += self.sigmoid(inputs_t5_extra) * inputs_t5_extra
                        inputs_t5 = torch.cat([vid_prefix_embed, inputs_t5_rgb], dim=2)
                        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)
                        inputs_t5 = inputs_t5.reshape(b, -1, inputs_t5.shape[-1])
                        atts_t5 = atts_t5.reshape(b, -1)                            
                else:
                    inputs_t5 = inputs_t5.reshape(b, -1, inputs_t5.shape[-1])
                    atts_t5 = atts_t5.reshape(b, -1)
            
            # [F1, F2, F3,..., A]
            elif 'audio' in self.modalities:
                if 'espresso' in self.task:
                    
                    fusion_modal = torch.cat(fusion_modal, dim=-1)
                    inputs_t5_extra = self.fusion(fusion_modal)
                    if 'concat' in self.task:
                        inputs_t5 = inputs_t5.reshape(b, -1, inputs_t5.shape[-1])
                        atts_t5 = atts_t5.reshape(b, -1)
                        
                        inputs_t5_extra = self.sigmoid(inputs_t5_extra) * inputs_t5_extra
                        vpe = vid_prefix_embed.reshape(b, -1, vid_prefix_embed.shape[-1])
                        inputs_t5_extra = inputs_t5_extra.reshape(b, -1, inputs_t5_extra.shape[-1])
                        inputs_t5 = torch.cat([inputs_t5, vpe, inputs_t5_extra], dim=1)
                        atts_t5 = torch.cat([atts_t5, atts_t5], dim=1)
                    else:
                        inputs_t5_rgb += self.sigmoid(inputs_t5_extra) * inputs_t5_extra
                        inputs_t5 = torch.cat([vid_prefix_embed, inputs_t5_rgb], dim=2)
                        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)
                        inputs_t5 = inputs_t5.reshape(b, -1, inputs_t5.shape[-1])
                        atts_t5 = atts_t5.reshape(b, -1)                       
                else:
                    audio_prefix_embed, audio_prefix_mask = self.get_prefix_embedding(self.audio_prefix, b, device) 
                    inputs_t5 = inputs_t5.reshape(b, -1, inputs_t5.shape[-1])
                    atts_t5 = atts_t5.reshape(b, -1)
                    # seems no prefix works better for audio-video reasoning
                    inputs_t5 = torch.cat([inputs_t5, t5_inputs['audio'].reshape(b, -1, t5_inputs['audio'].shape[-1])], dim=1)
                    atts_t5 = torch.cat([atts_t5, t5_atts['audio'].reshape(b, -1)], dim=1)
            
            elif 'pc' in self.modalities:                
                if 'espresso' in self.task:                                        
                    fusion_modal = torch.cat(fusion_modal, dim=-1)
                    inputs_t5_extra = self.fusion(fusion_modal)
                    if 'concat' in self.task:
                        inputs_t5 = inputs_t5.reshape(b, -1, inputs_t5.shape[-1])
                        atts_t5 = atts_t5.reshape(b, -1)
                        
                        inputs_t5_extra = self.sigmoid(inputs_t5_extra) * inputs_t5_extra
                        vpe = vid_prefix_embed.reshape(b, -1, vid_prefix_embed.shape[-1])
                        inputs_t5_extra = inputs_t5_extra.reshape(b, -1, inputs_t5_extra.shape[-1])
                        inputs_t5 = torch.cat([inputs_t5, vpe, inputs_t5_extra], dim=1)
                        atts_t5 = torch.cat([atts_t5, atts_t5], dim=1)
                    else:
                        inputs_t5_rgb += self.sigmoid(inputs_t5_extra) * inputs_t5_extra
                        inputs_t5 = torch.cat([vid_prefix_embed, inputs_t5_rgb], dim=2)
                        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)
                        inputs_t5 = inputs_t5.reshape(b, -1, inputs_t5.shape[-1])
                        atts_t5 = atts_t5.reshape(b, -1)                    
                else:
                    pc_prefix_embed, pc_prefix_mask = self.get_prefix_embedding(self.pc_prefix, b, device) 
                    inputs_t5 = inputs_t5.reshape(b, -1, inputs_t5.shape[-1])
                    atts_t5 = atts_t5.reshape(b, -1)
                    inputs_t5 = torch.cat([inputs_t5, pc_prefix_embed.squeeze(), t5_inputs['pc'].reshape(b, -1, t5_inputs['pc'].shape[-1])], dim=1)
                    atts_t5 = torch.cat([atts_t5, pc_prefix_mask.squeeze(), t5_atts['pc'].reshape(b, -1)], dim=1)
                    
        elif 'audio' in self.modalities: # audio 
            inputs_t5 = t5_inputs['audio'].reshape(b, -1, t5_inputs['audio'].shape[-1])
            atts_t5 = t5_atts['audio'].reshape(b, -1)
            audio_prefix_embed, audio_prefix_mask = self.get_prefix_embedding(self.audio_prefix, b, device) 
            inputs_t5 = torch.cat([audio_prefix_embed.squeeze(), inputs_t5], dim=1)
            atts_t5 = torch.cat([audio_prefix_mask.squeeze(), atts_t5], dim=1)
        
        elif 'pc' in self.modalities: # pc
            inputs_t5 = t5_inputs['pc'].reshape(b, -1, t5_inputs['pc'].shape[-1])
            atts_t5 = t5_atts['pc'].reshape(b, -1)
            pc_prefix_embed, pc_prefix_mask = self.get_prefix_embedding(self.audio_prefix, b, device) 
            inputs_t5 = torch.cat([pc_prefix_embed.squeeze(), inputs_t5], dim=1)
            atts_t5 = torch.cat([pc_prefix_mask.squeeze(), atts_t5], dim=1)


        inputs_embeds = torch.cat([inputs_t5, input_text_embeds], dim=1)
        encoder_atts = torch.cat([atts_t5, input_text.attention_mask], dim=1)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):

            output_tokens = self.t5_tokenizer(
                answer, padding="longest", truncation=True,
                max_length=self.max_txt_len, return_tensors="pt").to(device)
            targets_qa = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100)
            output_tokens_mask = output_tokens.attention_mask
                
            outputs = self.t5_model(
                    inputs_embeds=inputs_embeds, attention_mask=encoder_atts,
                    decoder_attention_mask=output_tokens_mask, return_dict=True, labels=targets_qa)
            loss = outputs.loss
                    
        return {'loss': loss+MI_loss+loss_rec}
    
    def encode_input(self, input, modality, training=True):

        ln = getattr(self, f"ln_{modality}")

        if modality in ['rgb', 'depth', 'flow', 'norm']:
            modality = 'visual'
        if modality in ['audio']:
            modality = 'audio'
        if modality in ['pc']:
            modality = 'pc'

        encoder = getattr(self, f"{modality}_encoder")

        if modality == 'visual':
            b, t, c, w, h = input.shape     
            input = input.reshape(-1, c, w, h)
            if training:
                image_embeds = ln(encoder(input))
            else:
                with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu"))):
                    image_embeds = ln(encoder(input))
            _, n, _ = image_embeds.shape
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(input.device) # bt n c

            # image_embeds_patches = image_embeds[:, 1:, :]
            # compressed_embeds, _ = self.compressor(image_embeds_patches)
            # image_embeds = torch.cat((image_embeds[:, :1, :], compressed_embeds), dim=1)
            # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(input.device) 
            return image_embeds, image_atts
        
        if modality == 'audio':
            embeds, atts = [], []
            for j in range(input.size(1)):
                this_frame = input[:,j,:,:]
                if training:
                    embeds.append(encoder(this_frame))
                else:
                    with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu"))):
                        embeds.append(encoder(this_frame))
                atts.append(torch.ones(embeds[j].size()[:-1], dtype=torch.long).to(input.device))
            
            embeds = torch.stack(embeds, dim=1)
            atts = torch.stack(atts, dim=1)
            embeds = self.projection_audio(embeds) 
            embeds = ln(embeds.reshape(-1, embeds.shape[-2], embeds.shape[-1]))
            atts = atts.reshape(-1, atts.shape[-1])

            return embeds, atts
        
        if modality == 'pc':
            # use pre-extracted features
            pass
            #return embeds, atts
    
    def get_qformer_embedding(self, embeds, atts, device, modality, bs):

        project = getattr(self, f"t5_proj_{modality}")
        query_tokens = getattr(self, f"query_tokens_{modality}")
        query_tokens = query_tokens.expand(embeds.shape[0], -1, -1)

        skip_flag = self.skip[self.modalities.index(modality)]
        modality_ = modality + skip_flag

        query_output = self.Qformer.bert(
            query_embeds=query_tokens, encoder_hidden_states=embeds,
            encoder_attention_mask=atts, return_dict=True, modular=modality_)
        
        query = query_output.last_hidden_state.clone()
        inputs_t5 = project(query_output.last_hidden_state)

        if modality in ['rgb', 'depth', 'flow', 'norm']:
            inputs_t5 = inputs_t5.reshape(-1, self.frame_num, inputs_t5.shape[-2], inputs_t5.shape[-1])
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)
        
        if modality in ['audio']:
            inputs_t5 = inputs_t5.reshape(bs, -1, inputs_t5.shape[-2], inputs_t5.shape[-1])
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)
        
        if modality in ['pc']:
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)

        return inputs_t5, atts_t5, query
    
    def get_prefix_embedding(self, prefix_, b, device):
        prefix = self.t5_tokenizer(
                    prefix_, padding="longest", add_special_tokens=False,
                    truncation=True, max_length=self.max_txt_len, return_tensors="pt",).to(device) # 
        prefix_id = torch.repeat_interleave(prefix.input_ids.unsqueeze(0), b, 0)
        prefix_mask = torch.repeat_interleave(prefix.attention_mask.unsqueeze(0), b, 0)
        prefix_embed = self.t5_model.encoder.embed_tokens(prefix_id) # b t n_word c
        return prefix_embed, prefix_mask
        
    @torch.no_grad()
    def generate(self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5, max_length=30,
        min_length=1, top_p=0.9,
        repetition_penalty=1.0, length_penalty=1.0,
        num_captions=1, temperature=1,):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        out = {}
        qid = samples['question_id']
        qa_text = samples['qa_input']
        answer = samples['qa_output']
        b = len(qa_text)

        input_embed_dict, input_att_dict = {}, {}

        for modal in self.modalities:
            input = samples[modal]
            # visual modality pre-process
            if modal in ['rgb', 'depth', 'norm', 'flow']:
                if input.shape[1] == 3:
                    input = input.permute(0, 2, 1, 3, 4)
            # 3d: direct load pre-processed features
            if modal == 'pc':
                with torch.cuda.amp.autocast(dtype=torch.float32):
                    pc_embeds = samples["pc_feat"]
                    pc = samples["pc"].long()
                    all_pcs = torch.zeros((pc_embeds.shape))
                    for j in range(pc.shape[0]):
                        pcs = []
                        for i in range(3):
                            pc_i = pc[j][:, i]
                            pcs.append(self.pos_embedding[pc_i])
                        pcs = torch.cat(pcs, -1)
                        all_pcs[j][:, :1407] = pcs
                    all_pcs = all_pcs.cuda()
                pc_embeds = pc_embeds + 0.01 * all_pcs
                atts = torch.ones(pc_embeds.size()[:-1], dtype=torch.long).to(pc_embeds.device)
                input_embed_dict[modal], input_att_dict[modal] = pc_embeds, atts
            else:
                input_embed_dict[modal], input_att_dict[modal] = self.encode_input(input, modal, training=False)
        
        device = input_embed_dict[list(input_embed_dict.keys())[0]].device
        fusion_modal = []
        input_text= self.t5_tokenizer(
                qa_text, padding="longest", truncation=True,
                max_length=self.max_txt_len, return_tensors="pt").to(device)
        input_text_embeds = self.t5_model.encoder.embed_tokens(input_text.input_ids) 

        with torch.no_grad():
            
            t5_inputs, t5_atts, t5_query = {}, {}, {}
            for modal in self.modalities:
                t5_inputs[modal], t5_atts[modal], t5_query[modal] = self.get_qformer_embedding(input_embed_dict[modal], input_att_dict[modal], device, modal, b)
            
            if 'rgb' not in self.modalities: # Show why cannot drop V: show TD CREMA	
                if len(self.modalities) == 1:

                    vid_prefix_embed, vid_prefix_mask = self.get_prefix_embedding(self.vid_prefix, b, device)
                    inputs_t5 = vid_prefix_embed
                    atts_t5 = vid_prefix_mask

                    if 'espresso' in self.task and 'concat' in self.task:

                        inputs_t5 = inputs_t5.reshape(b, -1, inputs_t5.shape[-1]) #actually vpe
                        
                        atts_t5 = torch.cat([vid_prefix_mask, t5_atts[modal]], dim=2) # b, t, n_word + m 
                        atts_t5 = atts_t5.reshape(b, -1) #torch.Size([16, 16])

                        fusion_modal.append(t5_inputs[modal])
                        fusion_modal = torch.cat(fusion_modal, dim=-1) 
                        inputs_t5_extra = self.fusion(fusion_modal) #same dim as fusion modal
                        inputs_t5_extra = self.sigmoid(inputs_t5_extra) * inputs_t5_extra #activate 
                        inputs_t5_extra = inputs_t5_extra.reshape(b, -1, inputs_t5_extra.shape[-1])
                        inputs_t5 = torch.cat([inputs_t5, inputs_t5_extra], dim=1)
                    
                        MI_loss=0.
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
            
            # different modality combination
            if 'rgb' in self.modalities:
                inputs_t5_rgb = t5_inputs['rgb']
                atts_t5_rgb = t5_atts['rgb']
                vid_prefix_embed, vid_prefix_mask = self.get_prefix_embedding(self.vid_prefix, b, device)
                inputs_t5 = torch.cat([vid_prefix_embed, inputs_t5_rgb], dim=2) # b, t, n_word + m, c
                atts_t5 = torch.cat([vid_prefix_mask, atts_t5_rgb], dim=2) # b, t, n_word + m 
                for modal in self.modalities:
                    if modal == 'rgb':
                        continue
                    if modal in ['depth', 'norm', 'flow']:
                        if 'espresso' in self.task:
                            fusion_modal.append(t5_inputs[modal])
                        else:
                            inputs_t5 = torch.cat([inputs_t5, t5_inputs[modal]], dim=2)
                            atts_t5 = torch.cat([atts_t5, t5_atts[modal]], dim=2)  
                    if modal in ['pc']:
                        if 'espresso' in self.task:
                            pc = t5_inputs[modal]
                            pc = pc.unsqueeze(1)
                            pc = torch.repeat_interleave(pc, self.frame_num, 1) 
                            fusion_modal.append(pc)

                    if modal in ['audio']:
                        if 'espresso' in self.task:
                            audio = t5_inputs[modal]
                            audio = audio.mean(dim=1)
                            audio = audio.unsqueeze(1)
                            audio = torch.repeat_interleave(audio, self.frame_num, 1) 
                            fusion_modal.append(audio)
                        
                # visual only input
                if 'audio' not in self.modalities and 'pc' not in self.modalities:
                    if 'espresso' in self.task:
                        fusion_modal = torch.cat(fusion_modal, dim=-1)
                        inputs_t5_extra = self.fusion(fusion_modal)
                        
                        if 'concat' in self.task:
                            inputs_t5 = inputs_t5.reshape(b, -1, inputs_t5.shape[-1])
                            atts_t5 = atts_t5.reshape(b, -1)
                            
                            inputs_t5_extra = self.sigmoid(inputs_t5_extra) * inputs_t5_extra
                            vpe = vid_prefix_embed.reshape(b, -1, vid_prefix_embed.shape[-1])
                            inputs_t5_extra = inputs_t5_extra.reshape(b, -1, inputs_t5_extra.shape[-1])
                            inputs_t5 = torch.cat([inputs_t5, vpe, inputs_t5_extra], dim=1)
                            atts_t5 = torch.cat([atts_t5, atts_t5], dim=1)                        
                        else:
                            inputs_t5_rgb += self.sigmoid(inputs_t5_extra) * inputs_t5_extra                        
                            inputs_t5 = torch.cat([vid_prefix_embed, inputs_t5_rgb], dim=2)
                            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)
                            inputs_t5 = inputs_t5.reshape(b, -1, inputs_t5.shape[-1])
                            atts_t5 = atts_t5.reshape(b, -1)
                        
                        
                        
                    else:
                        inputs_t5 = inputs_t5.reshape(b, -1, inputs_t5.shape[-1])
                        atts_t5 = atts_t5.reshape(b, -1)
                # [F1, F2, F3,..., A]
                elif 'audio' in self.modalities:
                    if 'espresso' in self.task:
                        fusion_modal = torch.cat(fusion_modal, dim=-1) # 16, 4, 32, 8192
                        inputs_t5_extra = self.fusion(fusion_modal)
                        if 'concat' in self.task:
                            inputs_t5 = inputs_t5.reshape(b, -1, inputs_t5.shape[-1])
                            atts_t5 = atts_t5.reshape(b, -1)
                            
                            inputs_t5_extra = self.sigmoid(inputs_t5_extra) * inputs_t5_extra
                            vpe = vid_prefix_embed.reshape(b, -1, vid_prefix_embed.shape[-1])
                            inputs_t5_extra = inputs_t5_extra.reshape(b, -1, inputs_t5_extra.shape[-1])
                            inputs_t5 = torch.cat([inputs_t5, vpe, inputs_t5_extra], dim=1)
                            atts_t5 = torch.cat([atts_t5, atts_t5], dim=1)                        
                        else:
                            inputs_t5_rgb += self.sigmoid(inputs_t5_extra) * inputs_t5_extra                        
                            inputs_t5 = torch.cat([vid_prefix_embed, inputs_t5_rgb], dim=2)
                            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)
                            inputs_t5 = inputs_t5.reshape(b, -1, inputs_t5.shape[-1])
                            atts_t5 = atts_t5.reshape(b, -1)
                    else:
                        audio_prefix_embed, audio_prefix_mask = self.get_prefix_embedding(self.audio_prefix, b, device) 
                        inputs_t5 = inputs_t5.reshape(b, -1, inputs_t5.shape[-1])
                        atts_t5 = atts_t5.reshape(b, -1)
                        inputs_t5 = torch.cat([inputs_t5, t5_inputs['audio'].reshape(b, -1, t5_inputs['audio'].shape[-1])], dim=1)
                        atts_t5 = torch.cat([atts_t5, t5_atts['audio'].reshape(b, -1)], dim=1)
                
                elif 'pc' in self.modalities:
                    if 'espresso' in self.task:
                        fusion_modal = torch.cat(fusion_modal, dim=-1) # 16, 4, 32, 8192
                        inputs_t5_extra = self.fusion(fusion_modal)
                        if 'concat' in self.task:
                            inputs_t5 = inputs_t5.reshape(b, -1, inputs_t5.shape[-1])
                            atts_t5 = atts_t5.reshape(b, -1)
                            
                            inputs_t5_extra = self.sigmoid(inputs_t5_extra) * inputs_t5_extra
                            vpe = vid_prefix_embed.reshape(b, -1, vid_prefix_embed.shape[-1])
                            inputs_t5_extra = inputs_t5_extra.reshape(b, -1, inputs_t5_extra.shape[-1])
                            inputs_t5 = torch.cat([inputs_t5, vpe, inputs_t5_extra], dim=1)
                            atts_t5 = torch.cat([atts_t5, atts_t5], dim=1)                        
                        else:
                            inputs_t5_rgb += self.sigmoid(inputs_t5_extra) * inputs_t5_extra                        
                            inputs_t5 = torch.cat([vid_prefix_embed, inputs_t5_rgb], dim=2)
                            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)
                            inputs_t5 = inputs_t5.reshape(b, -1, inputs_t5.shape[-1])
                            atts_t5 = atts_t5.reshape(b, -1)
                    else:
                        pc_prefix_embed, pc_prefix_mask = self.get_prefix_embedding(self.pc_prefix, b, device) 
                        inputs_t5 = inputs_t5.reshape(b, -1, inputs_t5.shape[-1])
                        atts_t5 = atts_t5.reshape(b, -1)
                        inputs_t5 = torch.cat([inputs_t5, pc_prefix_embed.squeeze(), t5_inputs['pc'].reshape(b, -1, t5_inputs['pc'].shape[-1])], dim=1)
                        atts_t5 = torch.cat([atts_t5, pc_prefix_mask.squeeze(), t5_atts['pc'].reshape(b, -1)], dim=1)
                    
            elif 'audio' in self.modalities: # audio only
                inputs_t5 = t5_inputs['audio'].reshape(b, -1, t5_inputs['audio'].shape[-1])
                atts_t5 = t5_atts['audio'].reshape(b, -1)
                audio_prefix_embed, audio_prefix_mask = self.get_prefix_embedding(self.audio_prefix, b, device) 
                inputs_t5 = torch.cat([audio_prefix_embed.squeeze(), inputs_t5], dim=1)
                atts_t5 = torch.cat([audio_prefix_mask.squeeze(), atts_t5], dim=1)
            
            elif 'pc' in self.modalities: # pc only
                inputs_t5 = t5_inputs['pc'].reshape(b, -1, t5_inputs['pc'].shape[-1])
                atts_t5 = t5_atts['pc'].reshape(b, -1)
                pc_prefix_embed, pc_prefix_mask = self.get_prefix_embedding(self.audio_prefix, b, device) 
                inputs_t5 = torch.cat([pc_prefix_embed.squeeze(), inputs_t5], dim=1)
                atts_t5 = torch.cat([pc_prefix_mask.squeeze(), atts_t5], dim=1)            
            
            inputs_embeds = torch.cat([inputs_t5, input_text_embeds], dim=1)
            encoder_atts = torch.cat([atts_t5, input_text.attention_mask], dim=1)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                
                if self.downstream_task == 'mcqa':
                    outputs = self.t5_model.generate(
                        inputs_embeds=inputs_embeds, attention_mask=encoder_atts,
                        do_sample=use_nucleus_sampling, top_p=top_p,
                        temperature=temperature, num_beams=1,
                        max_new_tokens=max_length, min_length=min_length,
                        repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                        num_return_sequences=num_captions, return_dict_in_generate=True,
                        output_hidden_states=True, output_scores=True)
                    try:
                        pred_logits = outputs.scores[1]
                    except:
                        pred_logits = outputs.scores[0]
                    pred_logits = pred_logits[:, self.answer_id] # b, 5
                    pred_ans = torch.argmax(pred_logits, dim=-1).cpu().tolist() 

                elif self.downstream_task == 'oeqa':
                    outputs = self.t5_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=encoder_atts,
                        do_sample=False,
                        num_beams=num_beams,
                        max_new_tokens=max_length,
                        min_length=min_length,
                        length_penalty=length_penalty,
                        )
                    pred_ans = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
        out['output_text'] = pred_ans
        out['answer'] = answer
        out['qid'] = qid

        return out

    @torch.no_grad()
    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        **kwargs
        ):
        if isinstance(samples["qa_input"], str):
            samples["qa_input"] = [samples["qa_input"]]
        
        text_input = samples["qa_input"]
        samples["prompt"] = text_input

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty
        )['output_text']

        if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
            output_text = self._lemmatize(output_text)
    
        output_text = [o if o != "" else "unanswerable" for o in output_text]
        return output_text
    
    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        frame_num = cfg.get("frame_num", 8)
        answer_num = cfg.get("answer_num", 5) 
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)
        task = cfg.get("task", 'train')
        modalities = cfg.get("modalities", 'rgb')
        downstream_task = cfg.get("downstream_task", 'mcqa')

        lora_rank = cfg.get("lora_rank", 64)
        lora_layer = cfg.get("lora_layer", None)
        lora_dropout = cfg.get("lora_dropout", 0.1)
        missing_mode = cfg.get("missing_mode", 0)
        gen_loss_weight = cfg.get("gen_loss_weight", 0.1)
        mmqa_ckpt = cfg.get("mmqa_ckpt", '')

        model = cls(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            frame_num=frame_num,
            answer_num=answer_num,
            task=task,
            downstream_task=downstream_task,
            modalities=modalities,
            lora_rank=lora_rank,
            lora_layer=lora_layer,
            lora_dropout=lora_dropout,
            missing_mode=missing_mode,
            gen_loss_weight=gen_loss_weight
        )
        
        model.load_checkpoint_from_config(cfg)
        model.load_mmqa(mmqa_ckpt)
        print_trainable_parameters(model)

        return model