"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
from collections import OrderedDict

from lavis.datasets.datasets.multimodal_classification_datasets import (
    MultimodalClassificationDataset,
)
from lavis.datasets.datasets.base_dataset import BaseDataset


class __DisplMixin:
    def displ_item(self, index):
        ann = self.annotation[index]
        vname = ann["video"]
        vpath = os.path.join(self.vis_root, vname)

        return OrderedDict(
            {"file": vpath, "question": ann["question"], "answer": ann["answer"]}
        )


class VideoQADataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def _build_class_labels(self, ans_path):
        ans2label = json.load(open(ans_path))

        self.class_labels = ans2label

    def _get_answer_label(self, answer):
        if answer in self.class_labels:
            return self.class_labels[answer]
        else:
            return len(self.class_labels)

    def __getitem__(self, index):
        assert (
            self.class_labels
        ), f"class_labels of {__class__.__name__} is not built yet."

        ann = self.annotation[index]

        vname = ann["video"]
        vpath = os.path.join(self.vis_root, vname)

        frms = self.vis_processor(vpath)
        question = self.text_processor(ann["question"])

        return {
            "video": frms,
            "text_input": question,
            "answers": self._get_answer_label(ann["answer"]),
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }

class ThermalQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, **kwargs):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.modalities = kwargs['modalities']

        for modality in self.modalities:
            setattr(self, f"{modality}_root", kwargs[f"{modality}_root"])
            setattr(self, f"{modality}_processor", kwargs[f"{modality}_processor"])
        
        self.annotation = [ann for ann in self.annotation if ann['video_id'] in self.sample_ids]
        
    def get_existing_frame_annotations(self):
        return [f.split('.')[0] for f in os.listdir(self.frame_root)]
    
    def get_existing_thermal_annotations(self):
        return [f.split('.')[0] for f in os.listdir(self.thermal_root)]

    def get_existing_depth_annotations(self):
        return [f.split('.')[0] for f in os.listdir(self.depth_root)]

    def get_frame_path(self, ann):
        return os.path.join(self.frame_root, f'{ann["question_id"]}/')
    
    def get_thermal_path(self, ann):
        return os.path.join(self.thermal_root, f'{ann["question_id"]}/')
    
    def get_depth_path(self, ann):
        return os.path.join(self.depth_root, f'{ann["question_id"]}/')
    
    def __getitem__(self, index):
        
        result, flow_flag = None, False
        out = {}

        while result is None:
            ann = self.annotation[index]
            qid = ann['question_id'] 
            q = ann['question']

            if 'start' in ann:
                start, end = float(ann['start']), float(ann['end'])
                clip = [start, end]
            else:
                clip = None  

            qa_prompt = 'Based on the frames and thermal heatmap information, answer the question using a single word or phrase.' + q
            answers = ann['answer']
            duration = 1

            for modality in self.modalities:
                if modality == 'frame':
                    indices, clip = None, None
                    ann[f"{modality}_path"] = getattr(self, f"get_{modality}_path")(ann)
                    frms, indices = getattr(self, f"{modality}_processor")(ann[f"{modality}_path"], clip_proposal=clip, indices=indices)
                    rgb = frms.permute(1, 0, 2, 3)
                    assert len(rgb) == getattr(self, f"{modality}_processor").n_frms
                    ann['rgb'] = rgb
                
                if modality == 'depth':
                    assert indices is not None
                    ann[f"{modality}_path"] = getattr(self, f"get_{modality}_path")(ann)
                    depth, _ = getattr(self, f"{modality}_processor")(ann[f"{modality}_path"], clip_proposal=clip, indices=indices, type='depth')
                    ann['depth'] = depth
                
                if modality == 'thermal':
                    assert indices is not None
                    ann[f"{modality}_path"] = getattr(self, f"get_{modality}_path")(ann)
                    thermal, _ = getattr(self, f"{modality}_processor")(ann[f"{modality}_path"], clip_proposal=clip, indices=indices, type='thermal')
                    ann['thermal'] = thermal
                       
            out['qa_input'] = qa_prompt
            out['qa_output'] = answers
            out['question_id'] = qid
            out['duration'] = duration
            
        return out