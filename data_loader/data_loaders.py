from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import Sampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseDataLoader
import pickle
import numpy as np
import json
import os
from tqdm import tqdm
from torch._six import container_abcs, string_classes, int_classes, FileNotFoundError
import re
import random
import copy

import pdb

def load_word_dict(file_name, anony=False):
    with open(file_name, 'r') as f:
        tmp = json.load(f)
    word2vec = []
    word2id = {}
    for x in tmp:
        if x['word'] not in word2id:
            word2id[x['word']] = len(word2id)
            word2vec.append(np.array(x['vec']))
        else:
            print('WARNING: repeated token in word dict: %s' % x['word'])
    word2id['<unk>'] = len(word2id)
    word2vec.append(np.random.rand(word2vec[0].shape[0]))
    if anony:
        word2id['<head>'] = len(word2id)
        word2vec.append(np.random.rand(word2vec[0].shape[0]))
        word2id['<tail>'] = len(word2id)
        word2vec.append(np.random.rand(word2vec[0].shape[0]))
    word2id['<pad>'] = len(word2id)
    word2vec.append(np.zeros(word2vec[0].shape[0]))
    word2vec = np.stack(word2vec)
    print('word dict loaded, word vector size : {}'.format(word2vec.shape))
    return word2id, word2vec

def load_rel_dict(file_name):
    with open(file_name, 'r') as f:
        rel2id = json.load(f)
    print('relation dict loaded, found %d relations' % len(rel2id))
    return rel2id
        
    


class BaseNytDataset(Dataset):
    def _preprocess(self, data_dir):
        if self.src == self.mode:
            preprocessed_filename = os.path.join(data_dir, 'procressed', self.mode + '_%d'%(1 if self.anonymization else 0))
        else:
            preprocessed_filename = os.path.join(data_dir, 'procressed', self.src + '_' + self.mode + '_%d'%(1 if self.anonymization else 0))
        if self.select != 0:
            preprocessed_filename += '_select_%d'%self.select
        preprocessed_filename += '.pickle'
        print('try loading preprocessed data from %s'%preprocessed_filename)
        if os.path.exists(preprocessed_filename):
            with open(preprocessed_filename, 'rb') as f:
                return pickle.load(f)
        else:
            try:
                os.mkdir(os.path.join(data_dir, 'procressed'))
            except FileExistsError:
                pass
            with open(os.path.join(data_dir, self.src + '.json'), 'r') as f:
                origin_data = json.load(f)
        print("Pre-processing data...")
        fact2index = {}
        processed_data = []
        pbar = tqdm(total=len(origin_data))
        for i,ins in enumerate(origin_data):
            if ins['relation'] in self.rel2id:
                data_rel = self.rel2id[ins['relation']]
            else:
                print('WARNING: find relation not in dict: %s'%ins['relation'])
                data_rel = self.rel2id['NA']
            sentence = ' '.join(ins['sentence'].split())  # delete extra spaces
            is_extend = ins['is_extend']
            head = ins['head']['word']
            tail = ins['tail']['word']
            if self.mode == 'train':
                cur_relfact = ins['head']['id'] + '#' + ins['tail']['id'] + '#' + ins['relation']
                if cur_relfact not in fact2index:
                    fact2index[cur_relfact] = len(processed_data)
                    processed_data.append({'key':cur_relfact,'all_origin_sentences':set(),'sentence_set_origin_size':0,'relation':data_rel,'all_relations':np.zeros(len(self.rel2id)),'sentences':{'word':[],'pos1':[],'pos2':[],'mask':[],'length':[],'size':0},'extend_sentences':{'word':[],'pos1':[],'pos2':[],'mask':[],'length':[], 'size':0}})
                curr_data = processed_data[fact2index[cur_relfact]]
                curr_data['all_relations'][data_rel] = 1
            else:
                cur_entpair = ins['head']['id'] + '#' + ins['tail']['id']
                if cur_entpair not in fact2index:
                    fact2index[cur_entpair] = len(processed_data)
                    processed_data.append({'key':cur_entpair,'all_origin_sentences':set(),'sentence_set_origin_size':0,'all_relations':np.zeros(len(self.rel2id)),'sentences':{'word':[],'pos1':[],'pos2':[],'mask':[],'length':[],'size':0},'extend_sentences':{'word':[],'pos1':[],'pos2':[],'mask':[],'length':[], 'size':0}})
                curr_data = processed_data[fact2index[cur_entpair]]
                curr_data['all_relations'][data_rel] = 1
            if sentence in curr_data['all_origin_sentences']:
                continue
            else:
                curr_data['all_origin_sentences'].add(sentence)
            if is_extend == 0:
                curr_data['sentence_set_origin_size'] += 1
                curr_data = curr_data['sentences']
                if self.select != 0 and self.select <=2 and curr_data['size'] == self.select:
                    continue
            else:
                curr_data = curr_data['extend_sentences']
                if self.select >=10 and curr_data['size'] == self.select:
                    continue

            curr_data['size'] += 1
            p1 = sentence.find(' ' + head + ' ')
            p2 = sentence.find(' ' + tail + ' ')
            if p1 == -1:
                if sentence[:len(head) + 1] == head + " ":
                    p1 = 0
                elif sentence[-len(head) - 1:] == " " + head:
                    p1 = len(sentence) - len(head)
                else:
                    p1 = 0 # shouldn't happen
            else:
                p1 += 1
            if p2 == -1:
                if sentence[:len(tail) + 1] == tail + " ":
                    p2 = 0
                elif sentence[-len(tail) - 1:] == " " + tail:
                    p2 = len(sentence) - len(tail)
                else:
                    p2 = 0 # shouldn't happen
            else:
                p2 += 1

            words = sentence.split()
            data_word = np.zeros(self.max_length, dtype=np.int32)
            data_pos1 = np.arange(self.max_length, dtype=np.int32)
            data_pos2 = np.arange(self.max_length, dtype=np.int32)
            data_mask = np.zeros(self.max_length, dtype=np.int32)     
            cur_pos = 0
            pos1 = -1
            pos2 = -1
            for j, word in enumerate(words):
                if j < self.max_length:
                    if word in self.word2id:
                        data_word[j] = self.word2id[word]
                    else:
                        data_word[j] = self.word2id['<unk>']
                if cur_pos == p1:
                    pos1 = j
                    p1 = -1
                    if self.anonymization and j < self.max_length:
                        data_word[j] = self.word2id['<unk>']
                if cur_pos == p2:
                    pos2 = j
                    p2 = -1
                    if self.anonymization and j < self.max_length:
                        data_word[j] = self.word2id['<unk>']
                cur_pos += len(word) + 1
            for j in range(j + 1, self.max_length):
                data_word[j] = self.word2id['<pad>']
            data_length = len(words)
            if len(words) > self.max_length:
                data_length = self.max_length
            if pos1 == -1 or pos2 == -1:
                raise Exception("[ERROR] Position error, index = {}, sentence = {}, head = {}, tail = {}".format(i, sentence, head, tail))
            if pos1 >= self.max_length:
                pos1 = self.max_length - 1
            if pos2 >= self.max_length:
                pos2 = self.max_length - 1
            pos_min = min(pos1, pos2)
            pos_max = max(pos1, pos2)
            data_pos1 = data_pos1 - pos1 + self.max_length
            data_pos2 = data_pos2 - pos2 + self.max_length
            data_mask[data_length:] = 0
            data_mask[pos_max+1:data_length] = 3
            data_mask[pos_min+1:pos_max + 1] = 2
            data_mask[:pos_min + 1] = 1
            curr_data['word'].append(data_word)
            curr_data['pos1'].append(data_pos1)
            curr_data['pos2'].append(data_pos2)
            curr_data['mask'].append(data_mask)
            curr_data['length'].append(data_length)
            pbar.update(1)
        pbar.close()
        pbar = tqdm(total=len(processed_data))
        for sample in processed_data:
            sample.pop('all_origin_sentences')
            for x in ['sentences', 'extend_sentences']:
                if sample[x]['size'] == 0:
                    sample[x]['word'] = np.ones((0, self.max_length),dtype=np.int32)
                    sample[x]['pos1'] = np.ones((0, self.max_length),dtype=np.int32)
                    sample[x]['pos2'] = np.ones((0, self.max_length),dtype=np.int32)
                    sample[x]['mask'] = np.ones((0, self.max_length),dtype=np.int32)
                    sample[x]['length'] = np.ones((0), dtype=np.int64)
                    continue
                for y in ['word','pos1','pos2','mask','length']:
                    sample[x][y] = np.stack(sample[x][y])
            pbar.update(1)
        pbar.close()
        with open(preprocessed_filename, 'wb') as f:
            pickle.dump(processed_data, f)
        return processed_data
        
    def __init__(self, data_dir, word2id, rel2id, max_length, mode='train', src='train',select=0,anonymization=False):
        self.max_length = max_length
        self.word2id = word2id
        self.rel2id = rel2id
        self.mode = mode
        self.src = src
        self.select = select
        self.anonymization = anonymization
        self.data = self._preprocess(data_dir)
        if mode == 'train':
            self.rel2count = np.zeros(len(self.rel2id))
            for sample in self.data:
                self.rel2count[sample['relation']] += 1
                
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

def _cat_collate_fn(batch, key=None):
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        return torch.cat(batch, 0)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))
            try:
                return torch.cat([torch.from_numpy(b) for b in batch], 0)
            except RuntimeError:
                print(key)
                raise Exception
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: _cat_collate_fn([d[key] for d in batch], key) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [_cat_collate_fn(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))

class cust_collate_fn:
    def __init__(self, mode,method):
        self.mode = mode
        self.method = method
    def __call__(self, raw_batch):
        scope = [0, 0]
        tmp_key = []
        for i, sample in enumerate(raw_batch):
            if 'key' in sample:
                tmp_key.append(sample['key'])
                sample.pop('key')
            sample['sentences']['bag_id'] = i * np.ones(sample['sentences']['size'],dtype=int)
            sample['sentences']['in_bag_index'] = np.arange(sample['sentences']['size'])
            sample['extend_sentences']['bag_id'] = i * np.ones(sample['extend_sentences']['size'],dtype=int)
            sample['extend_sentences']['in_bag_index'] = np.arange(sample['extend_sentences']['size'])
            sample['sentences']['scope'] = np.array([[scope[0], scope[0] + sample['sentences']['size']]])
            sample['extend_sentences']['scope'] = np.array([[scope[1], scope[1] + sample['extend_sentences']['size']]])
            scope[0] += sample['sentences']['size']
            scope[1] += sample['extend_sentences']['size']
            if self.mode != 'train':
                sample['all_relations'] = np.expand_dims(sample['all_relations'], 0)
                sample["relation"] = 0
            if self.method == 0:
                sample['sentences']['relation'] = sample["relation"] * np.ones(sample['sentences']['size'],dtype=int)
                sample.pop('extend_sentences')
            elif self.method == 1:
                for x in ['word', 'pos1', 'pos2', 'mask', 'length', 'bag_id']:
                    sample['sentences'][x] = np.concatenate((sample['sentences'][x], sample['extend_sentences'][x]))
                sample['sentences']['in_bag_index'] = np.concatenate((sample['sentences']['in_bag_index'], sample['sentences']['size'] + sample['extend_sentences']['in_bag_index']))
                sample['sentences']['size'] += sample['extend_sentences']['size']
                sample['sentences']['scope'][0, 1] += sample['extend_sentences']['size']
                scope[0] += sample['extend_sentences']['size']
                sample['sentences']['relation'] = sample["relation"] * np.ones(sample['sentences']['size'],dtype=int)
                sample.pop('extend_sentences')
            elif self.method == 2:
                sample['sentences']['relation'] = sample["relation"] * np.ones(sample['sentences']['size'], dtype=int)
                sample['extend_sentences']['relation'] = sample["relation"] * np.ones(sample['extend_sentences']['size'],dtype=int)
            else:
                raise Exception('undefined method')
        batch = _cat_collate_fn(raw_batch)
        batch['num_bag'] = len(raw_batch)
        if self.method == 0 or self.method == 1:
            batch['max_bag'] = max(batch['sentences']['size']).item()
        elif self.method == 2:
            batch['max_bag'] = (max(batch['sentences']['size']).item(), max(batch['extend_sentences']['size']).item())
        if self.mode == 'train':
            target = batch['relation']
        else:
            target = batch['all_relations']
        batch['key'] = tmp_key
        return batch, target

class cust_batch_sampler(Sampler):
    def __init__(self, datasource, batch_size, method, batch_type,shuffle=True):
        self.type = batch_type
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.method = method
        self.index_set = []
        self.total_size = sum([size for i,size in self.index_set])
        for i in range(len(datasource)):
            if method == 0:
                self.index_set.append((i,datasource[i]['sentences']['size']))
            else:
                self.index_set.append((i, datasource[i]['sentences']['size'] + datasource[i]['extend_sentences']['size']))
        self.total_size = sum([size for i,size in self.index_set])

    def __iter__(self):
        batch = []
        if self.type == 0:
            if self.shuffle:
                random.shuffle(self.index_set)
            for i, _ in self.index_set:
                batch.append(i)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0:
                yield batch
        else:
            if self.shuffle:
                random.shuffle(self.index_set)
            curr_batch_size = 0
            for i, size in self.index_set:
                if curr_batch_size + size < self.batch_size:
                    batch.append(i)
                    curr_batch_size += size
                else:
                    yield batch
                    batch = [i]
                    curr_batch_size = size
            if len(batch) > 0:
                yield batch
    def __len__(self):
        if self.type == 0:
            return (len(self.index_set) + self.batch_size - 1) // self.batch_size
        else:
            return self.total_size // self.batch_size
class BaseNytLoader(DataLoader):
    def _filter_subset(self, mode):
        """
        0: keep all
        1: only table
        2: only sentence = 1
        3: only sentence > 1
        """
        if mode == 0:
            return self.dataset
        elif mode == 1:
            indices = [i for i in range(len(self.dataset)) if self.dataset[i]['extend_sentences']['size'] != 0]
        elif mode == 2:
            indices = [i for i in range(len(self.dataset)) if self.dataset[i]['sentence_set_origin_size'] == 1]
        elif mode == 3:
            indices = [i for i in range(len(self.dataset)) if self.dataset[i]['sentence_set_origin_size'] > 1]
        else:
            raise Exception('unkown filter mode')
        return Subset(self.dataset, indices)
        
    def _decode(self, sentence):
        if isinstance(sentence[0], int):
            return ' '.join([self.id2word[x] for x in sentence if x != 0])
        else:
            return [' '.join([self.id2word[x] for x in y if x != 0]) for y in sentence]

    def __init__(self, data_dir, word2id, rel2id, max_length, batch_size, validation_split=0 , mode = 'train', src='train', select=0, method=0, filtering_mode=0, anonymization=False,shuffle=True, num_workers=1,batch_type=0):
        """
        method:
            0: only sentence
            1: merge sentence and extend sentence
            2: seperate sentence and extend sentence
        """
        self.max_length = max_length
        self.dataset = BaseNytDataset(data_dir, word2id, rel2id, max_length, mode, src, select, anonymization=anonymization)
        if mode == 'train':
            self.rel2count = self.dataset.rel2count
        self.id2word = {value:key for key,value in word2id.items()}
        self.subset = self._filter_subset(filtering_mode)
        self.n_samples = len(self.subset) if batch_type==0 else 'na'
        self.method = method
        if batch_type == 0:
            super(BaseNytLoader, self).__init__(batch_size=batch_size,shuffle=shuffle,dataset=self.subset, num_workers=num_workers,collate_fn=cust_collate_fn(mode=mode,method=method))
        else:
            self.batch_sampler = cust_batch_sampler(self.subset,batch_size,method,batch_type,shuffle)
            super(BaseNytLoader, self).__init__(batch_sampler=self.batch_sampler, dataset=self.subset, num_workers=num_workers, collate_fn=cust_collate_fn(mode=mode, method=method))
            
