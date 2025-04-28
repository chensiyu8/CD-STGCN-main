import logging
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_transformers import RobertaTokenizer

__all__ = ['MMDataLoader']

logger = logging.getLogger('MMSA')

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATASET_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
            'simsv2': self.__init_simsv2,
        }
        DATASET_MAP[args['dataset_name']]()

    def __init_mosi(self):
        if self.args['custom_feature']:
            # use custom feature file extracted with MMSA-FET
            with open(self.args['custom_feature'], 'rb') as f:
                data = pickle.load(f)
        else:
            # use deault feature file specified in config file
            with open(self.args['featurePath'], 'rb') as f:
                data = pickle.load(f)
        
        if self.args.get('use_bert', None):
            self.text = data[self.mode]['text_bert'].astype(np.float32)
            self.args['feature_dims'][0] = 768
        else:
            self.text = data[self.mode]['text'].astype(np.float32)
            self.args['feature_dims'][0] = self.text.shape[2]
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.args['feature_dims'][1] = self.audio.shape[2]
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.args['feature_dims'][2] = self.vision.shape[2]
        self.raw_text = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']
        #
        if self.args.get('sentilare', None):
            #处理pos_id,senti_id,polarity_id
            input_ids = []
            pos_tag_ids = data[self.mode]['pos_id']
            senti_word_ids = data[self.mode]['senti_id']
            polarity_ids =[]
            # for pos, input, text  in zip(pos_tag_ids, input_ids, self.raw_text):
            #     if len(pos) > 50 or len(input) > 50:
            #         print(pos, text)
            tokenizer = RobertaTokenizer.from_pretrained(self.args.pretrained, do_lower_case=True)
            pos_ids_list = []
            senti_ids_list = []
            polarity_ids_list = []
            text_bert_list = []
            for i, single_text in enumerate(data[self.mode]['raw_text']):
                input_ids, pos_ids_t, senti_word_ids_t, polarity_ids_t, input_mask, segment_ids = convert_to_features(self.args, single_text, pos_tag_ids[i], senti_word_ids[i], 50, tokenizer)
                pos_ids_list.append(pos_ids_t)
                senti_ids_list.append(senti_word_ids_t)
                polarity_ids_list.append(polarity_ids_t)
                text_bert = [input_ids, input_mask, segment_ids]
                text_bert_list.append(text_bert)
            self.pos_ids =  np.array(pos_ids_list)
            self.senti_ids =  np.array(senti_ids_list)
            self.polarity_ids = np.array(polarity_ids_list)
            self.text = np.array(text_bert_list).astype(np.float32)
           # print(np.array(self.pos_ids).shape, np.array(self.senti_ids).shape, np.array(self.polarity_ids).shape)

        # Overide with custom modality features
        if self.args['feature_T']:
            with open(self.args['feature_T'], 'rb') as f:
                data_T = pickle.load(f)
            if self.args.get('use_bert', None):
                self.text = data_T[self.mode]['text_bert'].astype(np.float32)
                self.args['feature_dims'][0] = 768
            else:
                self.text = data_T[self.mode]['text'].astype(np.float32)
                self.args['feature_dims'][0] = self.text.shape[2]
        if self.args['feature_A']:
            with open(self.args['feature_A'], 'rb') as f:
                data_A = pickle.load(f)
            self.audio = data_A[self.mode]['audio'].astype(np.float32)
            self.args['feature_dims'][1] = self.audio.shape[2]
        if self.args['feature_V']:
            with open(self.args['feature_V'], 'rb') as f:
                data_V = pickle.load(f)
            self.vision = data_V[self.mode]['vision'].astype(np.float32)
            self.args['feature_dims'][2] = self.vision.shape[2]

        self.labels = {
            # 'M': data[self.mode][self.args['train_mode']+'_labels'].astype(np.float32)
            'M': np.array(data[self.mode]['regression_labels']).astype(np.float32)
        }
        if self.args['dataset_name'] == 'sims' or self.args['dataset_name'] == 'simsv2':
            for m in "TAV":
                self.labels[m] = data[self.mode]['regression' + '_labels_' + m].astype(np.float32)

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        if not self.args['need_data_aligned']:
            if self.args['feature_A']:
                self.audio_lengths = list(data_A[self.mode]['audio_lengths'])
            else:
                self.audio_lengths = data[self.mode]['audio_lengths']
            if self.args['feature_V']:
                self.vision_lengths = list(data_V[self.mode]['vision_lengths'])
            else:
                self.vision_lengths = data[self.mode]['vision_lengths']
        self.audio[self.audio == -np.inf] = 0

        if self.args.get('data_missing'):
            # Currently only support unaligned data missing.
            self.text_m, self.text_length, self.text_mask, self.text_missing_mask = self.generate_m(self.text[:,0,:], self.text[:,1,:], None,
                                                                                        self.args.missing_rate[0], self.args.missing_seed[0], mode='text')
            Input_ids_m = np.expand_dims(self.text_m, 1)
            Input_mask = np.expand_dims(self.text_mask, 1)
            Segment_ids = np.expand_dims(self.text[:,2,:], 1)
            self.text_m = np.concatenate((Input_ids_m, Input_mask, Segment_ids), axis=1)

            if self.args['need_data_aligned']:
                self.audio_lengths = np.sum(self.text[:,1,:], axis=1, dtype=np.int32)
                self.vision_lengths = np.sum(self.text[:,1,:], axis=1, dtype=np.int32)

            self.audio_m, self.audio_length, self.audio_mask, self.audio_missing_mask = self.generate_m(self.audio, None, self.audio_lengths,
                                                                                        self.args.missing_rate[1], self.args.missing_seed[1], mode='audio')
            self.vision_m, self.vision_length, self.vision_mask, self.vision_missing_mask = self.generate_m(self.vision, None, self.vision_lengths,
                                                                                        self.args.missing_rate[2], self.args.missing_seed[2], mode='vision')

        if self.args.get('need_normalized'):
            self.__normalize()
    
    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def __init_simsv2(self):
        return self.__init_mosi()

    def generate_m(self, modality, input_mask, input_len, missing_rate, missing_seed, mode='text'):
        
        if mode == 'text':
            input_len = np.argmin(input_mask, axis=1)
        elif mode == 'audio' or mode == 'vision':
            input_mask = np.array([np.array([1] * length + [0] * (modality.shape[1] - length)) for length in input_len])
        np.random.seed(missing_seed)
        missing_mask = (np.random.uniform(size=input_mask.shape) > missing_rate) * input_mask
        
        assert missing_mask.shape == input_mask.shape
        
        if mode == 'text':
            # CLS SEG Token unchanged.
            for i, instance in enumerate(missing_mask):
                instance[0] = instance[input_len[i] - 1] = 1
            
            modality_m = missing_mask * modality + (100 * np.ones_like(modality)) * (input_mask - missing_mask) # UNK token: 100.
        elif mode == 'audio' or mode == 'vision':
            modality_m = missing_mask.reshape(modality.shape[0], modality.shape[1], 1) * modality
        
        return modality_m, input_len, input_mask, missing_mask

    def __truncate(self):
        # NOTE: truncate input to specific length.
        def do_truncate(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if((instance[index] == padding).all()):
                        if(index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+20])
                            break
                    else:                        
                        truncated_feature.append(instance[index:index+20])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature
        
        text_length, audio_length, video_length = self.args['seq_lens']
        self.vision = do_truncate(self.vision, video_length)
        self.text = do_truncate(self.text, text_length)
        self.audio = do_truncate(self.audio, audio_length)

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # For visual and audio modality, we average across time:
        # The original data has shape (max_len, num_examples, feature_dim)
        # After averaging they become (1, num_examples, feature_dim)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if 'use_bert' in self.args and self.args['use_bert']:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
            'raw_text': self.raw_text[index],
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            # 'pos_id': torch.Tensor(self.pos_ids[index]).long(),
            # 'senti_id': torch.Tensor(self.senti_ids[index]).long(),
            # 'polarity_id': torch.Tensor(self.polarity_ids[index]).long(),
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        } 
        if not self.args['need_data_aligned']:
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]
        if self.args.get('data_missing'):
            sample['text_m'] = torch.Tensor(self.text_m[index])
            sample['text_missing_mask'] = torch.Tensor(self.text_missing_mask[index])
            sample['audio_m'] = torch.Tensor(self.audio_m[index])
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['audio_mask'] = self.audio_mask[index]
            sample['audio_missing_mask'] = torch.Tensor(self.audio_missing_mask[index])
            sample['vision_m'] = torch.Tensor(self.vision_m[index])
            sample['vision_lengths'] = self.vision_lengths[index]
            sample['vision_mask'] = self.vision_mask[index]
            sample['vision_missing_mask'] = torch.Tensor(self.vision_missing_mask[index])
        if self.args.get('sentilare', None):
            sample['pos_id'] = torch.Tensor(self.pos_ids[index]).long()
            sample['senti_id'] = torch.Tensor(self.senti_ids[index]).long()
            sample['polarity_id'] = torch.Tensor(self.polarity_ids[index]).long()

        return sample

def MMDataLoader(args, num_workers):

    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args['seq_lens'] = datasets['train'].get_seq_len() 

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args['batch_size'],
                       num_workers=num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }
    
    return dataLoader


def prepare_sentilare_input(args, tokens, pos_ids, senti_ids, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP] + [SEP]
    pos_ids = [4] + pos_ids + [4] + [4]
    senti_ids = [2] + senti_ids + [2] + [2]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = args.seq_lens[0] - len(input_ids)
    padding = [0] * pad_length

    # Pad inputs
    input_ids += padding
    pos_ids += [4] * pad_length
    senti_ids += [2] * pad_length
    polarity_ids = [5] * len(input_ids)
    input_mask += padding
    segment_ids += padding

    return input_ids, pos_ids, senti_ids, polarity_ids, input_mask, segment_ids

#针对一句话
def convert_to_features(args, text, pos_ids, senti_ids, max_seq_length, tokenizer):
    #sentence - level
    input_ids = []
    pos_tag_ids = pos_ids
    #print(pos_tag_ids)
    senti_word_ids = senti_ids
    polarity_ids = []
    input_mask = []
    segment_ids =[]
    #print(text)
    text = text.split()
    tokens, inversions, = [], []
    for idx, word in enumerate(text):
        #print(word)
        tokenized = tokenizer.tokenize(word)
        tokens.extend(tokenized)
        inversions.extend([idx] * len(tokenized))

    # Check inversion
    assert len(tokens) == len(inversions)

    aligned_pos_ids = []
    aligned_senti_ids = []
    #print(tokens, inversions)
    for inv_idx in inversions:
       # print(pos_tag_ids)
        aligned_pos_ids.append(pos_tag_ids[inv_idx])
        aligned_senti_ids.append(senti_word_ids[inv_idx])

    pos_tag_ids = aligned_pos_ids
    senti_word_ids = aligned_senti_ids

    # Truncate input if necessary

    if len(tokens) > max_seq_length - 3:
        tokens = tokens[: max_seq_length - 3]
        pos_tag_ids = pos_tag_ids[: max_seq_length - 3]
        senti_word_ids = senti_word_ids[: max_seq_length - 3]

    input_ids,  pos_tag_ids, senti_word_ids, polarity_ids, input_mask, segment_ids = prepare_sentilare_input(
        args, tokens, pos_tag_ids, senti_word_ids, tokenizer
    )
    # Check input length
    assert len(input_ids) == args.seq_lens[0]
    assert len(input_mask) == args.seq_lens[0]
    assert len(segment_ids) == args.seq_lens[0]
    return input_ids, pos_tag_ids, senti_word_ids, polarity_ids, input_mask, segment_ids
