from utils.metrics import convert_to_dict_pre
import torch
import numpy as np

class CompanyPredict():
    def __init__(self, model, config, tokenizer, word_pad_idx = 0) -> None:
        self.model = model
        self.tokenizer = tokenizer
        # BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)
        self.label2id = config.label2id
        self.id2label = {_id: _label for _label, _id in list(config.label2id.items())}
        self.word_pad_idx = word_pad_idx
        self.device = config.device

    def data_process(self, company_names):
        sentences_ori = []
        sentences = []
        token_len = []
        for company in company_names:
            words = ['[CLS]'] + self.tokenizer.tokenize(company)
            sentences_ori.append(words)
            word_lens = []
            for w in words[1:]: #防止有英文
                word_lens.append(1) #len(w))
            token_start_idxs = 1 + np.cumsum([0] + word_lens[:-1])
            sentences.append((self.tokenizer.convert_tokens_to_ids(words), token_start_idxs))
        # print(list(zip(sentences_ori[1][1:], sentences[1][-1])))

        # print(list(zip(words[1:], sentences[-1][-1])))
        # batch length
        batch_len = len(sentences)

        # compute length of longest sentence in batch
        max_len = max([len(s[0]) for s in sentences])

        # padding data 初始化
        batch_data = self.word_pad_idx * np.ones((batch_len, max_len))
        max_label_len = 0
        batch_label_starts = []
        sent_data = []
        # padding and aligning
        for j in range(batch_len):
            cur_len = len(sentences[j][0])
            batch_data[j][:cur_len] = sentences[j][0] #sentence
            # 找到有标签的数据的index（[CLS]不算）
            label_start_idx = sentences[j][-1] # word start index
            label_starts = np.zeros(max_len)
            label_starts[[idx for idx in label_start_idx if idx < max_len]] = 1
            batch_label_starts.append(label_starts)
            max_label_len = max(int(sum(label_starts)), max_label_len)

        # convert data to torch LongTensors
        batch_data = torch.tensor(np.array(batch_data), dtype=torch.long)
        batch_label_starts = torch.tensor(np.array(batch_label_starts), dtype=torch.long)
        #         print(batch_data.shape, batch_labels.shape)
        #         print(batch_data, batch_labels)
        # shift tensors to GPU if available
        batch_data, batch_label_starts = batch_data.to(self.device), batch_label_starts.to(self.device)
        
        return batch_data, batch_label_starts, sentences_ori
    
    def predict(self, batch_data, batch_label_starts, tokens, sentences):
        batch_masks = batch_data.gt(0)

        batch_output = self.model((batch_data, batch_label_starts),
                token_type_ids=None, attention_mask=batch_masks)

        embedding = batch_output[1]
        batch_output = batch_output[0]

        batch_output = self.model.crf.viterbi_decode(batch_output, mask=batch_masks[:, 1:])

        pred_tags = [[self.id2label.get(idx) for idx in indices] for indices in batch_output]
        
        all_end = []
        for pred_label, token, senten in zip(pred_tags, tokens, sentences):
            sentence = senten.replace(' ', '')
            res = convert_to_dict_pre(token[1:], pred_label)
            for label in ['place', 'brand', 'trade', 'suffix']:
                if label not in res.keys():
                    res[label] = []
                else:
                    for i in range(len(res[label])):
                        res[label][i][0] = sentence[res[label][i][1]:res[label][i][2]]
                
            all_end.append(res)
        # all_end = predict_output_pre(pred_tags, sentences_ori)
        return all_end, pred_tags, embedding