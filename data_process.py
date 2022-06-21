import math
import config
from helper import get_gram2id
def getBMES(input_str):
    """
    将每个输入词转换为BMES标注
    """
    output_str = []
    if len(input_str) == 1:
        output_str.append('S')
    elif len(input_str) == 2:
        output_str = ['B', 'I']
    else:
        M_num = len(input_str) - 2
        M_list = ['M'] * M_num
        output_str.append('B')
        output_str.extend(M_list)
        output_str.append('I')
    return output_str

def load_data(data_dir):
    """
    返回句子、BMES标签、词性标注标签
    """    
    with open(data_dir, 'r', encoding='utf-8-sig') as f:  
        word_list = []
        seg_list  = []
        pos_list  = []
        segpos_list = []
        flag =[]    #用于标志前后句之间是否有关联，0：无关联；1：前后句之间有关联
        num_conjun = 1  #用于标志句子间的联系
        tail_pun=['。','”','！','？','；']    #语句结尾可能的标点符号
        max_len = 511
        for line in f:       
            if len(line)==1:
                continue
            word = []
            seg = []
            pos = []
            segpos =[]
            pretail = 0
            temptail = 0
            line=line.strip(' ')
            line=line.strip('\n')
            text=line.split(' ')
            for word_pos in text:
                sp_c = word_pos.split('/')
                word_c = sp_c[0]
                pos_c = sp_c[-1]
                if pos_c == '。':
                    print(line)
                length_c = len(word_c)
                if length_c == 1:
                    word.append(word_c)
                    seg.append('S')
                    pos.append(pos_c)
                    segpos.append('S'+pos_c)
                elif length_c == 2:
                    word.append(word_c[0])
                    seg.append('B')
                    pos.append(pos_c)
                    segpos.append('B' + pos_c)

                    word.append(word_c[1])
                    seg.append('E')
                    pos.append(pos_c)
                    segpos.append('E' + pos_c)
                else:
                    if word_c:
                        word.append(word_c[0])
                        seg.append('B')
                        pos.append(pos_c)
                        segpos.append('B' + pos_c)
                        for middle_c in word_c[1:-1]:
                            word.append(middle_c)
                            seg.append('I')
                            pos.append(pos_c)
                            segpos.append('I' + pos_c)
                        word.append(word_c[-1])
                        seg.append('E')
                        pos.append(pos_c)
                        segpos.append('E' + pos_c)
                    else:
                        print(text)
                        print('Sentence data error!')
            length = len(word)
            assert length == len(seg)
            assert length == len(pos)
            assert length == len(segpos)
            if length > max_len:
                prestart = 0
                for i in range(0,length):
                    if (i==length-1) or ((word[i] in tail_pun) and (word[i+1] not in tail_pun)):
                        pretail = temptail
                        temptail = i
                    if pretail < prestart + max_len and temptail >= prestart + max_len:
                        word_list.append(['[CLS]']+word[prestart : pretail+1])
                        seg_list.append(seg[prestart : pretail+1])
                        pos_list.append(pos[prestart : pretail+1])
                        segpos_list.append(segpos[prestart: pretail+1])
                        prestart = pretail+1
                        flag.append(num_conjun)
                if prestart <= length-1:
                    word_list.append(['[CLS]'] + word[prestart: ] )
                    seg_list.append( seg[prestart: ] )
                    pos_list.append( pos[prestart: ] )
                    segpos_list.append( segpos[prestart: ] )
                    flag.append(num_conjun)
                num_conjun = num_conjun + 1
            else:
                word_list.append(['[CLS]'] + word )
                seg_list.append( seg )
                pos_list.append( pos)
                segpos_list.append( segpos)
                flag.append(0)

    gram2id = []
    gram2count = []
    # gram2id, gram2count = get_gram2id(word_list, config.ngram_type, config.ngram_length, config.ngram_threshold)
    gram_list = []
    positions = []
    gram_maxlen = []
    for sentence in word_list:
        sentence = sentence[1:]
        if config.use_attention :
            ngram_list = []
            matching_position = []
            ngram_list_len = []
            for i in range(config.cat_num):
                ngram_list.append([])
                matching_position.append([])
                ngram_list_len.append(0)
            for i in range(len(sentence)):
                for j in range(0, config.ngram_length):
                    if i + j + 1 > len(sentence):
                        break
                    ngram = ''.join(sentence[i: i + j + 1])
                    if ngram in gram2id:
                        index = int(min(config.cat_num, math.log2(gram2count[ngram]))) - 1
                        assert 0 <= index < config.cat_num
                        channel_index = index
                        try:
                            index = ngram_list[channel_index].index(ngram)
                        except ValueError:
                            ngram_list[channel_index].append(ngram)
                            index = len(ngram_list[channel_index]) - 1
                            ngram_list_len[channel_index] += 1
                        for k in range(j + 1):
                            matching_position[channel_index].append((i + k, index))
        else:
            ngram_list = None
            matching_position = None
            ngram_list_len = None
        max_ngram_len = max(ngram_list_len) if ngram_list_len is not None else None
        gram_list.append(ngram_list)
        positions.append(matching_position)
        gram_maxlen.append(max_ngram_len)

    return word_list,seg_list,pos_list,segpos_list,flag,gram_list,positions,gram_maxlen,gram2id

# sentences,seg,pos,segpos,flag=load_data('zuozhuan_train_utf8.txt')