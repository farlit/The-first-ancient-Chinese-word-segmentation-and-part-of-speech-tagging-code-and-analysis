def load_data(data_dir):
    with open(data_dir, 'r', encoding='utf-8-sig') as f:
        word_list = []
        seg_list = []
        pos_list = []
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
            pretail = 0
            temptail = 0
            line=line.strip(' ')
            line=line.strip('\n')
            for char in line:
                word.append(char)
                seg.append(0)
                pos.append(0)
            length = len(word)
            if length > max_len:
                prestart = 0
                for i in range(0,length):
                    if (i==length-1) or ((word[i] in tail_pun) and (word[i+1] not in tail_pun)):
                        pretail = temptail
                        temptail = i
                    if pretail < prestart + max_len and temptail >= prestart + max_len:
                        word_list.append(['[CLS]']+word[prestart : pretail+1])
                        seg_list.append(seg[prestart: pretail + 1])
                        pos_list.append(pos[prestart: pretail + 1])
                        prestart = pretail+1
                        flag.append(num_conjun)
                if prestart <= length-1:
                    word_list.append(['[CLS]'] + word[prestart: ] )
                    seg_list.append(seg[prestart:])
                    pos_list.append(pos[prestart:])
                    flag.append(num_conjun)
                num_conjun = num_conjun + 1
            else:
                word_list.append(['[CLS]'] + word )
                seg_list.append(seg)
                pos_list.append(pos)
                flag.append(0)
    return word_list,seg_list,pos_list,flag
