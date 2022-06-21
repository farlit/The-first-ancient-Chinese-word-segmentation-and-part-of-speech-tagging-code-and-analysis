# coding=utf-8
# please use python 3.0+

"""
Before using it, you need to make sure that you place
    this py file,
    the text to evaluate,
    and the gold answer
in the same folder.
"""

"""
usage:

1. Open cmd to enter the folder you saved the above three files.

2. enter:  python eval_EvaHan_2022.py eval_text_name gold_text_name
   such as: python eval_EvaHan_2022.py testa_unicatt_1_close.txt testa_gold.txt 

3. the outcome will be shown in cmd and saved in a txt file in the current folder. 

"""

import sys


def count_prf(contestant, answer):
    outcome = dict()
    with open(contestant, encoding='utf-8-sig') as con:
        read_con = [i for i in con.read().split('\n') if i]

    with open(answer, encoding='utf-8-sig') as ans:
        read_ans = [i for i in ans.read().split('\n') if i]
    con_queue = []
    ans_queue = []
    assert len(read_con) == len(read_ans)
    para_group = zip(read_con, read_ans)
    for paras in para_group:
        con_para, ans_para = paras[0], paras[1]
        con_chips = [i for i in con_para.split(' ') if i]
        for chip_c in con_chips:
            sp_c = chip_c.split('/')
            word_c = sp_c[0]
            pos_c = sp_c[-1]
            length_c = len(word_c)
            if length_c == 1:
                con_queue.append([word_c, 'S-' + pos_c])
            elif length_c == 2:
                con_queue.append([word_c[0], 'B-' + pos_c])
                con_queue.append([word_c[-1], 'E-' + pos_c])
            else:
                if word_c:
                    con_queue.append([word_c[0], 'B-' + pos_c])
                    for middle_c in word_c[1:-1]:
                        con_queue.append([middle_c, 'I-' + pos_c])
                    con_queue.append([word_c[-1], 'E-' + pos_c])
                else:
                    print(con_para)
        ans_chips = [i for i in ans_para.split(' ') if i]
        for chip_a in ans_chips:
            sp_a = chip_a.split('/')
            word_a = sp_a[0]
            pos_a = sp_a[-1]
            length_a = len(word_a)
            if length_a == 1:
                ans_queue.append([word_a, 'S-' + pos_a])
            elif length_a == 2:
                ans_queue.append([word_a[0], 'B-' + pos_a])
                ans_queue.append([word_a[-1], 'E-' + pos_a])
            else:
                if word_a:
                    ans_queue.append([word_a[0], 'B-' + pos_a])
                    for middle_a in word_a[1:-1]:
                        ans_queue.append([middle_a, 'I-' + pos_a])
                    ans_queue.append([word_a[-1], 'E-' + pos_a])
                else:
                    print(ans_para)
    '''
    for i, j in enumerate(con_queue):
        if j[0] != ans_queue[i][0]:
            break
        print(j[1])
    '''
    assert len(con_queue) == len(ans_queue)

    queue_group = zip(con_queue, ans_queue)
    # 分词
    origin, label = [], []
    for queues in queue_group:
        c_q = queues[0]
        a_q = queues[-1]
        assert c_q[0] == a_q[0]
        origin.append(a_q[-1].split('-')[0])
        label.append(c_q[-1].split('-')[0])
    correct_num = 0
    index = 0
    length = len(origin)
    while index < length:
        if origin[index] == label[index] == 'S':
            correct_num += 1
            index += 1
        elif origin[index] == label[index] == 'B':
            index += 1
            while origin[index] == label[index]:
                if origin[index] == 'E':
                    correct_num += 1
                    index += 1
                    break
                else:
                    index += 1
            else:
                index += 1
        else:
            index += 1
    deno1 = label.count("E") + label.count('S')
    deno2 = origin.count('E') + origin.count('S')
    p = correct_num / deno1
    r = correct_num / deno2
    f = (2 * p * r) / (p + r)
    outcome['WordSeg'] = [p, r, f]

    # 词性标注
    queue_group = zip(con_queue, ans_queue)
    origin, label = [], []
    for queues in queue_group:
        c_q = queues[0]
        a_q = queues[-1]
        assert c_q[0] == a_q[0]
        origin.append(a_q[-1].split('-'))
        label.append(c_q[-1].split('-'))
    correct_num = 0
    index = 0
    length = len(origin)
    while index < length:
        if origin[index][0] == label[index][0] == 'S':
            if origin[index][1] == label[index][1]:
                correct_num += 1
            index += 1
        elif origin[index][0] == label[index][0] == 'B':
            if origin[index][1] == label[index][1]:
                index += 1
                while (origin[index][0] == label[index][0]) and (origin[index][1] == label[index][1]):
                    if origin[index][0] == 'E':
                        correct_num += 1
                        index += 1
                        break
                    else:
                        index += 1
                else:
                    index += 1
            else:
                index += 1
        else:
            index += 1

    p = correct_num / deno1
    r = correct_num / deno2
    f = (2 * p * r) / (p + r)
    outcome['PoS'] = [p, r, f]

    return outcome


value = count_prf(sys.argv[1], sys.argv[2])
result_line = "The result of {} is：".format(sys.argv[1])
tool_line = "+" + "-" * 17 + '+' + "-" * 9 + '+' + '-' * 9 + '+' + '-' * 9 + '+'
task_line = '|' + '       Task      ' + '|' + ' ' * 3 + ' P ' + ' ' * 3 + '|' + ' ' * 3 + ' R ' + ' ' * 3 + '|' + ' ' * 3 + ' F1 ' + ' ' * 2 + '|'
wsg_line = ('|' + 'Word segmentation' + '|' + ' ' + '{:.4f}' + ' ' + '|' + ' ' + '{:.4f}' + ' ' + '|' + ' ' + '{:.4f}' + ' ' + '|').format((value['WordSeg'][0])*100, (value['WordSeg'][1])*100, (value['WordSeg'][2])*100)
pos_line = ('|' + '   Pos tagging   ' + '|' + ' ' + '{:.4f}' + ' ' + '|' + ' ' + '{:.4f}' + ' ' + '|' + ' ' + '{:.4f}' + ' ' + '|').format((value['PoS'][0])*100, (value['PoS'][1])*100, (value['PoS'][2])*100)

print(result_line)
print(tool_line)
print(task_line)
print(tool_line)
print(wsg_line)
print(tool_line)
print(pos_line)
print(tool_line)

ouput = 'eval_outcome_' + sys.argv[1]
with open(ouput, 'w', encoding='utf-8') as fo:
    fo.write(result_line + '\n' + task_line + '\n')
    fo.write(wsg_line + '\n' + pos_line +'\n')

print('\n' + ouput + ' is created' + '\n')
print('Finished'+ '\n')
