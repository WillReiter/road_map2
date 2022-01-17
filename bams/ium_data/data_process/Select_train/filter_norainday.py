# -*- coding: utf-8 -*-  
#将dbz均值小于50的时间序列从bj_train_set.txt中除去
#dbz均值小于50的时间序列第一帧保存在Select_norainday.txt

select_no_rainday = 'no_rainday.txt'
save_txt = 'result.txt'
total_txt = 'bj_train_set.txt'
filter_day = []

with open(select_no_rainday, 'r') as select:
    for line in select:
        filter_day.append(line[0:14])

l = len(filter_day)

with open(save_txt, 'w') as save:    
    with open(total_txt, 'r') as total:
        line_index = 0
        for line in total:
            day = line[0:14]
            if line_index!=l and day == filter_day[line_index]:
                line_index += 1
            else:
                save.write(str(line))
            
