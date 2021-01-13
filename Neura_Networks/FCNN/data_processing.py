#数据预处理
import os
import shutil
import numpy as np
import pandas as pd
class pre_data:
    def __init__(self, path, name, time, small, medium):
        self.name = name
        self.path = path
        self.final_path = 'D:/deeplearning/Pre_data'
        self.file_name_list = os.listdir(self.path)
        self.file_path = ''
        self.file_count = 0
        self.time = "00:00:" + time
        self.small = small
        self.medium = medium
        self.small_name = self.name + '1'
        self.medium_name = self.name + '2'
        self.large_name = self.name + '3'
        self.all_txt_value = []
        self.final_path_children = 'D:/deeplearning/Pre_data' + '/' + self.name
        if os.path.exists(self.final_path_children):
            shutil.rmtree(self.final_path_children)
    def determine_file_order(self):
        (self.file_name_list).sort(key=(lambda x: int(x[5:-2])))
        self.file_count = len(self.file_name_list)
    def merge_file(self):
        for i in range(0, self.file_count, 2):
            file_1 = self.file_name_list[i]
            file_2 = self.file_name_list[i + 1]
            file_1_2 = str(int(i/2 + 1)) + self.name + ".txt"
            os.chdir(self.path)
            with open(file_1,'r') as fa:
                with open(file_2,'r') as fb:
                    with open(file_1_2,'w') as fc:
                        for line in fa:
                            fc.write(line.strip('\r\n'))
                            fc.write(fb.readline())
    def move_file(self):
        os.chdir(self.final_path)
        os.mkdir(self.name)
        self.final_path = 'D:/deeplearning/Pre_data' + '/' + self.name
        self.file_name_list = os.listdir(self.path)
        for filename in self.file_name_list:
            if '.txt' in filename:
                self.file_path = self.path + '/' + filename
                shutil.move(self.file_path, self.final_path)
    def data_collation(self):
        self.file_name_list = os.listdir(self.final_path)
        (self.file_name_list).sort(key=(lambda x: int(x[:-8])))
        for txt in self.file_name_list:
            print(txt)
            Txt = pd.read_table(self.final_path + '/' + txt ,index_col=False, header=1, error_bad_lines=False)
            Txt = pd.DataFrame(Txt.iloc[::-1]) #倒序
            Txt = Txt.dropna(axis=1) #删除空白列
            col_name = ["时间1", "稳压器压力", "稳压器水位", "上充流量", "SG1给水流量", "SG1出口压力", "SG1出口蒸汽流量", "时间2","主蒸汽母管压力", "安全壳压力", "安全壳温度", "安全壳放射性", "地坑水位", "冷却剂平均温度"]
            Txt.columns = col_name #重命名行
            Txt = Txt.drop(columns="时间2")#删除重复列
            Txt.index = range(Txt.shape[0])
            same_second_value = []
            for j in range(1, Txt.shape[1]):
                same_second_parameters_value = []
                for i in range(Txt.shape[0]):
                    print(self.time, Txt.iloc[i,0])
                    if self.time in Txt.iloc[i,0]:
                        same_second_parameters_value.append(Txt.iloc[i,j])
                print(same_second_parameters_value, len(same_second_parameters_value))
                average_value = np.mean(same_second_parameters_value)
                print(average_value)
                same_second_value.append(average_value)
            self.all_txt_value.append(same_second_value)
    def divide_scope(self):
        print(self.all_txt_value)
        if self.name != 'NORM':
            for i in range(self.small):
                self.all_txt_value[i].append(self.small_name)
            for i in range(self.small, self.medium):
                self.all_txt_value[i].append(self.medium_name)
            for i in range(self.medium, 100):
                self.all_txt_value[i].append(self.large_name)
        if self.name == 'NORM':
            for i in range(5):
                self.all_txt_value[i].append('norm')
        return self.all_txt_value
Type = {'LOCA':(12, 43), 'MSLB':(12, 38), 'SGTR':(15, 37), 'NORM':(0, 0)}
# Type = {'LOCA':(30, 60), 'MSLB':(30, 60), 'SGTR':(30, 60), 'NORM':(0, 0)}
basic_path = 'D:/deeplearning/RawData/'

# time = '03'
for i in range(60):
    data = []
    print(i)
    if i < 10:
        time = '0' + str(i)
    else:
        time = str(i)
    for (accident, scope) in Type.items():
        Pre_data = pre_data(basic_path + accident, accident, time, scope[0], scope[1])
        Pre_data.determine_file_order()
        Pre_data.merge_file()
        Pre_data.move_file()
        Pre_data.data_collation()
        result = Pre_data.divide_scope()
        for j in range(len(result)):
            data.append(result[j])
    data = pd.DataFrame(data)
    data.to_csv('D:/deeplearning/dataset1/'+ time + '.csv')