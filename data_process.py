class CLAParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.metadata = {}
        self.classes = []

    def parse(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()

        # 解析元数据
        metadata_line = lines[0].strip().split()
        print(metadata_line)
        self.metadata['dataset_name'] = metadata_line[0]
        self.metadata['version'] = int(metadata_line[1])
        # self.metadata['num_classes'] = int(metadata_line[2])
        # self.metadata['num_samples'] = int(metadata_line[3])

        # 解析类别信息
        tokens = []
        for line in lines[3:]:

            # 如果是空白行
            if not line.strip():

                class_info = {
                    'name': tokens[0],
                    'id': int(tokens[1]),
                    'count': int(tokens[2]),
                    'samples': list(map(int, tokens[3:]))
                }
                print(class_info)
                self.classes.append(class_info)
                tokens = []
                continue

            temp = line.strip().split()
            tokens += temp

            # class_info = {
            #     'name': tokens[0],
            #     'id': int(tokens[1]),
            #     'count': int(tokens[2]),
            #     'samples': list(map(int, tokens[3:]))
            # }


# 示例用法
file_path = '..\datasets\shrec_13\SHREC2013_Sketch_Evaluation\SHREC13_SBR_Model.cla'
cla_parser = CLAParser(file_path)
cla_parser.parse()

# 获取解析结果
metadata = cla_parser.metadata
classes = cla_parser.classes

import os
from shutil import copyfile
from multy_view_2 import processOff
def create_folders_and_copy_files(obj_data, data_dir, output_dir):
    # 创建对象文件夹
    object_folder = os.path.join(output_dir, obj_data['name'])
    os.makedirs(object_folder, exist_ok=True)

    # 对每个sample执行操作
    for idx, sample in enumerate(obj_data['samples']):
        # 创建子文件夹
        sample_folder = os.path.join(object_folder, f'item_{idx}')
        os.makedirs(sample_folder, exist_ok=True)

        # 读取并写入 OFF 文件
        off_filename = f"m{sample}.off"
        print(off_filename)
        off_filepath = os.path.join(data_dir, off_filename)
        if os.path.exists(off_filepath):
            print(off_filepath)
            destination_path = os.path.join(sample_folder, off_filename)
            processOff(off_filepath, sample_folder)
            # copyfile(off_filepath, destination_path)
            # print(f"Successfully copied {off_filename} to {sample_folder}")
        else:
            print(f"Error: {off_filename} not found at {data_dir}")

object_data = {'name': 'wrist-watch', 'ID': 0, 'Count': 5, 'Samples': [598, 599, 600, 601, 602]}

# 数据目录和输出目录
data_directory = "..\datasets\shrec_13\SHREC13_SBR_TARGET_MODELS\models"
output_directory = "..\datasets\shrec_13\multy_view"

# 打印解析结果
print("Metadata:", metadata)
print("\nClasses:")
for class_info in classes:
    # label = 1
    # if (label == 1):
    #     if(class_info['name'] != 'sword' ):
    #         print(class_info['name'])
    #         continue
    #     label = 2
    print(f"Name: {class_info['name']}, ID: {class_info['id']}, Count: {class_info['count']}, Samples: {class_info['samples']}")
    create_folders_and_copy_files(class_info, data_directory, output_directory)