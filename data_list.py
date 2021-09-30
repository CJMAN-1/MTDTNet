import os


def make_list():
    # # Cityscapes list
    # for mode in ['train', 'val']:
    #     print(mode)
    #     image_text_file = os.path.join('data_list', 'Cityscapes', mode + '_imgs.txt')
    #     label_text_file = os.path.join('data_list', 'Cityscapes', mode + '_labels.txt')
    #     if not os.path.exists(image_text_file) or not os.path.exists(label_text_file):
    #         image_root = os.path.join('data', 'Cityscapes', 'Images', mode)
    #         label_root = os.path.join('data', 'Cityscapes', 'GT', mode)
    #         file_list = os.listdir(image_root)
    #         f1 = open(image_text_file, mode='wt')
    #         f2 = open(label_text_file, mode='wt')
    #         for folder in file_list:
    #             image_folder = os.path.join(image_root, folder)
    #             label_folder = os.path.join(label_root, folder)
    #             image_list = os.listdir(image_folder)

    #             for image in image_list:
    #                 image_name_split = image.split('_')
    #                 label = image_name_split[0] + '_' + image_name_split[1] + '_' + image_name_split[
    #                     2] + '_' + 'gtFine_labelIds.png'
    #                 image_file = os.path.join(image_folder, image)
    #                 label_file = os.path.join(label_folder, label)
    #                 f1.write(image_file + '\n')
    #                 f2.write(label_file + '\n')
    #         f1.close()
    #         f2.close()

    # # GTA5 list
    # image_text_file = os.path.join('data_list', 'GTA5', 'train_imgs.txt')
    # label_text_file = os.path.join('data_list', 'GTA5', 'train_labels.txt')
    # if not os.path.exists(image_text_file) or not os.path.exists(label_text_file):
    #     image_root = os.path.join('data', 'GTA5', 'Images')
    #     label_root = os.path.join('data', 'GTA5', 'GT')
    #     image_list = os.listdir(image_root)
    #     label_list = os.listdir(label_root)

    #     f1 = open(image_text_file, mode='wt')
    #     f2 = open(label_text_file, mode='wt')

    #     for ifolder in image_list:
    #         folder_split = ifolder.split('_')
    #         lfolder = folder_split[0] + '_' + 'labels'
    #         image_folder = os.path.join(image_root, ifolder, 'images')
    #         label_folder = os.path.join(label_root, lfolder, 'labels')
    #         image_list = os.listdir(image_folder)
    #         for image in image_list:
    #             image_file = os.path.join(image_folder, image)
    #             label_file = os.path.join(label_folder, image)
    #             f1.write(image_file + '\n')
    #             f2.write(label_file + '\n')
 
    # GTA5 list
    image_text_file = os.path.join('/data/DRANet_v2/data_list', 'GtoC', 'train_imgs.txt')
    label_text_file = os.path.join('/data/DRANet_v2/data_list', 'GtoC', 'train_labels.txt')
    # if not os.path.exists(image_text_file) or not os.path.exists(label_text_file):
    image_root = os.path.join('/data/DRANet_v2/data', 'GtoC', 'Images')
    label_root = os.path.join('/data/DRANet_v2/data', 'GtoC', 'GT')
    image_list = os.listdir(image_root)
    label_list = os.listdir(label_root)
    img_num_list = []
    label_num_list = []
    for i in range(len(image_list)):
        img_num_list.append(image_list[i][0:5])
    for i in range(len(label_list)):
        label_num_list.append(label_list[i][0:5])

    f1 = open(image_text_file, mode='wt')
    f2 = open(label_text_file, mode='wt')

    for image in img_num_list:
        if image in label_num_list:
            image_file = os.path.join('/data/DRANet_v2/data/GtoC/Images', image+'.jpg')
            label_file = os.path.join('/data/DRANet_v2/data/GtoC/GT', image+'.png')
            f1.write(image_file + '\n')
            f2.write(label_file + '\n')
                               


# for mode in ['train', 'valid']:
#     print(mode)
#     image_text_file = os.path.join(os.getcwd(), 'data', 'synthetic_digits', mode + '.txt')
#     image_root = os.path.join(os.getcwd(), 'data', 'synthetic_digits', 'imgs_' + mode)
#     file_list = os.listdir(image_root)
#     f1 = open(image_text_file, mode='wt')
#     for folder in file_list:
#         image_folder = os.path.join(image_root, folder)
#         image_list = os.listdir(image_folder)
#
#         for image in image_list:
#             # image_name_split = image.split('_')
#             # label = image_name_split[0] + '_' + image_name_split[1] + '_' + image_name_split[2] + '_' + 'gtFine_labelIds.png'
#             image_file = os.path.join(image_folder, image)
#             # label_file = os.path.join(label_folder, label)
#             image_file_split = image_file.split('/')
#             label = image_file_split[-1].split('_')[0]
#
#             f1.write(image_file + ' ' + str(label) + '\n')
#             # f2.write(label_file + '\n')

if __name__ == '__main__':
    make_list()