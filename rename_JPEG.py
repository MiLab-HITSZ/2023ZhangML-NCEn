import os


class ImageRename():
    def __init__(self):
        self.path = r"./data/voc2007/VOCdevkit/VOC2007/JPEGImages_old"  # 需要改名的文件

        self.path1 = r"./data/voc2007/VOCdevkit/VOC2007/JPEGImages"  # 改名后文件存在的路径

    # 'norain-1000x2.png', 'norain-1001x2.png',
    def re_name(self):
        filelist = os.listdir(self.path)
        # print(filelist)

        total_num = len(filelist)
        print(total_num)
        print("=======")

        for item in filelist:

            # 对图片的名字进行分割和提取
            number1 = int(item.split(".")[-2])
            # print(str(number1))

            # first = number1.split("_")[0]
            # second = number1.split("_")[1]
            #
            # #
            # first1 = int(first)
            # second2 = int(second)

            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path1), str(number1) + '.jpg')
                os.rename(src, dst)
                print('converting %s to %s ...' % (src, dst))


if __name__ == '__main__':
    newname = ImageRename()
    newname.re_name()
