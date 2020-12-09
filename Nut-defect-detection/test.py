import warnings
warnings.filterwarnings('ignore')
import sys
import paddle.fluid as fluid
import paddle
import numpy as np
import os
import pandas as pd

paddle.enable_static()

def load_img(img):
    is_color = True
    resize_size = 448
    # crop_size = 100  # 剪切尺寸，最后图片的size是这个值，不是resize_size
    img = paddle.dataset.image.load_image(file=img, is_color=is_color)
    img = paddle.dataset.image.simple_transform(im=img,
                                                resize_size=resize_size,
                                                is_color=is_color, is_train=False)
    img = img.astype('float32')
    img *= 0.007843
    img -= [0.5, 0.5, 0.5]
    img = img / 0.5
    return img

def pred_data(path_img_test):
# 构建执行器
    USE_GPU = False
    place = fluid.CUDAPlace(0) if USE_GPU else fluid.CPUPlace()  # 使用CPU执行训练
    infer_exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()
    # 载入model
    with fluid.scope_guard(scope=inference_scope):
        currentpath = os.path.dirname(sys.argv[0])
        [inference_program,
         feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(
            dirname=os.path.join(currentpath, 'model_'), executor=infer_exe)

        test_imgs_dir = path_img_test
        img_data, img_paths, img_names = [], [], []
        for img_name in os.listdir(test_imgs_dir):
            img_path = os.path.join(test_imgs_dir, img_name)
            img_paths.append(img_path)
            img_data.append(load_img(img_path))
            img_names.append(img_name.split('.')[0])
        img_data = np.array(img_data).astype("float32")

        result = infer_exe.run(program=inference_program,
                               feed={feed_target_names[0]: img_data},
                               fetch_list=fetch_targets)
    infer_label = [np.argmax(x) for x in result[0]]
    class_dict = {0: 'neg', 1: 'pos'}
    pred_class = []
    for i in infer_label:
        pred_class.append(class_dict[i])
    submit = pd.DataFrame({'id': img_names, 'label': pred_class})
    return submit

def main(path_img_test, path_submit):
    # 预测图片
    result = pred_data(path_img_test)
    # 写出预测结果
    result.to_csv(path_submit, index=None, encoding='utf-8')
	
if __name__ == "__main__":
    path_img_test = sys.argv[1]
    path_submit = sys.argv[2]

    # path_img_test = r'C:\Users\11982\Desktop\dataset\luomu_detect\trainset\test'
    # path_submit = r'C:\Users\11982\Desktop\dataset\test.csv'
    main(path_img_test, path_submit)
