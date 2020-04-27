import numpy as np
from skimage import io
from skimage.transform import resize
from multiprocessing import Pool
import matplotlib.pyplot as plt

# HOG
from skimage.feature import hog
# adaboost
from sklearn.ensemble import AdaBoostClassifier
from joblib import load,dump
import time

def load_im_test():
    test_ = [np.array(io.imread('test/%04d.jpg' % (i_ + 1),as_gray=True)) for i_ in range(500)]
    return test_


def min_face(img_, F_, size_h_, size_l_, step_size_):
    '''
    Expand the image to make the smallest face of all images suit for the size of windows.
    With this, we can just reduce the size of image to creat pyramids without 
    risk of missing small facces.
    Use size_h_+step_size_incase and size_l_+step_size_ that the image is too small.
    input:
        img_: input image of shape (x,y)
        F_: fraction, window size / smmallest face size 
        size_h_: size of lignes of window, default 60
        size_l_: size of colonnes of window, default 40
        step_size_: size of every step, default 10
    return :
        imag_: expanded image
    '''
    x_, y_ = img_.shape
    x_re_ = int(float(x_)*F_)
    y_re_ = int(float(y_)*F_)
    if x_re_ <= size_h_+step_size_ or y_re_ <= size_l_+step_size_:
        print (None)

    img_ = resize(img_, (x_re_, y_re_), anti_aliasing = True)
    return img_


def pyramid(image_, f_, size_h_ = 60,size_l_ = 40):
    '''
    creat pyramid of images with ratio f 
    until the size smmaller than window's size
    input:
        image_ : image, numpy array
        size_h_: size of lignes of window, default 60
        size_l_: size of colonnes of window, default 40
        f: ratio of scaling
    return :
        img_pyd: list of images of different shapes
    '''
    x_, y_ = image_.shape
    img_pyd = []
    while( x_ > size_h_ and y_ > size_l_):
        img_pyd.append(image_)
        x_ = int(x_ * f_)
        y_ = int(y_ * f_)
        image_ = resize(image_, (x_, y_),anti_aliasing = True)
    return img_pyd


def slide_windows(imsize_x_, imsize_y_, ind_, size_h_, size_l_, step_size_):
    '''
    Creat labels of  windows slide on one image
    input:
        imsize_x_: size x of image
        imsize_y_: size y of image
        ind_: index of image, store in first columns
        size_h_: size of lignes of window, default 60
        size_l_: size of colonnes of window, default 40
        step_size_: size of every step, default 10
    return :
        windows_: array of labels of windows, every window stored in one lign
    '''

    index_ = ind_+1
    if imsize_x_ < size_h_ or imsize_y_ < size_l_:
        return None
    nb_total = ( (imsize_x_ - size_h_)//step_size_ + 1 ) * ( (imsize_y_ - size_l_)//step_size_ + 1 )
    windows_ = np.zeros((nb_total,6))
    k_ = 0
    for i_ in range(0, imsize_x_ - size_h_, step_size_):
        for j_ in range(0, imsize_y_ - size_l_, step_size_):
            windows_[k_] = np.array([index_, i_, j_, size_h_, size_l_, 0])
            k_ += 1
    return windows_


def hog_feature(image_, size_h_ = 60, size_l_=40, ori_=9, cell_=8, block_=3):
    '''
    Generate hog features with image of shape (size_h_,size_l_), 
    or images of shape (n,size_h_,size_l_)
    input:
        image_ : image(s), numpy array
        size_h_: size of lignes of image(s), default 60
        size_l_: size of colonnes of image(s), default 40
        ori_: number of orientations of hog, default 9
        cell_: sqrt(number) of pixels in one cell, default 8
        block_: sqrt(number) of cells in one block, default 3
    return :
        hog_features: numpy array of hog features, each hog feature stored in lignes
    '''
    #calculate the size of output hog features
    size_ = ori_ * block_ * block_ * ((size_h_ - cell_ * (block_ - 1)) // cell_) * (
            (size_l_ - cell_ * (block_ - 1)) // cell_)
    
    # for 1 vector(image)   
    if image_.ndim == 2:
        hog_features = hog(image_, orientations=ori_, pixels_per_cell=(cell_, cell_),
                           cells_per_block=(block_, block_), block_norm='L2-Hys',
                           visualize=False, transform_sqrt=False,
                           feature_vector=True, multichannel=None)
    # for n vectors(images)
    else:
        hog_features = np.zeros((image_.shape[0], size_))
        for j_, image_j in enumerate(image_):
            hog_features[j_] = hog(image_j, orientations=ori_, pixels_per_cell=(cell_, cell_),
                        cells_per_block=(block_, block_), block_norm='L2-Hys',
                        visualize=False, transform_sqrt=False,
                        feature_vector=True, multichannel=None)
    return hog_features


def prdict_face(window_hogs_, models_):
    score_ = np.zeros((window_hogs_.shape[0]))
    mask_ = np.ones((window_hogs_.shape[0]))
    for i in range(5):
        score_ = models_[i].decision_function(window_hogs_)
        mask_ = mask_ * (models_[i].decision_function(window_hogs_)>0.1)
    score_[~mask_.astype(bool)] = 0
    
#    score_ = models_.decision_function(window_hogs_)
#    mask_ = models_.predict(window_hogs_)
#    score_[~mask_.astype(bool)] = 0

    return score_


def IoU(a_, b_):
    '''
    Calculate le overlap
    input:
        a,b: labels with : index of image, i, j, h, l = a[0:5]
    return:
        fraction (intersect surface / Union surface)
    '''
    a_y_ = a_[1] + a_[3]
    a_x_ = a_[2] + a_[4]
    b_y_ = b_[1] + b_[3]
    b_x_ = b_[2] + b_[4]
    overlapY_ = (b_y_ - b_[1]) + (a_y_ - a_[1]) - (max(a_y_, b_y_) - min(a_[1], b_[1]))
    overlapX_ = (b_x_ - b_[2]) + (a_x_ - a_[2]) - (max(a_x_, b_x_) - min(a_[2], b_[2]))

    if overlapY_ <= 0 or overlapX_ <= 0:
        Inter_ = 0
    else:
        Inter_ = overlapX_ * overlapY_
    Union_ = a_[3] * a_[4] + b_[3] * b_[4] - Inter_
    return Inter_ / Union_


def NMS(labels_):
    '''
    Non Maximum Suppression.
    Keep the labels who have the biggist score, and delete all others overlapping with it 
    input:
        labels_: labels of windows with score in the last column

    return :
        labels_[mask_.astype(bool)]: labels after NMS
    '''

    mask_ = np.ones((labels_.shape[0]))
    for i_ in range(labels_.shape[0] - 1):
        for j_ in range(i_ + 1, labels_.shape[0]):
            inter_ = IoU(labels_[i_], labels_[j_])
            if inter_ > 0.5:
                if labels_[i_, 5] > labels_[i_, 5]:
                    mask_[j_] = 0
                else:
                    mask_[i_] = 0
    return labels_[mask_.astype(bool)]


def get_label(image_, ind_, models_, F_, f_, size_h_=60, size_l_=40, step_size_=10 ):
    '''
    Get the false positives
    input:
        image_: input image of shape (im_x_,im_y_)
        ind_: index of image, store in first columns
        labels_: labels of faces in input image
        F_: fraction, window size / smmallest face size 
        f: ratio of scaling in creating image pyramid
        threshold_decision: threshold of result of decision function.         
                            for Adaboost, the decision function equals to
                                2 * (probability of 1 - probability of 0 )
        size_h_: size of lignes of window, default 60
        size_l_: size of colonnes of window, default 40
        step_size_: size of every step, default 10
    return :
        fp: labels of falses positives
    '''
    print("\nThe {}th image:".format(ind_+1))
    face_labels_ = np.zeros((0,6))
    # get the image of max size
    image_max = min_face(image_, F_, size_h_, size_l_, step_size_) 
    print("\nGet pyramid!")
    pyds = pyramid(image_max, f_, size_h_,size_l_) # get image pyramid

    print("\nBegin slide windows in pyramid!")

    for ind_pyd, pyd in enumerate(pyds):
        print("\nThe {}th class in pyramid!".format(ind_pyd+1))
        imsize_x, imsize_y = pyd.shape
        # get labels of windows
        window_labels_ = slide_windows(imsize_x, imsize_y, ind_, size_h_, size_l_, step_size_)
        if window_labels_.shape[0] == 0:
            continue
        window_images = np.zeros((window_labels_.shape[0],size_h_,size_l_))
        # for every label, get the related image
        for k, window_ in enumerate(window_labels_):
            x,y = list(map(int,window_[1:3]))         
            window_images[k] = pyd[x:x+size_h_,y:y+size_l_]
        
        # get hog features of windows
        hog_ = hog_feature(window_images, size_h_, size_l_)
        # predict with decision function.
        #score_ = prdict_face(hog_, models_)
        score_ = models_.decision_function(hog_)
        # stock the  result s in labels of windows
        window_labels_[:,5] = score_
        #window_labels_ = window_labels_[score_!=0]
        window_labels_ = window_labels_[score_>-0.3]

        ratio = f_**ind_pyd # ind_pyd is number of times of resize, so f_**ind_pyd is the scal ratio
        # make the matrix of transformation
        trans_ = np.diag(np.array((1, 1 / (F_*ratio), 1 / (F_*ratio), 1 / (F_*ratio), 1 / (F_*ratio), 1)))
        # get the real labels
        window_labels_ = np.dot(window_labels_, trans_)
        #delete windows which is too small
        window_labels_ = window_labels_[(window_labels_[:,3]>=5) * (window_labels_[:,4]>=5)]
        face_labels_ = np.concatenate((face_labels_,window_labels_),axis = 0)
        print("\nFaces detected size is {}".format(face_labels_.shape)) 
    print("\nFaces detected size is {}".format(face_labels_.shape))    

    # set labels i,j,h,l to integer
    face_labels_[:,0:5] = face_labels_[:,0:5].astype(int)
    # delete overlaping windows
    face_labels_ = NMS(face_labels_)

    return face_labels_


def detect_with_multip(images_, models_, F_, f_):
    '''
    get_label(image_, ind_, model_, F_, f_, 
                                      size_h_=60, size_l_=40, step_size_=10 )
    Because of the long time of ruuning algorithms, choose multiprocess to run it faster.
    Choose pool of 8 process.
    And divide all data in 5 parts just to track the progress.
    input:
        input nesessary variables for the function get_fp
    return:
        fp_labels: labels of all falses possitives
    '''
    fp_labels_results_ = []
    print("\n Begin slide window!")
    print(time.asctime( time.localtime(time.time()) ))
    
    #20.37
    pool1 = Pool(processes=6)
    for i in range(500):
        fp_labels_results_.append(pool1.apply_async(get_label, args=(images_[i],i,models_,F_,f_)))
    pool1.close()
    pool1.join()
    
    print("\n Finish detection!")
    print(time.asctime( time.localtime(time.time()) ))
    print("\n Length fp_labels_image: ",len(fp_labels_results_))

    fp_labels_ = np.zeros((0, 6))
    for fp_label_result_ in fp_labels_results_:
        fp_label_ = fp_label_result_.get()
        if fp_label_.shape[0] != 0:
            fp_labels_ = np.concatenate((fp_labels_, fp_label_), axis=0)

    return fp_labels_

def sort_with_score(face_labels_):
    label_arg_ = np.argsort(face_labels_[:,5])
    face_labels_ = face_labels_[label_arg_[::-1]]
    return face_labels_


def plot_face(image_,labels_):
    '''
    Plot the image with labels predicted
    input:
        image_: input image
        labels_: labels of faces detected in input image

    return :
            a figure
    '''

    # Trace le graphe de prédiction
    plt.figure("Image") # 图像窗口名称
    plt.imshow(image_,cmap='gray')
    plt.axis('on') # 关掉坐标轴为 off
    plt.title('image') # 图像题目
    for label in labels_:
        i_, j_, h_, l_ = label[1:5]
        plt.plot((j_, j_), (i_, i_+h_), 'r')
        plt.plot((j_+l_, j_+l_), (i_, i_+h_), 'r')
        plt.plot((j_, j_+l_), (i_, i_), 'r')
        plt.plot((j_, j_+l_), (i_+h_, i_+h_), 'r')
    plt.show()


def plot_9_images(labels_, images_, index_):
        # Trace le graphe de prédiction
        
    plt.figure("Images") # 图像窗口名称
    for ind_ in range(9):
        plt.subplot(3,3,ind_+1)
        plt.imshow(images_[ind_ + index_],cmap='gray')

        for label in labels_[labels_[:,0]==(ind_ + index_+1)]:
            i_, j_, h_, l_ = label[1:5]
            plt.plot((j_, j_), (i_, i_+h_), 'r')
            plt.plot((j_+l_, j_+l_), (i_, i_+h_), 'r')
            plt.plot((j_, j_+l_), (i_, i_), 'r')
            plt.plot((j_, j_+l_), (i_+h_, i_+h_), 'r')
    plt.axis('off') # 关掉坐标轴为 off
    plt.title('images') # 图像题目
    plt.show()
    
    
if __name__ == "__main__":
    '''
    This function is to detect faces with models trained in 'train.py' and stock 
    their labels and scores in 'detection.txt'.
    
    Labels : index, i, j, h, l, s in every lign
            index: index of current image        
            i, j: coordinate of the left top corner of current image 
            h, l: length and width of current image
            s: score of the detected face (Here I summ the scores from all cascade models)
    '''

    test = load_im_test()
    print(" Finish loading!")
    size_h = 60
    size_l = 40
    labels = np.zeros((0,6))
    label_s = []
    F = 3.076923076923077
    f = 0.8 #ratio to resize images
    ori = 9
    cell = 8
    block = 3   
    print("\nGet the models!")
 #   ada2 = load('secondetrain') 
#    ada2 = load('ada_train2')
#    ada1 = load('ada')
    
    svm2 = load('svm_train2')
    
    print("\nBegin detection!")
    
    
    
#    face_labels_ada = detect_with_multip(test, ada2, F, f)
#    face_labels_ada[0:5] = face_labels_ada[0:5].astype(int)
#    face_labels_ada_sorted = sort_with_score(face_labels_ada)
#    
#    np.savetxt('detection.txt',face_labels_ada_sorted,fmt='%d %d %d %d %d %.2f')    

    face_labels_svm = detect_with_multip(test, svm2, F, f)
    face_labels_svm[0:5] = face_labels_svm[0:5].astype(int)
    face_labels_svm_sorted = sort_with_score(face_labels_svm)
    dump(face_labels_svm_sorted, 'face_labels_svm_sorted') 
    np.savetxt('detection_svm.txt',face_labels_svm_sorted,fmt='%d %d %d %d %d %.2f')    


#    plot_9_images(face_labels_svm_sorted, test, 290)
#    i = 137
#    detected = face_labels_svm_sorted[face_labels_svm_sorted[:,0]==(i+1)]
#    plot_face(test[i],detected)
#    
    
#    lab = get_label(zsnb, 0, ada2, F, f, size_h_=60, size_l_=40, step_size_=10 )
#    lab[0:5] = lab[0:5].astype(int)
#
#    lab_ada_sorted = sort_with_score(lab)
#    plot_face(zsnb,lab_ada_sorted[0:1])
#    
    
#    lab_svm = get_label(test[205], 0, svm2, F, f, size_h_=60, size_l_=40, step_size_=10 )
#    lab_svm[0:5] = lab_svm[0:5].astype(int)
#    plot_face(test[205],lab_svm)
    
#    
#    
#    labels = np.loadtxt('detection1.txt')
#    i = 500
#    lab_plot = labels[labels[:,0]==i]
#    plot_face(test[i-1],lab_plot[0:2])
