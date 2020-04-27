import numpy as np
import matplotlib.pyplot as plt
import random
import time
from multiprocessing import Pool

from skimage import io
from skimage.transform import resize
from sklearn.model_selection import validation_curve
from sklearn.metrics import classification_report
# HOG
from skimage.feature import hog

# adaboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc 
from sklearn.model_selection import train_test_split
# save model
from joblib import dump,load


def load_im_train():
    '''
    Load training images
    output:
        train_: train images of shape (x,y), list of length 1000
    '''

    train_ = [np.array(io.imread('train/%04d.jpg' % (i_ + 1),as_gray = True)) for i_ in range(1000)]
    return train_


def sqr_pos_face(labels_,size_h_,size_l_):
    face_labels_ = np.zeros((labels_.shape))
    for ind_, label_ in enumerate(labels_):
        ind_im_, i_, j_, h_, l_ = label_[0:5]
        diff_h_ = int(abs(h_ - 1.5*l_)//2)
        diff_l_ = int(abs(h_ - 1.5*l_)//3)
        if h_ < (1.5*l_):
            if i_ < diff_h_:
                i_ = 0
            else:
                i_ = i_ - diff_h_
            h_ = 1.5*l_
        if h_ > (1.5*l_):
            if j_ < diff_l_:
                j_ = 0
            else:
                j_ = j_ - diff_l_
            l_ = h_/1.5
        face_labels_[ind_] = np.array([ind_im_, i_, j_, h_, l_]).astype(int)
    return face_labels_

def generate_neg(images_, size_h_, size_l_, nb_ = 2000):
    '''
    Generate negative examples
    input:
        im_: image
        size_h_: size of lignes of negative example, default 60
        size_l_: size of colonnes of negative example, default 40
    return :
        neg_: negative examples of format image
        neg_l: negative examples of format labels
    '''
    neg_ = np.empty((nb_, size_h_, size_l_))
    neg_l = np.zeros((nb_,6))
    ind_ = 0
    for im_ in images_:
        x_, y_ = im_.shape
        for k_ in range(4):
            i_ = random.randint(1, x_ - size_h_) - 1
            j_ = random.randint(1, y_ - size_l_) - 1
            neg_[ind_] = im_[i_:i_ + size_h_, j_:j_ + size_l_]
            neg_l[ind_] = np.array([0,i_,j_,size_h_,size_l_,0])
            ind_ += 1
    return neg_,neg_l


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


def IoU(a_, b_):
    '''
    Calculate le overlap
    input:
        a,b: labels with : index of image, i, j, h, l = a[0:5]
    return:
        fraction (intersect surface / Union surface)
    '''
    
    a_x_ = a_[1] + a_[3]
    a_y_ = a_[2] + a_[4]
    b_x_ = b_[1] + b_[3]
    b_y_ = b_[2] + b_[4]

    overlapX_ = b_[3] + a_[3] - (max(a_x_, b_x_) - min(a_[1], b_[1]))
    overlapY_ = b_[4] + a_[4] - (max(a_y_, b_y_) - min(a_[2], b_[2]))
    if overlapY_ == min(a_[2], b_[2]) and overlapX_ == min(a_[1], b_[1]):
        return 1
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


def get_fp(image_, ind_, labels_, model_, F_, f_, threshold_decision_ = 0.15,
           size_h_=60, size_l_=40, step_size_=10 ):
    '''
    Get the false positives
    input:
        image_: input image of shape (im_x_,im_y_)
        ind_: index of image, store in first columns
        labels_: labels of faces in input image
        F_: fraction, window size / smmallest face size 
        f_: ratio of scaling in creating image pyramid
        threshold_decision_: threshold of result of decision function.         
                            for Adaboost, the decision function equals to
                                2 * (probability of 1 - probability of 0 )
        size_h_: size of lignes of window, default 60
        size_l_: size of colonnes of window, default 40
        step_size_: size of every step, default 10
    return :
        fp_: labels of falses positives
    '''

    print("\nthe {}th image:".format(ind_+1))
    label_ = labels_[labels_[:, 0] == (ind_ + 1)]
    # get the image of max size
    image_max_ = min_face(image_, F_, size_h_, size_l_, step_size_) 
    print("\nGet pyramid!")
    pyds_ = pyramid(image_max_, f_, size_h_,size_l_) # get image pyramid

    positives_ = np.zeros((0,6))
    print("\nBegin slide windows in pyramid!")
    print("\nPyramidsize: {}".format(len(pyds_)))
    for ind_pyd_, pyd_ in enumerate(pyds_):
        print("the {}th image in pyramid!".format(ind_pyd_ + 1))
        imsize_x, imsize_y = pyd_.shape
        # get labels of windows
        window_labels_ = slide_windows(imsize_x, imsize_y, ind_, size_h_, size_l_, step_size_)
        if window_labels_.shape[0] == 0:
            continue
        window_images_ = np.zeros((window_labels_.shape[0],size_h_,size_l_))
        # for every label, get the related image
        for k, window_ in enumerate(window_labels_):
            x,y = list(map(int,window_[1:3]))         
            window_images_[k] = pyd_[x:x+size_h_,y:y+size_l_]

        # get hog features of windows
        h_ = hog_feature(window_images_, size_h_, size_l_)
        # predict with decision function.
        s_ = model_.decision_function(h_)
        
        # stock the  result s in labels of windows
        window_labels_[:,5] = s_

        # if result s is bigger than thres hold, get the label
        positif_ = window_labels_[s_ > threshold_decision_]

        ratio_ = f_**ind_pyd_ # ind_pyd_ is number of times of resize, so f_**ind_pyd_ is the scal ratio
        # make the matrix of transformation
        trans_ = np.diag(np.array((1, 1 / (F_*ratio_), 1 / (F_*ratio_), 1 / (F_*ratio_), 1 / (F_*ratio_), 1)))
        # get the real labels
        positif_ = np.dot(positif_, trans_)
        #delete windows which is too small
        positif_ = positif_[(positif_[:,3]>=5) * (positif_[:,4]>=5)]
        positives_ = np.concatenate((positives_, positif_),axis = 0)

    # set labels i,j,h,l to integer
    positives_[:,0:5] = positives_[:,0:5].astype(int)

    # delete positives examples who have overlap with true face bigger than 0.5
    mask_ = np.ones((positives_.shape[0]))
    for indp_, p_ in enumerate(positives_):
        overlap_ = 0
        for l_ in label_:
            overlap_ = max(overlap_,IoU(p_,l_))
        if overlap_ > 0.5:
            mask_[indp_] = 0
    fp_ = positives_[mask_.astype(bool)]
    # delete overlaping windows
    fp_ = NMS(fp_)

    return fp_#,positives_[~mask_.astype(bool)]


def train_with_multip(train_, labels_, model_, F_, f_):
    '''
    Because of the long time of ruuning algorithms, choose multiprocess to run it faster.
    Choose pool of 8 process.
    And divide all data in 5 parts just to track the progress.
    input:
        input nesessary variables for the function get_fp
    return:
        fp_labels_: labels of all falses possitives
    '''

    fp_label_results_ = []
    print("\nBegin slide window!")
    print(time.asctime( time.localtime(time.time()) ))
    begin_ = time.time()


    pool1 = Pool(processes=8)
    for i_ in range(200):
        fp_label_results_.append(pool1.apply_async(get_fp, args=(train_[i_], i_, labels_, model_, F_, f_)))
    pool1.close()
    pool1.join()
    print("\n200!")
    print(time.asctime( time.localtime(time.time()) ))


    pool2 = Pool(processes=8)
    for i_ in range(600):
        fp_label_results_.append(pool2.apply_async(get_fp, args=(train_[i_+200], i_+200, labels_, model_, F_, f_)))
    pool2.close()
    pool2.join()
    print("\n800!")
    print(time.asctime( time.localtime(time.time()) ))


    pool3 = Pool(processes=8)
    for i_ in range(200):
        fp_label_results_.append(pool3.apply_async(get_fp, args=(train_[i_+800], i_+800, labels_, model_, F_, f_)))
    pool3.close()
    pool3.join()
    print("\n1000!")    
    print(time.asctime( time.localtime(time.time()) ))

    fp_labels_ = np.zeros((0, 6))

    for fp_label_result_ in fp_label_results_:
        fp_label_ = fp_label_result_.get()
        if fp_label_.shape[0] != 0:
            fp_labels_ = np.concatenate((fp_labels_, fp_label_), axis=0)

    end_ = time.time()
    
    duration_ = (end_ - begin_)//60
    print("\nDuration of getting FP: {} mins".format(duration_))
    
    return fp_labels_


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


def plot_roc(X_, Y_, model):

    X_train, X_test, y_train, y_test = train_test_split(X_, Y_, test_size=.3,
                                                    random_state=0)
    y_score = model.fit(X_train, y_train).decision_function(X_test)

    fpr,tpr,threshold = roc_curve(y_test, y_score) ###计算真正率和假正率
    roc_auc = auc(fpr,tpr) ###计算auc的值

    plt.figure()
    lw = 2
    
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve with SVM after adding false positives')
    plt.legend(loc="lower right")
    plt.show()
    

   
    
if __name__ == "__main__":
    '''
    This function is to train a model with images of peoples in the folder 'train'
    with labels of faces given in file 'label_train.txt'. 
    Labels : index, i, j, h, l in every lign
            index: index of current image        
            i, j: coordinate of the left top corner of current image 
            h, l: length and width of current image

    This function repeat two phases as follows to finish:
        First phase:
            1. Get faces in images given, and randomly pick certain number of faces from image
            2. Use functions of feature extraction to get descriptive features. (eg. HOG, Haar)
                ps: Here I choose HOG because its quicker and can detect with profiles
            3. Train a model with features extracted. (eg. Adaboost, SVM)
                ps: Here I use Adaboost because it's good enough
            
            
    Second step:
            1. Detect faces using the model train in the first phase with algorithm Slide Windows 
                in image while size of window doesn't change and size of image changes.
                (Because we can easily resize images with functions like rescal, resize in
                package Skimage) And find all the Falses Positives to add into the faux examples> 
            2. Use functions of feature extraction to get descriptive features. 
            3. Retrain the model with features extracted. 
            
    And for better predict a face, I have used cascade classifiers in the last detection.
    At last, save the models using joblib.dump
    '''

    # get the data
    train = load_im_train()
    print("Finish loading data!")
    # get labels
    labels = np.loadtxt('label_train.txt', dtype=int)  # k, i, j, h, l, s
    size_h = 60
    size_l = 40
    # F means max fraction of (window size / smmallest face size) 3.076923076923077
    F = max(size_h/np.min(labels[:,3]),size_l/np.min(labels[:,4]))
    f = 0.8 #ratio to resize images
    im_index, i, j, k, l = labels[0]
    threshold_decision = 0.3
    # generate positif examples
    nb_pos = labels.shape[0]
    pos = np.zeros((labels.shape[0], size_h, size_l))
    face_sqr = sqr_pos_face(labels, size_h,size_l)
    for m, label in enumerate(face_sqr):
        im_indexm, im, jm, km, lm = list(map(int,label))
        pos[m] = resize(train[im_indexm - 1][im:im + km, jm:jm + lm], (size_h, size_l), anti_aliasing=True)

    # generate negatif examples randomly
    nb_neg = 2000
    neg,neg_label = generate_neg(train[0:500], size_h, size_l, nb_neg)
    
    # stock negatif and possitif together to creat train sets X_train_q and Y_train_1
    train_1 = np.concatenate([pos, neg], axis=0)
    nb_train_1 = nb_neg + nb_pos
    Y_train_1 = np.array([[1] * nb_pos + [-1] * nb_neg]).T.reshape((-1,))
    print("\nFinish gererating negative!")

    # feature extraction
    ori = 9
    cell = 8
    block = 3
    #for X, get their hog feature to train
    X_train_1 = hog_feature(train_1, size_h, size_l, ori, cell, block)
    print("\nFinish get hog features!")

    '''
    # first train
    '''
    # Adaboost
    '''
    This part is used to find the best maximum number of estimators for AdaboostClassifier
    Finally nb of estimators don't make big differences. And 80 seems to be a good choice.

    #CV
    np.random.seed(0)
    indices = np.arange(Y_train_1.shape[0])
    np.random.shuffle(indices)
    param_range = np.arange(5,40,5)
    train_scores, valid_scores = validation_curve(AdaBoostClassifier(), X_train_1, Y_train_1, 
                                                  "n_estimators",param_range, cv=8)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(valid_scores, axis=1)
    test_scores_std = np.std(valid_scores, axis=1)

    plt.title("Validation Curve with Adaboost")
    plt.xlabel("n_estimators")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig('./cross_validation.jpg')
    plt.show()
    '''
    

    target_names = ['class 0', 'class 1']
    X_train, X_test, y_train, y_test = train_test_split(X_train_1, Y_train_1, test_size=.3,
                                                    random_state=0)
    
    ada = AdaBoostClassifier(n_estimators=50)
    '''  
    y_pred = ada.fit(X_train, y_train).predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names))
    report_ada_train1 = classification_report(y_test, y_pred, target_names=target_names)
    '''
    ada.fit(X_train_1, Y_train_1)
    dump(ada, 'ada_train1') 
    print("\nFinish training with Adaboost!")


    svm = SVC(gamma='auto')
    '''
    y_pred = svm.fit(X_train, y_train).predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names))
    report_svc_train1 = classification_report(y_test, y_pred, target_names=target_names)
    '''
    svm.fit(X_train_1, Y_train_1)
    dump(svm, 'svm_train1')
    print("\nFinish training with SVM!")
    

    # get fp
    # slides windows
    # mean h/l  = 1.5195370402700235
    # biggest face:407 275 smallest window:23 13
    # so the image size vary from 60/407 (0.14742/0.15) to 40/13 (3)

#    ada = load('ada_train1')
#    svm = load('svm_train1')
    
    print("\nBegin getting fp with Adaboost and SVM!")
    fp_labels_ada = train_with_multip(train, labels, ada, F, f)

    fp_labels_ada[0:5] = fp_labels_ada[0:5].astype(int)
#    93mins
    fp_labels_svm = train_with_multip(train, labels, svm, F, f)
    fp_labels_svm[0:5] = fp_labels_svm[0:5].astype(int)

    dump(fp_labels_ada, 'fp_labels_ada.pkl') 
    dump(fp_labels_svm, 'fp_labels_svm.pkl') 
#    fp_labels_ada = load('fp_labels_ada.pkl')
#    fp_labels_svm = load('fp_labels_svm.pkl')
    # second train step
    # after getting labels of fps, get the related image and begin the second train step
     
    label_arg_ada = np.argsort(fp_labels_ada[:,5])
    fp_labels_ada = fp_labels_ada[label_arg_ada[::-1]]
    fp_labels_ada = fp_labels_ada[0:3000]

    fp_images_ada = np.zeros((0,60,40))
    for fp_label_i in fp_labels_ada:
        index_image, i_fp, j_fp, h_fp, l_fp = list(map(int,fp_label_i[0:5]))
        print(index_image, i_fp, j_fp, h_fp, l_fp)
        fp_window_i = train[index_image-1][i_fp:i_fp+h_fp,j_fp:j_fp+l_fp]
        fp_image_i = resize(fp_window_i,(size_h,size_l),anti_aliasing=True).reshape((1,size_h,size_l))
        fp_images_ada = np.concatenate((fp_images_ada, fp_image_i), axis=0)
    

    fp_hog_ada = hog_feature(fp_images_ada, size_h, size_l, ori, cell, block)
    X_train_2_ada = np.concatenate((X_train_1, fp_hog_ada),axis=0)
    Y_train_2_ada = np.concatenate((Y_train_1, -1 * np.ones(fp_hog_ada.shape[0])),axis=0)
    
    dump(fp_labels_ada, 'fp_labels_ada_3000.pkl') 
    dump(X_train_2_ada, 'X_train_2_ada.pkl') 
    dump(Y_train_2_ada, 'Y_train_2_ada.pkl')
    
    
    
    label_arg_svm = np.argsort(fp_labels_svm[:,5])
    fp_labels_svm = fp_labels_svm[label_arg_svm[::-1]]
    fp_labels_svm = fp_labels_svm[0:3000]

    fp_images_svm = np.zeros((0,60,40))
    for fp_label_i in fp_labels_svm:
        index_image, i_fp, j_fp, h_fp, l_fp = list(map(int,fp_label_i[0:5]))
        print(index_image, i_fp, j_fp, h_fp, l_fp)
        fp_window_i = train[index_image-1][i_fp:i_fp+h_fp,j_fp:j_fp+l_fp]
        fp_image_i = resize(fp_window_i,(size_h,size_l),anti_aliasing=True).reshape((1,size_h,size_l))
        fp_images_svm = np.concatenate((fp_images_svm, fp_image_i), axis=0)
    
    # second train step
    fp_hog_svm = hog_feature(fp_images_svm, size_h, size_l, ori, cell, block)
    X_train_2_svm = np.concatenate((X_train_1, fp_hog_svm),axis=0)
    Y_train_2_svm = np.concatenate((Y_train_1, -1 * np.ones(fp_hog_svm.shape[0])),axis=0)
    
    dump(fp_labels_svm, 'fp_labels_svm_3000.pkl') 
    dump(X_train_2_svm, 'X_train_2_svm.pkl') 
    dump(Y_train_2_svm, 'Y_train_2_svm.pkl') 

 
    
    '''  
    f_labels,tp_labels = get_fp(train[1], 1, labels, ada, F, f, threshold_decision_ = 0, #0.2
           size_h_=60, size_l_=40, step_size_=10 )
    plot_face(train[0],f_labels)
    '''

    '''
    for index_image in range(1000): 
        for fp_label_i in fp_labels[fp_labels[:,0]==(index_image+1)]:
            i_fp, j_fp, h_fp, l_fp = fp_label_i[1:5]
            fp_window_i = train[index_image][i_fp:i_fp+h_fp,j_fp:j_fp+l_fp]
            fp_image_i = resize(fp_window_i,(60,40),anti_aliasing=True).reshape((1,60,40))
            fp_images = np.concatenate((fp_images, fp_image_i), axis=0)
    '''
    
#    plot_roc(X_train_2_ada, Y_train_2_ada, ada)
#    
#    
#    target_names = ['class 0', 'class 1']
#    X_train, X_test, y_train, y_test = train_test_split(X_train_2_ada, Y_train_2_ada, test_size=.3,
#                                                    random_state=0)
#    y_pred = ada.fit(X_train, y_train).predict(X_test)
#    print(classification_report(y_test, y_pred, target_names=target_names))
#    report_ada_train2 = classification_report(y_test, y_pred, target_names=target_names)
#    dump(report_ada_train2,'report_ada_train2')
#    
#    
#    plot_roc(X_train_2_svm, Y_train_2_svm, svm)
#    target_names = ['class 0', 'class 1']
#    X_train, X_test, y_train, y_test = train_test_split(X_train_2_svm, Y_train_2_svm, test_size=.3,
#                                                    random_state=0)    
#    y_pred = svm.fit(X_train, y_train).predict(X_test)
#    print(classification_report(y_test, y_pred, target_names=target_names))
#    report_svm_train2 = classification_report(y_test, y_pred, target_names=target_names)
#    dump(report_svm_train2,'report_svm_train2')
#    

    
    
    svm2 = SVC(gamma='auto')
    svm2.fit(X_train_2_svm,Y_train_2_svm)
    dump(svm2,'svm_train2')
    ada2 = AdaBoostClassifier(n_estimators=50)
    ada2.fit(X_train_2_ada,Y_train_2_ada)
    dump(ada2, 'ada_train2') 
    
    
    
    '''
    ada2 = []
    ada2_1 = AdaBoostClassifier(n_estimators=20)
    ada2_2 = AdaBoostClassifier(n_estimators=40)
    ada2_3 = AdaBoostClassifier(n_estimators=60)
    ada2_4 = AdaBoostClassifier(n_estimators=80)
    ada2_5 = AdaBoostClassifier(n_estimators=100)
    ada2_1.fit(X_train_2, Y_train_2)
    ada2_2.fit(X_train_2, Y_train_2)
    ada2_3.fit(X_train_2, Y_train_2)
    ada2_4.fit(X_train_2, Y_train_2)
    ada2_5.fit(X_train_2, Y_train_2)
    ada2 = [ada2_1,ada2_2,ada2_3,ada2_4,ada2_5]
    dump(ada2, 'secondetrain') 
    '''