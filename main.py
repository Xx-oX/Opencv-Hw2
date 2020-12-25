import glob
import sys
import cv2
import numpy as np
import interface
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib import pyplot as plt


class MainUi(QtWidgets.QWidget, interface.Ui_Form):
    def __init__(self, parent=None):
        super(MainUi, self).__init__(parent)
        self.setupUi(self)

        # btn connect
        self.btn1_1.clicked.connect(self.func1_1)
        self.btn1_2.clicked.connect(self.func1_2)
        self.btn2_1.clicked.connect(self.func2_1)
        self.btn2_2.clicked.connect(self.func2_2)
        self.btn2_3.clicked.connect(self.func2_3)
        self.btn2_4.clicked.connect(self.func2_4)
        self.btn3_1.clicked.connect(self.func3)
        self.btn4_1.clicked.connect(self.func4)

        # init
        self.imgs = []
        self.intrinsic = np.zeros((3,3))
        self.distortion = np.zeros((5,1))
        self.rvecs = np.zeros((1,3))
        self.tvecs = np.zeros((1, 3))

        self.preprocess()

    def preprocess(self):
        CHECKERBOARD = (11, 8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objpoints = []
        imgpoints = []
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        image = glob.glob('./Hw2_Dataset/Datasets/Q2_Image/*.bmp')
        for fname in image:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                     cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
                self.imgs.append(img)

        ret2, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        self.intrinsic = mtx
        self.distortion = dist
        self.rvecs = rvecs
        self.tvecs = tvecs

    def func1_1(self):
        ori_img1 = cv2.imread('./Hw2_Dataset/Datasets/Q1_Image/coin01.jpg')
        ori_img2 = cv2.imread('./Hw2_Dataset/Datasets/Q1_Image/coin02.jpg')

        img1 = cv2.cvtColor(ori_img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(ori_img2, cv2.COLOR_BGR2GRAY)
        img1 = cv2.GaussianBlur(img1, (11, 11), 0)
        img2 = cv2.GaussianBlur(img2, (11, 11), 0)

        edge1 = cv2.Canny(img1, 100, 200)
        edge2 = cv2.Canny(img2, 100, 200)
        edge1 = cv2.cvtColor(edge1, cv2.COLOR_GRAY2BGR)
        edge2 = cv2.cvtColor(edge2, cv2.COLOR_GRAY2BGR)
        edge1 *= np.array((0, 0, 1), np.uint8)
        edge2 *= np.array((0, 0, 1), np.uint8)

        img1 = cv2.addWeighted(ori_img1, 0.5, edge1, 0.5, 0.0)
        img2 = cv2.addWeighted(ori_img2, 0.5, edge2, 0.5, 0.0)

        cv2.imshow('img1', img1)
        cv2.imshow('img2', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def func1_2(self):
        ori_img1 = cv2.imread('./Hw2_Dataset/Datasets/Q1_Image/coin01.jpg')
        ori_img2 = cv2.imread('./Hw2_Dataset/Datasets/Q1_Image/coin02.jpg')

        img1 = cv2.cvtColor(ori_img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(ori_img2, cv2.COLOR_BGR2GRAY)
        img1 = cv2.GaussianBlur(img1, (11, 11), 0)
        img2 = cv2.GaussianBlur(img2, (11, 11), 0)

        ret1, bin1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
        ret2, bin2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
        contour1, hierarchy1 = cv2.findContours(bin1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour2, hierarchy2 = cv2.findContours(bin2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # for i in range(len(contour1)):
        #     im = img1.copy()
        #     cv2.drawContours(im, contour1, i, (0, 0, 255), 3)
        #     cv2.imshow('result', im)
        #     cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 吃到外框:(
        str_1 = "There are " + str(len(contour1)-1) + "  coins in coin01.jpg"
        str_2 = "There are " + str(len(contour2)-1) + "  coins in coin02.jpg"
        self.lbl1_1.setText(str_1)
        self.lbl1_2.setText(str_2)

    def func2_1(self):
        for i in range(15):
            cv2.imshow('img', self.imgs[i])
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    def func2_2(self):
        np.set_printoptions(precision=3, suppress=True)
        print(self.intrinsic)

    def func2_3(self):
        index = self.comboBox.currentIndex() + 1
        R = np.zeros((3, 3))
        cv2.Rodrigues(self.rvecs[index], R, jacobian=0)

        extrinsic = np.hstack([R, self.tvecs[index]])
        np.set_printoptions(precision=3, suppress=True)
        print(extrinsic)

    def func2_4(self):
        np.set_printoptions(precision=3, suppress=True)
        print(self.distortion)

    def func3(self):
        img3s = []
        CHECKERBOARD = (11, 8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objpoints = []
        imgpoints = []
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        image = glob.glob('./Hw2_Dataset/Datasets/Q3_Image/*.bmp')
        for fname in image:
            img = cv2.imread(fname)
            img3s.append(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                     cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

        ret2, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        pyramid = np.array([[1, 1, 0], [3, 5, 0], [5, 1, 0], [3, 3, -3]], np.float32)
        imgPs = []
        for i in range(5):
            imgP, jac = cv2.projectPoints(pyramid, rvecs[i], tvecs[i], mtx, dist)
            imgPs.append(imgP)

        for i in range(5):
            plt.figure(i)
            plt.plot([imgPs[i][0][0][0], imgPs[i][1][0][0]], [imgPs[i][0][0][1], imgPs[i][1][0][1]])
            plt.plot([imgPs[i][0][0][0], imgPs[i][2][0][0]], [imgPs[i][0][0][1], imgPs[i][2][0][1]])
            plt.plot([imgPs[i][0][0][0], imgPs[i][3][0][0]], [imgPs[i][0][0][1], imgPs[i][3][0][1]])
            plt.plot([imgPs[i][1][0][0], imgPs[i][2][0][0]], [imgPs[i][1][0][1], imgPs[i][2][0][1]])
            plt.plot([imgPs[i][1][0][0], imgPs[i][3][0][0]], [imgPs[i][1][0][1], imgPs[i][3][0][1]])
            plt.plot([imgPs[i][2][0][0], imgPs[i][3][0][0]], [imgPs[i][2][0][1], imgPs[i][3][0][1]])
            plt.imshow(img3s[i])
            plt.axis('off')

        i = 0
        self.b = 0

        def onPress(event):
            if event.key == 'enter':
                self.b = 1

        while True:
            if self.b != 0:
                break
            if i == 5:
                i = 0
            fig = plt.figure(i)
            fig.canvas.mpl_connect('key_press_event', onPress)
            plt.pause(1)
            i += 1

        for i in range(5):
            fig = plt.figure(i)
            plt.close(fig)

    def func4(self):
        imgL = cv2.imread('./Hw2_Dataset/Datasets/Q4_Image/imgL.png', 0)
        imgR = cv2.imread('./Hw2_Dataset/Datasets/Q4_Image/imgR.png', 0)
        stereo = cv2.StereoBM_create(numDisparities=160, blockSize=11)
        disparity = stereo.compute(imgL, imgR)
        fig = plt.figure(0)
        plt.imshow(disparity, 'gray')
        plt.axis('off')

        def onClick(event):
            x, y = event.xdata, event.ydata
            d = disparity[int(y), int(x)]
            depth = (178 * 2826)/d
            # print(d, depth)
            depth = int(depth)
            plt.text(2000, 1750, 'disparity: ' + str(d) + '\ndepth: ' + str(depth),
                     bbox=dict(fill=True, edgecolor='blue', linewidth=0))
            fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', onClick)
        plt.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    ui = MainUi()
    ui.show()
    sys.exit(app.exec_())
