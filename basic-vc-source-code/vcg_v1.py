import numpy as np
import cv2


def condi_inv(image, noise):
    layer = np.zeros(image.shape)
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            if image[i][j] == 255:
                layer[i][j] = noise[i][j]
            else:
                if noise[i][j] == 255:
                    layer[i][j] = 0
                else:
                    layer[i][j] = 255
    return layer


def main():
    # 輸入檔名以讀檔
    file = input('Input the file name: ')
    src = cv2.imread(file, 0)
    h, w = map(lambda l: l//2, src.shape)

    # 重新調整大小 (圖太大跑很久XD)
    image = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)

    # 建立噪圖 (隨機生成)
    noise = np.random.uniform(0, 255, image.shape)
    
    # 控制像素閥值 (令灰階圖層轉換成黑白圖層)
    th_args = (127, 255, cv2.THRESH_BINARY)
    _, imageLayer = cv2.threshold(image, *th_args)
    _, noiseLayer = cv2.threshold(noise, *th_args)

    # 利用原圖與噪圖 生成出 條件性反轉的密圖
    secretLayer = condi_inv(imageLayer, noiseLayer)

    # 存檔
    cv2.imwrite('secret_1.png', secretLayer)
    cv2.imwrite('secret_2.png', noiseLayer)

    # 驗證疊圖
    verific = np.zeros(image.shape)
    for i in range(h):
        for j in range(w):
            if secretLayer[i][j] and noiseLayer[i][j]:
                verific[i][j] = 255

    cv2.imwrite('verific.png', verific)


if __name__ == "__main__":
    main()
