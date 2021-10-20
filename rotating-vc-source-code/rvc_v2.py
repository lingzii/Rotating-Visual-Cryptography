import numpy as np
import cv2

maskMode = [[([[1, 1], [1, 0]], [[1, 0], [1, 0]]),
            ([[1, 1], [1, 0]], [[1, 1], [0, 0]])],
            [([[1, 1], [1, 0]], [[0, 0], [1, 1]]),
            ([[1, 1], [1, 0]], [[0, 1], [0, 1]])]]


def pixilated(img):
    tmp = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    _, tmp = cv2.threshold(tmp, 127, 255, cv2.THRESH_BINARY)
    tmp = cv2.resize(tmp, (512, 512), interpolation=cv2.INTER_AREA)
    return tmp


def getRotPoints(center, point):
    result = []
    for _ in range(4):
        vector = point[1]-center[1], center[0]-point[0]
        point = center[0]+vector[0], center[1]+vector[1]
        result.append(tuple(map(int, point)))
    return result


def getMask(n, i, j):
    rot = [(0, 0), (0, 1), (1, 1), (1, 0)]
    masks = []
    for src in maskMode[i][j]:
        mask = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                idx = rot.index((i, j))
                k, l = rot[(idx+int(n)) % 4]
                mask[k][l] = src[i][j]
        masks.append(mask)
    return masks


def clockwise(m):
    n = m.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            m[i][j], m[j][i] = m[j][i], m[i][j]
    for i in range(n//2):
        for j in range(n):
            m[i][j], m[n-1-i][j] = m[n-1-i][j], m[i][j]
    return m


def main():
    img1Layer = pixilated(cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE))
    img2Layer = pixilated(cv2.imread('image2.png', cv2.IMREAD_GRAYSCALE))

    h, w = map(lambda l: l//2, img1Layer.shape)
    center = ((h-1)/2, (w-1)/2)

    share1Layer = np.zeros(img1Layer.shape)
    share2Layer = np.zeros(img1Layer.shape)
    randomTable = np.random.uniform(0, 3, (h//2)**2)
    rd_idx = 0

    for i in range(h//2):
        for j in range(i, w-i-1):
            randomMode = randomTable[rd_idx]

            for k, l in getRotPoints(center, (i, j)):

                pixel_i, pixel_j = k*2, l*2
                idxes = [1 if img1Layer[pixel_i][pixel_j] else 0,
                         1 if img2Layer[pixel_i][pixel_j] else 0]
                mask1, mask2 = getMask(randomMode, *idxes)

                for m in range(2):
                    for n in range(2):
                        y, x = pixel_i+m, pixel_j+n
                        share1Layer[y][x] = 0 if mask1[m][n] else 255
                        share2Layer[y][x] = 0 if mask2[m][n] else 255

            rd_idx += 1

    cv2.imwrite('share1.png', share1Layer)
    cv2.imwrite('share2.png', share2Layer)

    share3Layer = clockwise(share1Layer.copy())
    verific1 = np.zeros(share3Layer.shape)
    verific2 = np.zeros(share3Layer.shape)
    for i in range(h*2):
        for j in range(w*2):
            if share1Layer[i][j] and share2Layer[i][j]:
                verific1[i][j] = 255
            if share2Layer[i][j] and share3Layer[i][j]:
                verific2[i][j] = 255

    cv2.imwrite('verific1.png', verific1)
    cv2.imwrite('verific2.png', verific2)


if __name__ == '__main__':
    main()
