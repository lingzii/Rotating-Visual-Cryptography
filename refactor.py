import numpy as np
import cv2

maskMode = [[([[1, 1], [1, 0]], [[1, 0], [1, 0]]),
            ([[1, 1], [1, 0]], [[1, 1], [0, 0]])],
            [([[1, 1], [1, 0]], [[0, 0], [1, 1]]),
            ([[1, 1], [1, 0]], [[0, 1], [0, 1]])]]


def pixilated(*imgs):
    th_args = (127, 255, cv2.THRESH_BINARY)
    result = []
    for img in imgs:
        tmp = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        _, tmp = cv2.threshold(tmp, *th_args)
        tmp = cv2.resize(tmp, (512, 512), interpolation=cv2.INTER_AREA)
        result.append(tmp)
    return result


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
    img1 = cv2.imread('p1.png', 0)
    img2 = cv2.imread('p2.png', 0)
    img1Layer, img2Layer = pixilated(img1, img2)

    h, w = img1Layer.shape
    center = ((h-1)/2, (w-1)/2)

    secret1Layer = np.zeros((h, w))
    secret2Layer = np.zeros((h, w))
    visitedTable = np.zeros((h, w))
    randomTable = np.random.uniform(0, 3, (h, w))

    for i in range(1, h, 2):
        for j in range(1, w, 2):

            if visitedTable[i][j]:
                continue

            for k, l in getRotPoints(center, (i, j)):

                values = [1 if img1Layer[k][l] else 0,
                          1 if img2Layer[k][l] else 0]
                mask1, mask2 = getMask(randomTable[i][j], *values)

                for m in range(2):
                    for n in range(2):
                        y, x = k+m-1, l+n-1
                        secret1Layer[y][x] = 0 if mask1[m][n] else 255
                        secret2Layer[y][x] = 0 if mask2[m][n] else 255

                visitedTable[k][l] = 1

    cv2.imwrite('secret1.png', secret1Layer)
    cv2.imwrite('secret2.png', secret2Layer)

    # 驗證疊圖
    secret3Layer = clockwise(secret1Layer.copy())
    verific1 = np.zeros((h, w))
    verific2 = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if secret1Layer[i][j] and secret2Layer[i][j]:
                verific1[i][j] = 255
            if secret2Layer[i][j] and secret3Layer[(i+1) % h][j]:
                verific2[i][j] = 255

    cv2.imwrite('verific1.png', verific1)
    cv2.imwrite('verific2.png', verific2)


if __name__ == "__main__":
    main()
