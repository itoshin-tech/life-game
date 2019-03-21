import numpy as np
import cv2
import time

# parameters
FIELD_H = 100
FIELD_W = 180
BONE_RATIO = 0.2
SHOW_SCALE = 4

def field2img(field):
    img = field.astype(np.uint8)
    img = img * 255
    img = cv2.resize(img, (int(FIELD_W * SHOW_SCALE), 
                           int(FIELD_H * SHOW_SCALE)))
    return img

def add_flame(field):
    h, w = field.shape
    field2 = np.zeros((h + 2, w + 2), dtype = np.uint8)
    field2[1:-1, 1:-1] = field.copy()
    field2[0, :] = field2[-2,:]
    field2[-1,:] = field2[1, :]
    field2[:, 0] = field2[:,-2]
    field2[:,-1] = field2[:, 1]
    return field2

def imtext(img, msg, r, size, thickness, col, bgcol):
    cv2.putText(img, msg, r,
        cv2.FONT_HERSHEY_PLAIN, size, 
        bgcol, int(4*thickness), 1)
    cv2.putText(img, msg, r,
        cv2.FONT_HERSHEY_PLAIN, size, 
        col, thickness, 1)

def init_field():
    field = 1 * (np.random.rand(FIELD_H, FIELD_W) < BONE_RATIO)
    field = field.astype(np.uint8)
    return field

# main ------------
previous_t = 0
current_t = 0
cnt = 0
mode = 0
fps = 0
np.random.seed(0)
field = init_field()

while True:
    # fps
    current_t = time.perf_counter()
    cnt += 1
    if current_t - previous_t > 1.0: 
        dt = current_t - previous_t
        previous_t = current_t
        fps = cnt / dt 
        cnt = 0

    # update field
    if mode == 0:
        # normal mode
        field = add_flame(field)
        field_next = np.zeros((FIELD_H, FIELD_W), dtype=np.uint8)
        for i in range(1,FIELD_H+1):
            for j in range(1,FIELD_W+1):
                count = field[i-1,j-1]+field[i-1,j]+field[i-1,j+1] \
                    + field[i  ,j-1]             +field[i  ,j+1] \
                    + field[i+1,j-1]+field[i+1,j]+field[i+1,j+1]

                if field[i, j] == 1:
                    if count == 2 or count == 3:
                        field_next[i-1, j-1] = 1
                    else:
                        field_next[i-1, j-1] = 0
                else:
                    if count == 3:
                        field_next[i-1, j-1] = 1
                    else:
                        field_next[i-1, j-1] = 0
    else:
        # fast update
        field2 = add_flame(field)
        templ = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
        count = cv2.matchTemplate(field2, templ, cv2.TM_CCORR)
        count = np.round(count, 0)
        count = count.astype(np.uint8)
        alive = (field == 1) * ((count == 2) + (count == 3))
        born = (field == 0) * (count == 3)
        field_next = np.zeros_like(field, dtype=np.uint8)
        field_next[np.where(alive)] = 1
        field_next[np.where(born)] = 1
    
    field = field_next

    # showing image
    img = field2img(field)
    mode_name =['Normal', 'Fast']
    imtext(img, '%s mode' % mode_name[mode], (5, 40), 3, 4, 
           120, 255)
    imtext(img, 'fps: %.1f' % fps, (5, 90), 3, 4, 
           60, 255)
    cv2.imshow('img', img)
 
    # key control
    INPUT = cv2.waitKey(1) & 0xFF
    if INPUT == ord(' '): # mode change
        mode = 1- mode
    if INPUT == ord('r'): # initalize the field
        field = init_field()
    if INPUT == ord('q'): # quit
        cv2.destroyAllWindows()
        break
