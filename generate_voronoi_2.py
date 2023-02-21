import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import pickle


from scipy.spatial import Voronoi, voronoi_plot_2d

image_path = './maps/TRUSCO3F_3_cleaned.png'
outside_path = './maps/TRUSCO3F_3_obstacles.png'
path = './maps/TRUSCO3F_3_inverse.png'
img_original = cv2.imread(image_path)
img = cv2.imread(path)

scale_percent = 25 # percent of original size
width = int(img_original.shape[1] * scale_percent / 100)
height = int(img_original.shape[0] * scale_percent / 100)
dim = (width, height)

img_original = cv2.resize(img_original, dim, interpolation=cv2.INTER_AREA)
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

## convert to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

marker = np.array([36, 28, 236]) #BGR equivalent for red


mask = cv2.inRange(img, marker, marker)
imask = mask>0
red = np.zeros_like(img, np.uint8)
red[imask] = img[imask]

img_gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
lower_threshold = 0
top_threshold = 10
ret, thresh = cv2.threshold(img_gray, top_threshold, 255, cv2.THRESH_BINARY)


# cv2.imshow("Polygon", thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

Y, X = np.where(thresh == 0)
points = np.array([X, Y]).T
print(points)
vor = Voronoi(points)
polygon_matrix = thresh == 0
print(polygon_matrix.shape)
# print(polygon_matrix[int(vor.vertices[128][1])][int(vor.vertices[128][0])])
print(polygon_matrix.shape)
print(thresh.shape)
print(len(vor.ridge_vertices))
finite_ridges = []
for ridge in vor.ridge_vertices:
    if ridge[0] == -1 or ridge[1] == -1:
        pass
    else: 
        finite_ridges.append(ridge)

print(len(finite_ridges))

filtered_ridges = []

# print(vor.vertices)

for ridge in finite_ridges:
    v1 = vor.vertices[ridge[0]]
    v2 = vor.vertices[ridge[1]]

    try:

        if (polygon_matrix[int(v1[1])][int(v1[0])] == False) and (polygon_matrix[int(v2[1])][int(v2[0])] == False):
            filtered_ridges.append(ridge)
        else:
            pass  
    except:
        pass


filtered_vertices = []
nogood_vertices = []
fucked_vertices = []

for vertex in vor.vertices:
    x = vertex[0]
    y = vertex[1]
    # print(vertex)

    try:
        if (polygon_matrix[int(y)][int(x)] == False):
            filtered_vertices.append(vertex)
        else:
            nogood_vertices.append(vertex)
    except Exception as e:
        # print(e)
        fucked_vertices.append(vertex)


# filtering vertices and edges
# vor.vertices = np.array(filtered_vertices)
z = np.array(filtered_vertices)
vor.ridge_vertices = np.array(filtered_ridges)


with open('voronoi.pkl', 'wb') as f:
    pickle.dump(vor, f, pickle.HIGHEST_PROTOCOL)
    print('pickled')

with open('filtered_vertices.pkl', 'wb') as f:
    pickle.dump(z, f, pickle.HIGHEST_PROTOCOL)
    print('pickled 2')

print('plotting...')

# fig = voronoi_plot_2d(vor, line_width=0.5, point_size=0.4, show_vertices=False)
# plt.scatter(z[:, 0], z[:, 1], s=0.2)

plt.imshow(img_original)
plt.show()




