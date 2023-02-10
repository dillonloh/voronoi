import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath

from shapely import Polygon, MultiPolygon, Point
from scipy.spatial import Voronoi, voronoi_plot_2d

image_path = './maps/TRUSCO3F_3_cleaned.png'
outside_path = './maps/TRUSCO3F_3_obstacles.png'
img_original = cv2.imread(image_path)
img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

marker = [36, 28, 236] #BGR equivalent for red
lower_threshold = 0
top_threshold = 150


# # finding points that are outside of the warehouse (in red)
# img_outside = cv2.imread(outside_path)
# X_out, Y_out = np.where(np.all(img_outside==marker, axis=2))
# zipped = np.column_stack((X_out,Y_out)).tolist()

# ### finding polygons

# ret,thresh = cv2.threshold(img_gray, top_threshold, 255, cv2.THRESH_BINARY)

# # find the contours
# contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# print("Number of contours detected:", len(contours))

# # polygons = []
# for cnt in contours:
#     cnt = np.squeeze(cnt)
#     if cnt.shape[0] >= 4 and cnt.shape[0] <= 50:
#         polygons.append(Polygon(cnt))

# print(len(polygons))

# multipolygon = MultiPolygon(polygons)

# Get X and Y coordinates of all obstacle pixels
Y, X = np.where((img_gray <= top_threshold) & (img_gray >= lower_threshold))

points = np.array([X, Y]).T

vor = Voronoi(points)

# bad_points = [] # indices of vertices in obstacles
# for i in range(len(vor.vertices)):
#     vertex = vor.vertices[i]
#     print(i, '/', len(vor.vertices))
#     if any((vertex == x).all() for x in zipped):
#         bad_points.append(i)

# print(len(bad_points))

# filtered_ridges = []
# for vertex in vor.ridge_vertices:
#     if vertex[0] == -1 or vertex[1] == -1: # check for ridges with infinite vertex
#         pass
#     else:
#         filtered_ridges.append(vertex)


# final_ridges = []
# for ridge in filtered_ridges:
#     if any((ridge[0] == x).all() for x in bad_points) or any((ridge[1] == x).all() for x in bad_points):
#         pass
#     else:    
#         final_ridges.append(ridge)




# #     for polygon in polygons:
# #         if multipolygon.contains(Point(vor.vertices[i])):
# #             bad_points.append(i)
# #             break
# # print(bad_points)
# # final_ridges = []
# # for ridge in filtered_ridges:
# #     if ridge[0] not in bad_points or ridge[1] not in bad_points:
# #         final_ridges.append(ridge)
# # print(np.array(filtered_ridges).shape)
# print(np.array(final_ridges).shape)

# for ridges in filtered_ridges

# vor.ridge_vertices = np.array(filtered_ridges)
# for vertex in vor.vertices:
#     if multipolygon.contains(vertex):
#         print('FUABCIA')
#     else:
#         print('OKAY BITCH')
fig = voronoi_plot_2d(vor, line_width=0.1, point_size=0.4, show_vertices=False)


plt.imshow(img_original)
plt.show()



# cv2.imshow("Polygon", thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

