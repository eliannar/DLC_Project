import numpy as np
import matplotlib.pyplot as plp
import pandas as pd
from mpldatacursor import datacursor

#לוודא שאני מוצא שלוש נקודות על המישור שהם טובות
p = np.array([4.91894012, 3.407223048,17.4638283]) #point of frame 400 from trail 2 of C:\Users\dvir.bens\Documents\DLC\dataAnaly\analysis_lib\3D
q = np.array([4.405137657,3.054596102,19.50006506])
d = np.array([4.706522428,0.800000239,19.34582274])
u = q-p
v = d-q


w2 = np.array([u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-v[0]*u[1]])
w = w2/np.linalg.norm(w2)
print(np.linalg.norm(w))
T = np.array([2,4,7,5,1,6,3,8])
n = 0
for i in [55,36,25,54,80,81,83,90]:
    Mov_point_data = pd.read_csv("C:/Users/dvir.bens/Documents/DLC/1304_3D/Data_ana/nana-nana_130421-trial{}_Location_3D.csv".format(i))  # reading the csv file
    L = np.zeros([len((Mov_point_data)),2])

    x = np.cross(w,v)/np.linalg.norm(np.cross(w,v))
    print(x)
    y = v/np.linalg.norm(v)

    for i in range(len(Mov_point_data)):
        p = np.array([Mov_point_data.loc[i,'finger_x'],Mov_point_data.loc[i,'finger_y'],Mov_point_data.loc[i,'finger_z']])
        proj_point = p-np.dot(p,w)*w
        L[i,:] = np.array([np.dot(x,p),np.dot(y,p)])

#    print(L)
    # plp.plot(L[:,0],L[:,1])
    # plp.show()

    # y_axis = L[23,:]-L[15,:]
    # #בשביל לקבל מערכת צירים של X ו Y שמתאימה עבור הראייה שלנו צריך לבחור שתי נקודות מתאימות
    # print(y_axis)
    # y_axis = y_axis/np.linalg.norm(y_axis)
    # thetha = -np.arccos(y_axis[1])
    # route_matrix = np.array(  [     [np.cos(thetha),np.sin(thetha)],  [-np.sin(thetha),np.cos((thetha))]       ]         )
    # print(route_matrix)
    # route_L = np.zeros([len((Mov_point_data)), 2])
    # for i in range(len(Mov_point_data)):
    #     route_L[i,:] = np.dot(route_matrix,L[i,:])
    # print(route_L[:,0])

#route_L = np.array([-route_L[:,0],route_L[:,1]])
#print(route_L)
#plp.plot(route_L[:,0],-route_L[:,1])
    plp.scatter(L[:, 0], L[:, 1], c=np.linspace(1, L.shape[0], L.shape[0]), cmap='Reds')
    plp.text(L[-1,0]+0.1, L[-1, 0]+0.1,s=str(T[n])+','+str(i),c='red')
    n = n + 1
    #ניסינו להדפיס את מספר המטרה על הגרף באמצעות TEXT עבור הנקודת האחרונה. משהו לא הסתדר
plp.show()






