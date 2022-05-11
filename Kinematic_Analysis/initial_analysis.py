import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#PATH = r'C:\Users\dvir.bens\Documents\DLC\dataAnaly\analysis_lib\3D_New\11-03-21\nana-try2-trail1_.avi_DLC_Dvir_3D.csv'   #path to the 3D csv file from DLC analysis
BODY_PART = 'finger'                                                                                         #body part that we want to analysis
FRAME_RATE = 120                                                                                             #number of frame rate belonging record videos software, it help us know the time difference between 2 coordinates data into excel.
THRESHOLD = 0.025
MOVEMENT_SENSITIVITY = 20

#PREPROCESSING
def preprocessing(PATH):
    body_cols = {'elbow': ['DLC_Dvir_3D', 'DLC_Dvir_3D.1', 'DLC_Dvir_3D.2'],                                     #dictionary that linking the body parts from the analysis to the colomns names
                'wrist': ['DLC_Dvir_3D.3', 'DLC_Dvir_3D.4', 'DLC_Dvir_3D.5'],
                'finger': ['DLC_Dvir_3D.6', 'DLC_Dvir_3D.7', 'DLC_Dvir_3D.8']}
    partial_data = pd.read_csv(PATH, header=0, usecols=body_cols[BODY_PART])                                     #reading the csv file (making the first raw as the names of the column(header=0)). Taking the colonms that have data to our specific body part( usecols=body_cols[BODY_PART] )
    #rename headers
    headers_names = ['{0}_x'.format(BODY_PART),'{0}_y'.format(BODY_PART),'{0}_z'.format(BODY_PART)]
    body_part_cols = [body_cols[BODY_PART][0], body_cols[BODY_PART][1], body_cols[BODY_PART][2]]
    d = dict(zip(body_part_cols,headers_names))
    partial_data = partial_data.rename(columns = d, inplace = False)                                            #replacing every time he see column name that is similar to the key in the dic d with his value, without changing permanently the column names in the original file (inplace=false)
    partial_data = partial_data.drop([0, 1])

    #convert argument to a float type
    partial_data['{0}_x'.format(BODY_PART)] = pd.to_numeric(partial_data['{0}_x'.format(BODY_PART)], downcast="float")
    partial_data['{0}_y'.format(BODY_PART)] = pd.to_numeric(partial_data['{0}_y'.format(BODY_PART)], downcast="float")
    partial_data['{0}_z'.format(BODY_PART)] = pd.to_numeric(partial_data['{0}_z'.format(BODY_PART)], downcast="float")

    #Save partial_data csv
    i = PATH.find('DLC_Dvir_3D')
    j = PATH.find('nana')
    New_PATH = PATH[:j] + 'Data_ana\\' + PATH[j:i] + 'Location_' + PATH[i + 9:]
    partial_data.to_csv (New_PATH)

    #creating diff data
    diff = partial_data.diff()
    i = PATH.find('DLC_Dvir_3D')
    j = PATH.find('nana')
    New_PATH = PATH[:j] + 'Data_ana\\' + PATH[j:i]  + 'Speed_' + PATH[i+4:]
    diff.to_csv (New_PATH)

    diff_x = diff['{0}_x'.format(BODY_PART)].plot()
    plt.show()
    diff_y = diff['{0}_y'.format(BODY_PART)].plot()
    plt.show()
    diff_z = diff['{0}_z'.format(BODY_PART)].plot()
    plt.show()

# L = []
# for i in range(1,4):
#     a = r'C:\Users\dvir.bens\Documents\DLC\dataAnaly\analysis_lib\3D\nana-try2-trail'
#     b = '_.avi_DLC_Dvir_3D.csv'
#     path = a+ str(i) + b
#     print(path)
#     L.append(path)
# print(L)

#preprocessing(r'C:\Users\dvir.bens\Documents\DLC\dataAnaly\analysis_lib\3D_New\11-03-21\nana-try2-trail1_.avi_DLC_Dvir_3D.csv')
#preprocessing(r'C:\Users\dvir.bens\Documents\DLC\dataAnaly\analysis_lib\3D_New\11-03-21\nana-try2-trail2_.avi_DLC_Dvir_3D.csv')
preprocessing(r'C:\Users\dvir.bens\Documents\DLC\dataAnaly\analysis_lib\3D\nana-try2-trail1_.avi_DLC_Dvir_3D.csv')







