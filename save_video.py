import numpy as np
import matplotlib.pyplot as plt
import imageio
import pandas as pd
import os as os


def quadraticBezier(t,points):
    B_x=(1-t)*((1-t)*points[0][0]+t*points[1][0])+t*((1-t)*points[1][0]+t*points[2][0])
    B_y=(1-t)*((1-t)*points[0][1]+t*points[1][1])+t*((1-t)*points[1][1]+t*points[2][1])
    return B_x,B_y

def cambrure(x,y, numPts):
    curve=[]
    t=np.array([i*1/numPts for i in range(0,numPts)])
    
    # calculate first Bezier curve
    B_x,B_y=quadraticBezier(t,[(0.,0.),(x,y),(1.,0.)])
    curve=list(zip(B_x,B_y))
    return curve

def find_camber_y(x, cambrure_coord):
    # pour un x donné il donne le y de la cambrure le plus proche
    for k,coord_camber in enumerate(cambrure_coord):
        if coord_camber[0] > x :
            return (coord_camber[1]+cambrure_coord[k-1][1])/2 # on prend la moyenne des deux 
    return 0.

def reconstruct_control_points(control_parameter):
    # Les points autour desquels on bouge
    x_param_cambrure, y_param_cambrure = control_parameter[-2:]  # les deux points definissant la cambrure 
    cambrure_coord = cambrure(x_param_cambrure, y_param_cambrure,16*40)
    base_points =[[1,0.001],              # trailing edge (top)
        [0.76,None],
        [0.52,None],
        [0.25,None],
        [0.1,None],
        [0,None],               # leading edge (top)
        [0,None],              # leading edge (bottom)
        [0.15,None],
        [0.37,None],
        [0.69,None],
        [1,-0.001]] 


    control_points = base_points[::] # les nouveaux control points on va construire avec le control_parameter 
    control_points[5][1] = control_parameter[0] + find_camber_y(control_points[5][0], cambrure_coord)
    control_points[6][1] = -control_parameter[0] + find_camber_y(control_points[6][0], cambrure_coord)
    for k in range(4):
        control_points[k+1][1] = control_parameter[1+k] + find_camber_y(control_points[k+1][0], cambrure_coord)

    for k in range(3):
        control_points[k+7][1] = control_parameter[5+k] +find_camber_y(control_points[k+7][0], cambrure_coord)
    return control_points


def airfoil1(ctlPts,numPts):
    curve=[]
    t=np.array([i*1/numPts for i in range(0,numPts)])
    
    # calculate first Bezier curve
    midX=(ctlPts[1][0]+ctlPts[2][0])/2
    midY=(ctlPts[1][1]+ctlPts[2][1])/2
    B_x,B_y=quadraticBezier(t,[ctlPts[0],ctlPts[1],[midX,midY]])
    curve=curve+list(zip(B_x,B_y))

    # calculate middle Bezier Curves
    for i in range(1,len(ctlPts)-3):
        midX_1=(ctlPts[i][0]+ctlPts[i+1][0])/2
        midY_1=(ctlPts[i][1]+ctlPts[i+1][1])/2
        midX_2=(ctlPts[i+1][0]+ctlPts[i+2][0])/2
        midY_2=(ctlPts[i+1][1]+ctlPts[i+2][1])/2

        B_x,B_y=quadraticBezier(t,[[midX_1,midY_1],ctlPts[i+1],[midX_2,midY_2]])
        curve=curve+list(zip(B_x,B_y))                     
   
    # calculate last Bezier curve
    midX=(ctlPts[-3][0]+ctlPts[-2][0])/2
    midY=(ctlPts[-3][1]+ctlPts[-2][1])/2

    B_x,B_y=quadraticBezier(t,[[midX,midY],ctlPts[-2],ctlPts[-1]])
    curve=curve+list(zip(B_x,B_y))
    curve.append(ctlPts[-1])
    return curve



def airfoil_plot(points, curve, title, control_parameter, file_save=None, xlim=None, ylim=None, ep=-1, finesse=None):
    xPts,yPts=list(zip(*points))
    xPts2,yPts2=list(zip(*curve))
    plt.plot(xPts,yPts,color='#E1A4A4')
    plt.plot(xPts2,yPts2,'b')
    plt.fill(xPts2,yPts2, color='b', alpha=0.5, label='Surface fermée')
    plt.plot(xPts,yPts,'o',mfc='none',mec='r',markersize=5)
    if finesse is not None:
        plt.text(0.75, 0.12, f'Episode = {ep}\nFinesse = {finesse}', bbox=dict(facecolor='white', alpha=0.8))



    x,y = control_parameter[-2:]
    cambrure_coord=cambrure(x,y,16*40)
    cambrure_coord = np.array(cambrure_coord)
    #plt.plot(cambrure_coord[:,0], cambrure_coord[:,1],'--', color='g', lw=1).    # decomment to have it
    #plt.plot((0,x,1),(0,y,0),'o',mfc='none',mec='g',markersize=5)

    plt.title(title)
    if xlim is not None :
        plt.xlim(xlim)
    else:
        plt.xlim(-0.05,1.05)

    if ylim is not None:
        plt.ylim(ylim)
    else: 
        plt.ylim(-0.55,0.55)
    plt.grid()

    if file_save is None:
        plt.show()
    else : 
        plt.savefig(file_save)
        plt.clf()


def plot_video(file, title, save_folder, title_video,ylim=(-0.05,0.15), cdt_init=None, title_init="", finesse_init=""):
    img= []
    L_control = []
    L_parameter = []
    img_files = []

    if cdt_init is not None:
        L_control.append(reconstruct_control_points(np.concatenate((cdt_init,np.zeros(2)))))
        img.append(airfoil1(L_control[0],16))
        L_parameter.append(np.concatenate((cdt_init,np.zeros(2))))
        airfoil_plot(L_control[0], img[0], title, L_parameter[0], file_save=save_folder+f'/video0',ylim=ylim, ep=title_init, finesse = finesse_init)
        img_files.append(save_folder +f'/video0.png')

    df = pd.read_csv(file + "/Values.txt", sep="\t", header=0)
    for i in range(df['Index'].max()):
        df_max = df.groupby(df['Index'])['Reward'].max()
        coeffs = df[["edge", "1","2","3","4","5","6","7"]][df['Reward']== df_max[i]].to_numpy()
        L_control.append(reconstruct_control_points(np.concatenate((coeffs[0],np.zeros(2)))))
        img.append(airfoil1(L_control[i],16))
        L_parameter.append(np.concatenate((coeffs[0],np.zeros(2))))
        airfoil_plot(L_control[i], img[i], title, L_parameter[i], file_save=save_folder+f'/video{i+1}',ylim=ylim, ep=i+1, finesse = round(df_max[i],3))

        img_files.append(save_folder +f'/video{i+1}.png')

    writer = imageio.get_writer(save_folder + '/' + title_video, fps=3)
    for im in img_files:
        writer.append_data(imageio.imread(im))
    writer.close()

    for image_file in img_files:
        os.remove(image_file)
