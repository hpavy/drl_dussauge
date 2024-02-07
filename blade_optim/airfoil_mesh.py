#--- GENERIC IMPORT ---------------------------------------------------------------------------------------------------------+
import os
import sys
import math
import time
import numpy as np
import gmsh
import matplotlib as plt
import datetime as dt


        #################################################
        ## ENVIRONMENT TO PLOT AN AIRFOIL              ##
        #################################################

## On construit les différentes aires ici 

class airfoil():

#--- CREATE OBJECT ---------------------------------------------------------------------------------------------------------+
    def __init__(self, path, geo, corde, index, angle):
    # Variable structure : une pour l'eading edge et les autres pour les points de contrôle : 
    # La première contrôle le bord d'attaque 
    # De 2 à 8 : contrôle les points 
    # De 9 à 10 si on fait varier la cambrure : jouent sur l'abscisse du point qui définit la cambrure 

        self.x_0         = geo 
        
        #np.array([ 0.02359928,  0.04828674,  0.07590444,  0.08340451,  0.05728329,
        #-0.04362299, -0.03815534, -0.02214095,  0.7004262 , -0.00406199])
                                             # La cambrure, pour l'instant elle ne varie pas 
        self.area        = 0  
        self.path        = path                                 
        self.angle       = angle   
        self.corde       = corde
        self.index       = index 


    #--- METHODES DE LA CLASSE ----------------------------------------------------------------------------------------------+

        

    def shape_generation_dussauge(self, control_parameters):
        """ Generate shape using the parametrisation in Dessauge paper  modify to take camber in account """

        control_points = self.reconstruct_control_points(control_parameters)  # Transforme les actions en control point
        curve          = self.airfoil(control_points,16)                      # Donne la courbe de l'aile
        self.area      = self.polygon_area(curve)
        curve          = self.rotate(curve)                                   # Si on met un angle d'attaque
        curve          = self.translate_curve(curve, 1, 1)

        
        ### On mesh le nouvel airfoil
        mesh_size      = 0.001 / 4                                               # Mesh size
        try:
            gmsh.initialize(sys.argv)                                         # Init GMSH
            gmsh.option.setNumber("General.Terminal", 1)                      # Ask GMSH to display information in the terminal
            model = gmsh.model                                                # Create a model and name it "shape"
            model.add("shape") 
            lc = 0.3 / 4

            # Définir les points du rectangle
            point1 = model.geo.addPoint(0, 0, 0, lc)
            point2 = model.geo.addPoint(3.5, 0, 0, lc)
            point3 = model.geo.addPoint(3.5, 2, 0, lc)
            point4 = model.geo.addPoint(0, 2, 0, lc)

            # Définir les lignes du rectangle
            line1 = model.geo.addLine(point1, point2)
            line2 = model.geo.addLine(point2, point3)
            line3 = model.geo.addLine(point3, point4)
            line4 = model.geo.addLine(point4, point1)  

            shapepoints = []
            for j in range(len(curve)):
                shapepoints.append(model.geo.addPoint(curve[j][0], curve[j][1], 0.0,mesh_size))
            shapepoints.append(shapepoints[0])
            shapespline = model.geo.addSpline(shapepoints)                 # Curveloop using splines
            
            contour =[]
            contour.append(line1)
            contour.append(line2)
            contour.append(line3)
            contour.append(line4)
            contour.append(shapespline)
            print(f"contour{contour}")
            
            model.geo.addCurveLoop([line1, line2, line3, line4, -shapespline])
            model.geo.addPlaneSurface([1],1)                                  # Surface  

            ### This command is mandatory and synchronize CAD with GMSH Model. 
            ### The less you launch it, the better it is for performance purpose
            model.geo.synchronize()
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.0)                  # gmsh version 2.0
            model.mesh.generate(2)                                             # Mesh (2D)
            gmsh.write(self.output_path+'/'+f'maillage{self.index}.msh')                     # Write on disk
            gmsh.finalize()                                                    # Finalize GMSH

        except Exception as e:
            ### Finalize GMSH
            gmsh.finalize()
            print('error: ', e)
            pass





    #--- CFD RESOLUTION -------------------------------------------------------------------------------------------------+

    def write_mesh(self, name):
        """ Return le reward : calcul l'airfoil, mesh, lance les simulations, calcul le reward """

        ### Create folders and copy cfd (please kill me)
        ### On met les résultats là dedans 
        self.output_path = self.path  #+'/'+str(name)  # Pour chaque épisode


        ### create the shape 
        self.shape_generation_dussauge(np.array(self.x_0))                        
        print('done')

                
    ### A function to replace text in files
    ### This function finds line containing string, erases the
    ### whole line it and replaces it with target
    def line_replace(self, string, line, target):
        command = "sed -i '/"+string+"/c\\"+line+"' "+target
        os.system(command)


#--- FONCTION DE PARAMETRISATION ----------------------------------------------------------------------------------------------+

    def quadraticBezier(self,t,points):
        B_x = (1-t)*((1-t)*points[0][0]+t*points[1][0])+t*((1-t)*points[1][0]+t*points[2][0])
        B_y = (1-t)*((1-t)*points[0][1]+t*points[1][1])+t*((1-t)*points[1][1]+t*points[2][1])
        return B_x,B_y

    def cambrure(self, x, y, numPts):
        """ Donne la cambrure avec le point qui la contrôle """ 
        curve   = []
        t       = np.array([i*1/numPts for i in range(0,numPts)])
        B_x,B_y = self.quadraticBezier(t,[(0.,0.),(x,y),(1.,0.)])
        curve   = list(zip(B_x,B_y))
        return np.array(curve)

    def find_camber_y(self, x, cambrure_coord):
        """ Pour un x donné il donne le y de la cambrure le plus proche """
        try :
            for k,coord_camber in enumerate(cambrure_coord):
                if coord_camber[0] > x :
                    return (coord_camber[1]+cambrure_coord[k-1][1])/2                      # On prend la moyenne des deux 
        except :
            return 0.

    def reconstruct_control_points(self, control_parameter):
        ### Les points autour desquels on bouge
        if len(control_parameter) == 10 : 
            x_param_cambrure, y_param_cambrure =  control_parameter[-2:]               # Les deux points definissant la cambrure 
        else :
            x_param_cambrure, y_param_cambrure =  0., 0.                               # Si on n'optimise pas avec la cambrure

        cambrure_coord = self.cambrure(x_param_cambrure, y_param_cambrure,16*40)
        base_points    =[[1,0.001],                                                    # Trailing edge (top)
                        [0.76,None],
                        [0.52,None],
                        [0.25,None],
                        [0.1,None],
                        [0,None],                                                      # Leading edge (top)
                        [0,None],                                                      # Leading edge (bottom)
                        [0.15,None],
                        [0.37,None],
                        [0.69,None],
                        [1,-0.001]] 

        ### Construction des control points pour génerer la courbe 
        control_points             = base_points[::]                                     
        control_points[5][1]       =  control_parameter[0] 
        control_points[5][1]       += self.find_camber_y(control_points[5][0], cambrure_coord)
        control_points[6][1]       = - control_parameter[0] 
        control_points[6][1]       += self.find_camber_y(control_points[6][0], cambrure_coord)
        for k in range(4):
            control_points[k+1][1] =  control_parameter[1+k] 
            control_points[k+1][1] += self.find_camber_y(control_points[k+1][0], cambrure_coord)
        for k in range(3):
            control_points[k+7][1] =  control_parameter[5+k] 
            control_points[k+7][1] +=self.find_camber_y(control_points[k+7][0], cambrure_coord)
        return control_points

    def airfoil(self,ctlPts,numPts):
        """ Crée la courbe de l'airfoil avec numPts nombre de points """
        curve = []
        t     = np.array([i*1/numPts for i in range(0,numPts)])
        
        ### Calculate first Bezier curve
        midX       = (ctlPts[1][0]+ctlPts[2][0])/2
        midY       = (ctlPts[1][1]+ctlPts[2][1])/2
        B_x,B_y    = self.quadraticBezier(t,[ctlPts[0],ctlPts[1],[midX,midY]])
        curve      = curve+list(zip(B_x,B_y))

        ### Calculate middle Bezier Curves
        for i in range(1,len(ctlPts)-3):
            midX_1  = (ctlPts[i][0]+ctlPts[i+1][0])/2
            midY_1  = (ctlPts[i][1]+ctlPts[i+1][1])/2
            midX_2  = (ctlPts[i+1][0]+ctlPts[i+2][0])/2
            midY_2  = (ctlPts[i+1][1]+ctlPts[i+2][1])/2
            B_x,B_y = self.quadraticBezier(t,[[midX_1,midY_1],ctlPts[i+1],[midX_2,midY_2]])
            curve   = curve+list(zip(B_x,B_y))                     
    
        ### Calculate last Bezier curve
        midX    = (ctlPts[-3][0]+ctlPts[-2][0])/2
        midY    = (ctlPts[-3][1]+ctlPts[-2][1])/2
        B_x,B_y = self.quadraticBezier(t,[[midX,midY],ctlPts[-2],ctlPts[-1]])
        curve   = curve+list(zip(B_x,B_y))
        curve.append(ctlPts[-1])
        return np.array(curve) * self.corde



    def rotate(self,curve):
        """ Met un angle d'attaque en multipliant la courbe par une matrice de rotation """
        curve         = curve
        rotate_matrix = np.array([
            [np.cos(self.angle), np.sin(self.angle)], [-np.sin(self.angle), np.cos(self.angle)]
            ])
        return curve @ rotate_matrix

    def polygon_area(self,curve):
        """ Crée un polynôme avec la courbe et calcul son aire """
        curve      = curve
        x          = curve [:,0]
        y          = curve[:,1]
        correction = x[-1] * y[0] - y[-1]* x[0]
        main_area  = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
        return 0.5*np.abs(main_area + correction)

    def translate_curve(self, curve, x_translate, y_translate):
        translate_curve = curve
        translate_curve[:,0] += x_translate
        translate_curve[:,1] += y_translate
        return list(translate_curve)

