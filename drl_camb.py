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
        ## ENVIRONMENT TO OPTIMISE AN AIRFOIL WITH DRL ##
        #################################################

## On fait bouger la cambrure avec le DRL ici

class drl_dussauge():

#--- CREATE OBJECT ---------------------------------------------------------------------------------------------------------+
    def __init__(self, path):
        self.name     = 'drl_dussauge'                                      # Fill structure
        self.act_size = 10                                                  # La cambrure à le droit de bouger
        self.obs_size = self.act_size
        self.obs      = np.zeros(self.obs_size)

    # Variable structure : une pour l'eading edge et les autres pour les points de contrôle : 
    # La première contrôle le bord d'attaque 
    # De 2 à 8 : contrôle les points 
    # De 9 à 10 si on fait varier la cambrure : jouent sur l'abscisse du point qui définit la cambrure 

        self.x_min       = np.array([0.005, 0, 0, 0, 0, -0.08, -0.1, -0.1,
         0.3, -0.1])
        self.x_max       = np.array([0.05, 0.07, 0.12, 0.07, 0.15, 0., 0., 0.,  # La cambrure peut bouger de -0.03 à +0.03
         0.7, 0.3]) 
        self.x_0         = np.array([                                           # Coord du naca 2412
            0.02340366,  0.08213517,  0.12281425,  0.12498064,
            0.08461117, -0.01325784,  0.00811609,  0.02124909, 
            0.68, +0.3])
        self.x_camb      = 0.3                                              # La cambrure, pour l'instant elle ne varie pas 
        self.y_camb      = 0.                                               # La cambrure, pour l'instant elle ne varie pas 
        self.area        = 0  
        self.path        = path
        self.finesse_moy = 0
        self.finesse_max = 0
        self.area_target = 0.08 
        self.area_min    = 0.1                                   
        self.angle       = 0.           
        self.alpha       = 200                                               # Le alpha dans la formule du reward                                                        
        self.episode     = 0                                                # Set episode number
        self.time_init   = 0.                                               # Pour connaître la durée d'un épisode
        self.time_end    = 0.


    #--- METHODES DE LA CLASSE ----------------------------------------------------------------------------------------------+

    def solve_problem_cimlib(self):
        """ Solve problem using cimlib and move vtu and drag folder. It changes properties."""

        os.system(
            'cd '+self.output_path+
            'cfd/.; touch run.lock; mpirun -n 8 /softs/cemef/cimlibxx/master/bin/cimlib_CFD_driver'+ 
            ' Principale.mtc > trash.txt;'
            )
        os.system('mv '+self.output_path+'cfd/Resultats/2d/* '+self.vtu_path+'.')
        os.system('mv '+self.output_path+'cfd/Resultats/Efforts.txt '+self.effort+'.')
        os.system('rm -r '+self.output_path+'cfd')
        os.system('cp -r '+self.vtu_path+'bulles_00400.vtu ./video/')                                  
        os.system('mv ./video/bulles_00400.vtu '+'./video/video_'+str(self.episode)+'.vtu')
        

    def shape_generation_dussauge(self, control_parameters):
        """ Generate shape using the parametrisation in Dessauge paper  modify to take camber in account """

        control_points = self.reconstruct_control_points(control_parameters)  # Transforme les actions en control point
        curve          = self.airfoil(control_points,16)                      # Donne la courbe de l'aile
        self.area      = self.polygon_area(curve)
        curve          = self.rotate(curve)                                   # Si on met un angle d'attaque
        
        ### On mesh le nouvel airfoil
        mesh_size      = 0.005                                                # Mesh size
        try:
            gmsh.initialize(sys.argv)                                         # Init GMSH
            gmsh.option.setNumber("General.Terminal", 1)                      # Ask GMSH to display information in the terminal
            model = gmsh.model                                                # Create a model and name it "shape"
            model.add("shape")        
            shapepoints = []
            for j in range(len(curve)):
                shapepoints.append(model.geo.addPoint(curve[j][0], curve[j][1], 0.0,mesh_size))
            shapepoints.append(shapepoints[0])
            shapespline = model.geo.addSpline(shapepoints)                    # Curveloop using splines
            model.geo.addCurveLoop([shapespline],1)
            model.geo.addPlaneSurface([1],1)                                  # Surface  

            ### This command is mandatory and synchronize CAD with GMSH Model. 
            ### The less you launch it, the better it is for performance purpose
            model.geo.synchronize()
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.0)                  # gmsh version 2.0
            model.mesh.generate(2)                                             # Mesh (2D)
            gmsh.write(self.output_path+'cfd/airfoil.msh')                     # Write on disk
            gmsh.finalize()                                                    # Finalize GMSH

        except Exception as e:
            ### Finalize GMSH
            gmsh.finalize()
            print('error: ', e)
            pass




    def compute_reward(self, control_parameters):
        """ Calcule le reward """
        try :
            with open(self.effort+'/Efforts.txt', 'r') as f:
                next(f)                                         # Skip header
                L_finesse    = [] 
                f.readline()
                for ligne in f :
                    cx, cy   = ligne.split()[-2:]
                    cx, cy   = -float(cx), -float(cy)
                    if cx*cy == 0.:
                        L_finesse.append(-100)                  # Si un des deux est nul on met un reward très faible  
                    else :
                        L_finesse.append(cy/cx)
                finesse = np.array(L_finesse)                   
            
        except :                                                # Si ça n'a pas marché 
            finesse = None
        

        #--- CALCUL DU REWARD ------------------------------------------------------------------------------------------+

        begin_take_finesse = 400                                       # When we begin to take the reward 

        if finesse is not None :  
            self.reward      = finesse[begin_take_finesse:].mean() 
            self.reward      -= self.punition_affine_marge(marge=0.1)  # Punition affine avec une marge de 10 % 
            self.finesse_moy = finesse[begin_take_finesse:].mean()
            self.finesse_max = finesse[begin_take_finesse:].max()

        else:                                                          # Si ça n'a pas tourné  
            self.reward      = 0
            self.finesse_moy = 0
            self.finesse_max = 0

        ### Ecriture dans Values
        print(os.path)
        if not os.path.isfile('Values.txt'):
            f = open('Values.txt','w')
            f.write(
                'Index'+'\t'+'edge'+'\t'+'1'+'\t'+'2'+'\t'+'3'+'\t'+'4'+'\t'+'5'+'\t'+'6'+
                '\t'+'7'+'\t'+'8'+'\t'+'9'+'\t'+'finesse_moy'+'\t'+'finesse_max'+'\t'+'Area'+'\t'+'Reward'+'\n'
                )
        else:
            f = open('Values.txt','a')
        f.write(
        str(self.episode)+'\t'+"{:.3e}".format(control_parameters[0])+'\t'+"{:.3e}".format(control_parameters[1])+'\t'
        +"{:.3e}".format(control_parameters[2])+'\t'+"{:.3e}".format(control_parameters[3])+'\t'
        +"{:.3e}".format(control_parameters[4])+'\t'+"{:.3e}".format(control_parameters[5])+'\t'
        +"{:.3e}".format(control_parameters[6])+'\t'+"{:.3e}".format(control_parameters[7])+'\t'
        +"{:.3e}".format(control_parameters[8])+'\t'+"{:.3e}".format(control_parameters[9])+'\t'
        +"{:.3e}".format(self.finesse_moy)+'\t'+"{:.3e}".format(self.finesse_max)+'\t'+"{:.3e}".format(self.area)+'\t'+
        "{:.3e}".format(self.reward)+'\n'
        )
        f.close()
        self.episode += 1 


    #--- CFD RESOLUTION -------------------------------------------------------------------------------------------------+

    def cfd_solve(self, x, ep):
        """ Return le reward : calcul l'airfoil, mesh, lance les simulations, calcul le reward """
        self.time_init=dt.datetime.now()                                        # On suit en temps le DRL
        if not os.path.isfile('temps_start.txt'):
            f = open('temps_start.txt','w')
            f.write('Index'+'\t'+'Heure start'+'\n')
            f.close()
        f = open('temps_start.txt','a')
        f.write(str(ep)+'\t'+ dt.datetime.now().strftime("%H:%M:%S")+'\n')
        f.close()

        ### Create folders and copy cfd (please kill me)
        ### On met les résultats là dedans 
        self.output_path = self.path+'/'+str(ep)+'/'  # Pour chaque épisode
        self.vtu_path    = self.output_path+'vtu/'
        self.effort      = self.output_path+'effort/'
        self.msh_path    = self.output_path+'msh/'
        self.t_mesh_path = self.output_path+'t_mesh/'
        
        os.makedirs(self.effort)
        os.makedirs(self.vtu_path)
        os.makedirs(self.msh_path)
        os.makedirs(self.t_mesh_path)
        os.system('cp -r cfd ' + self.output_path + '.')   
        
        ### Convert action to coordinates 
        # to_concatanate = np.array([self.x_camb, self.y_camb])                     ###### 1 enlever si cambrure bouge ######
        # control_parameters = np.concatenate((np.array(x), to_concatanate))        # On ajoute la cambrure qui est fixe
        #### enlever ça ou le toucher pour faire varier cambrure

        ### create the shape 
        self.shape_generation_dussauge(np.array(x))                        

        ### convert to .t
        os.system('cd '+self.output_path+'cfd ; python3 gmsh2mtc.py')
        os.system('cd '+self.output_path+'cfd ; cp -r airfoil.msh ../msh')
        os.system('cd '+self.output_path+'cfd ; module load cimlibxx/master')
        os.system('cd '+self.output_path+'cfd ; echo 0 | mtcexe airfoil.t')
        os.system('cd '+self.output_path+'cfd ; cp -r airfoil.t ../t_mesh')
        
        ### solving the problem
        if os.path.isfile(self.output_path+'cfd/airfoil.msh'):                     # Si le mesh a fonctionné
            self.solve_problem_cimlib()
            ### Compute the reward 
            self.compute_reward(np.array(x))
        else :
            self.reward      = 0
            self.finesse_moy = 0
            self.finesse_max = 0

        ### On écrit la durée
        self.time_end     = dt.datetime.now()
        difference        = self.time_end - self.time_init
        heures, reste     = divmod(difference.seconds, 3600)
        minutes, secondes = divmod(reste, 60)
        
        if not os.path.isfile('duree.txt'):
            f = open('duree.txt','w')
            f.write('Index'+'\t'+'Heure start'+'\t'+'Heure end'+'\t'+'Durée'+'\n')
            f.close()
        fi = open('duree.txt','a')
        fi.write(
            str(ep)+'\t'+ self.time_init.strftime("%H:%M:%S")+'\t'
            +self.time_end.strftime("%H:%M:%S")+'\t'
            +f"{str(heures)}:{str(minutes)}:{str(secondes)}"+'\n'
            )
        fi.close()
        return self.reward

                

    ### Take one step
    def step(self, actions, ep):
        conv_actions = self.convert_actions(actions)
        reward       = self.cfd_solve(conv_actions, ep)
        return reward, conv_actions

    ### Provide observation
    def observe(self):
        # Always return the same observation
        return self.obs

    ### Convert actions
    def convert_actions(self, actions):
        """ Converti les actions du DRL qui sont entre 0 et 1 """
        # Convert actions
        conv_actions  = self.act_size*[None]
        x_p           = self.x_max - self.x_0
        x_m           = self.x_0   - self.x_min

        for i in range(self.act_size):
            if (actions[i] >= 0.0):
                conv_actions[i] = self.x_0[i] + x_p[i]*actions[i]
            if (actions[i] <  0.0):
                conv_actions[i] = self.x_0[i] + x_m[i]*actions[i]
        return conv_actions

    ### Close environment
    def close(self):
        pass

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
        return curve

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
        return curve



    def rotate(self,curve):
        """ Met un angle d'attaque en multipliant la courbe par une matrice de rotation """
        curve         = np.array(curve)
        rotate_matrix = np.array([
            [np.cos(self.angle), np.sin(self.angle)], [-np.sin(self.angle), np.cos(self.angle)]
            ])
        return curve @ rotate_matrix

    def polygon_area(self,curve):
        """ Crée un polynôme avec la courbe et calcul son aire """
        curve      = np.array(curve)
        x          = curve [:,0]
        y          = curve[:,1]
        correction = x[-1] * y[0] - y[-1]* x[0]
        main_area  = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
        return 0.5*np.abs(main_area + correction)


#--- FONCTION DE PUNITION POUR LA SURFACE -------------------------------------------------------------------------------------+

    def punition_exponentielle(self):
        """ Donne la punition que l'on doit mettre dans le reward (exponentielle) """
        if self.area < self.area_min :
            return np.exp((self.area_min/self.area) -1) - 1       # vaut 0 au début 
        else : 
            return 0. 

    def punition_affine_marge(self, marge):
        """ Donne une punition affine de alpha * (S-Sref) avec marge de marge %"""
        if self.area_target * (1 - marge) < self.area < self.area_target * (1 + marge) : 
            return 0
        elif self.area < self.area_target * (1 - marge) : 
            return self.alpha * abs((1 - marge) * self.area_target - self.area)
        else : 
            return self.alpha * abs((1 + marge) * self.area_target - self.area)

    def punition_affine(self) : 
        """ Donne une punition de la forme alpha * abs(S-Sref) """
        return self.alpha * abs(self.area - self.area_target)
