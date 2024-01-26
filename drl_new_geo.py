# Generic imports
import os
import sys
import math
import time
import numpy as np
import gmsh
import matplotlib as plt
import datetime as dt

################################
### Environment for Wind Turbine 
class drl_dussauge():

    ### Create object
    def __init__(self, path):

        # Fill structure
        self.name     = 'drl_dussauge'
        #self.act_size = 4
        self.act_size = 8
        self.obs_size = self.act_size
        self.obs      = np.zeros(self.obs_size)
        # Variable : une pour l'eading edge et les autres pour les points de contrôle
        self.x_min    =  np.array([0.01, 0, 0, 0, 0, -0.3, -0.3, -0.3]) 
        self.x_max    =  np.array([0.1, 0.3, 0.3, 0.4, 0.4, 0., 0., 0.]) 
        self.x_0 = np.array( [0.03, 0.08, 0.125, 0.12, 0.08, -0.08, -0.1,-0.08]) * 0.53 # l'aile symétrique pour avoir une surface de 0.8
        self.x_camb = 0.3    # la cambrure, pour l'instant elle ne varie pas 
        self.y_camb = 0.    # la cambrure, pour l'instant elle ne varie pas 
        self.area = 0  # on va écrire les aires ici
        self.path     = path
        self.finesse_moy= 0
        self.finesse_max= 0
        self.area_target = 0.08 # La surface cible
        self.area_min = 0.1                                      ########### Ici on met la surface minimale autorisée
        self.angle = 0.  #inclinaison en radian            ########### Super important, regarder comment le prendre en compte
        self.alpha = 80  # pour avoir une surface de ref


        # Set episode number
        self.episode  = 0
        # Pour savoir la durée d'un épisode
        self.time_init = 0.
        self.time_end = 0.


    def solve_problem_cimlib(self):
            # Solve problem using cimlib and move vtu and drag folder
        try :
            os.system('cd '+self.output_path+'cfd/.; touch run.lock; mpirun -n 8 /softs/cemef/cimlibxx/master/bin/cimlib_CFD_driver Principale.mtc > trash.txt;')
            os.system('mv '+self.output_path+'cfd/Resultats/2d/* '+self.vtu_path+'.')
            os.system('mv '+self.output_path+'cfd/Resultats/Efforts.txt '+self.effort+'.')
            os.system('rm -r '+self.output_path+'cfd')
            # Save
            os.system('cp -r '+self.vtu_path+'bulles_00800.vtu ./video/')
            os.system('mv ./video/bulles_00800.vtu '+'./video/video_'+str(self.episode)+'.vtu')
        
        except : 
            print("La simulation ne s'est pas lancée ") ############ A enlever !!! Juste pour debugger 



    

    def shape_generation_dussauge(self, control_parameters):
        control_points = self.reconstruct_control_points(control_parameters)
        curve = self.airfoil(control_points,16)
        self.area = self.polygon_area(curve)
        curve = self.rotate(curve)
        mesh_size = 0.005 # Mesh size
        try:
            # Init GMSH
            gmsh.initialize(sys.argv)
            # Ask GMSH to display information in the terminal
            gmsh.option.setNumber("General.Terminal", 1)

            # Create a model and name it "shape"
            model = gmsh.model
            model.add("shape")        
            
            shapepoints = []
            for j in range(len(curve)):
                shapepoints.append(model.geo.addPoint(curve[j][0], curve[j][1], 0.0,mesh_size))
            shapepoints.append(shapepoints[0])

            

            # Curveloop using splines
            shapespline = model.geo.addSpline(shapepoints)
            model.geo.addCurveLoop([shapespline],1)
            
            

            # Surface  
            model.geo.addPlaneSurface([1],1)
            


            # This command is mandatory and synchronize CAD with GMSH Model. The less you launch it, the better it is for performance purpose
            model.geo.synchronize()

            # gmsh version 2.0
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.0)

            # Mesh (2D)
            model.mesh.generate(2)
            # Write on disk
            gmsh.write(self.output_path+'cfd/airfoil.msh')

            # Finalize GMSH
            gmsh.finalize()

        except Exception as e:
            # Finalize GMSH
            gmsh.finalize()
            print('error: ', e)
            pass




    def compute_reward(self, control_parameters):
        # Compute reward
        try :
            with open(self.effort+'/Efforts.txt', 'r') as f:
                next(f) # Skip header
                L_finesse = [] 
                f.readline()
                for ligne in f :
                    cx, cy = ligne.split()[-2:]
                    cx, cy = -float(cx), -float(cy)
                    if cx*cy == 0.:
                        L_finesse.append(-100)  # On a quelque chose de ridicule comme ça   
                    else :
                        L_finesse.append(cy/cx)
                finesse = np.array(L_finesse)  # 
            
        except : # Si ça n'a pas marché 
            finesse = None

        begin_take_reward = 1200  # When we begin to take the reward 


        # Compute new reward
        if finesse is not None :  
            self.reward = finesse[begin_take_reward:].mean() - self.alpha * abs(self.area - self.area_target)   # - self.punition()  On ne punnit plus 
            self.finesse_moy = finesse[begin_take_reward:].mean()
            self.finesse_max = finesse[begin_take_reward:].max()
        else: # Si ça n'a pas tourné  
            self.reward = -1000
            self.finesse_moy = -1000
            self.finesse_max = -1000

        
        # On écrit dans le gros fichier
        print(os.path)
        if not os.path.isfile('Values.txt'):
            f = open('Values.txt','w')
            f.write(
                'Index'+'\t'+'edge'+'\t'+'1'+'\t'+'2'+'\t'+'3'+'\t'+'4'+'\t'+'5'+'\t'+'6'+
                '\t'+'7'+'\t'+'finesse_moy'+'\t'+'finesse_max'+'\t'+'Area'+'\t'+'Reward'+'\n'
                )
        else:
            f = open('Values.txt','a')
        f.write(
        str(self.episode)+'\t'+"{:.3e}".format(control_parameters[0])+'\t'+"{:.3e}".format(control_parameters[1])+'\t'
        +"{:.3e}".format(control_parameters[2])+'\t'+"{:.3e}".format(control_parameters[3])+'\t'+"{:.3e}".format(control_parameters[4])+'\t'
        +"{:.3e}".format(control_parameters[5])+'\t'+"{:.3e}".format(control_parameters[6])+'\t'+"{:.3e}".format(control_parameters[7])+'\t'
        +"{:.3e}".format(self.finesse_moy)+'\t'+"{:.3e}".format(self.finesse_max)+'\t'+"{:.3e}".format(self.area)+'\t'+"{:.3e}".format(self.reward)+'\n'
        )
        f.close()
		
        self.episode      += 1 #new




    ### CFD resolution
    def cfd_solve(self, x, ep):
        # on écrit où nous en sommes 
        self.time_init=dt.datetime.now()
        if not os.path.isfile('temps_start.txt'):
            f = open('temps_start.txt','w')
            f.write('Index'+'\t'+'Heure start'+'\n')
            f.close()

        f = open('temps_start.txt','a')
        f.write(str(ep)+'\t'+ dt.datetime.now().strftime("%H:%M:%S")+'\n')
        f.close()

        # Create folders and copy cfd (please kill me)
        # On met les résultats là dedans 
        self.output_path = self.path+'/'+str(ep)+'/'  # Pour chaque épisode
        self.vtu_path    = self.output_path+'vtu/'
        self.effort   = self.output_path+'effort/'
        self.msh_path   = self.output_path+'msh/'
        self.t_mesh_path   = self.output_path+'t_mesh/'
        
        os.makedirs(self.effort)
        os.makedirs(self.vtu_path)
        os.makedirs(self.msh_path)
        os.makedirs(self.t_mesh_path)
        
        os.system('cp -r cfd ' + self.output_path + '.')   # A quoi sert cette ligne ???
        
        # Convert action to coordinates 
        to_concatanate = np.array([self.x_camb, self.y_camb])    ###### a enlever si cambrure bouge
        control_parameters = np.concatenate((np.array(x), to_concatanate))  #### enlever ça ou le toucher pour faire varier cambrure

        # create the shape 
        self.shape_generation_dussauge(control_parameters)

        # convert to .t
        os.system('cd '+self.output_path+'cfd ; python3 gmsh2mtc.py')
        os.system('cd '+self.output_path+'cfd ; cp -r airfoil.msh ../msh')
        os.system('cd '+self.output_path+'cfd ; module load cimlibxx/master')
        os.system('cd '+self.output_path+'cfd ; echo 0 | mtcexe airfoil.t')
        os.system('cd '+self.output_path+'cfd ; cp -r airfoil.t ../t_mesh')
        
        # solving the problem
        self.solve_problem_cimlib()

        # Compute the reward 
        self.compute_reward(control_parameters)

        # On écrit la durée

        self.time_end = dt.datetime.now()
        difference = self.time_end - self.time_init
        heures, reste = divmod(difference.seconds, 3600)
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


    ############ Bezier generation function ##############################################


    def quadraticBezier(self,t,points):
        B_x=(1-t)*((1-t)*points[0][0]+t*points[1][0])+t*((1-t)*points[1][0]+t*points[2][0])
        B_y=(1-t)*((1-t)*points[0][1]+t*points[1][1])+t*((1-t)*points[1][1]+t*points[2][1])
        return B_x,B_y

    def cambrure(self, x, y, numPts):
        # fonction pour avoir la cambrure
        curve=[]
        t=np.array([i*1/numPts for i in range(0,numPts)])
        # calculate first Bezier curve
        B_x,B_y=self.quadraticBezier(t,[(0.,0.),(x,y),(1.,0.)])
        curve=list(zip(B_x,B_y))
        return curve

    def find_camber_y(self, x, cambrure_coord):
        # pour un x donné il donne le y de la cambrure le plus proche
        for k,coord_camber in enumerate(cambrure_coord):
            if coord_camber[0] > x :
                return (coord_camber[1]+cambrure_coord[k-1][1])/2 # on prend la moyenne des deux 
        return 0.

    def reconstruct_control_points(self, control_parameter):
        # Les points autour desquels on bouge
        x_param_cambrure, y_param_cambrure =  control_parameter[-2:]  # les deux points definissant la cambrure 
        cambrure_coord = self.cambrure(x_param_cambrure, y_param_cambrure,16*40)
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
        control_points[5][1] =  control_parameter[0] + self.find_camber_y(control_points[5][0], cambrure_coord)
        control_points[6][1] = - control_parameter[0] + self.find_camber_y(control_points[6][0], cambrure_coord)
        for k in range(4):
            control_points[k+1][1] =  control_parameter[1+k] + self.find_camber_y(control_points[k+1][0], cambrure_coord)
        for k in range(3):
            control_points[k+7][1] =  control_parameter[5+k] +self.find_camber_y(control_points[k+7][0], cambrure_coord)

        return control_points

    def airfoil(self,ctlPts,numPts):
        curve=[]
        t=np.array([i*1/numPts for i in range(0,numPts)])
        
        # calculate first Bezier curve
        midX = (ctlPts[1][0]+ctlPts[2][0])/2
        midY = (ctlPts[1][1]+ctlPts[2][1])/2
        B_x,B_y = self.quadraticBezier(t,[ctlPts[0],ctlPts[1],[midX,midY]])
        curve = curve+list(zip(B_x,B_y))

        # calculate middle Bezier Curves
        for i in range(1,len(ctlPts)-3):
            midX_1=(ctlPts[i][0]+ctlPts[i+1][0])/2
            midY_1=(ctlPts[i][1]+ctlPts[i+1][1])/2
            midX_2=(ctlPts[i+1][0]+ctlPts[i+2][0])/2
            midY_2=(ctlPts[i+1][1]+ctlPts[i+2][1])/2

            B_x,B_y = self.quadraticBezier(t,[[midX_1,midY_1],ctlPts[i+1],[midX_2,midY_2]])
            curve=curve+list(zip(B_x,B_y))                     
    
        # calculate last Bezier curve
        midX=(ctlPts[-3][0]+ctlPts[-2][0])/2
        midY=(ctlPts[-3][1]+ctlPts[-2][1])/2

        B_x,B_y= self.quadraticBezier(t,[[midX,midY],ctlPts[-2],ctlPts[-1]])
        curve=curve+list(zip(B_x,B_y))
        curve.append(ctlPts[-1])
        return curve


    def rotate(self,curve):
        curve = np.array(curve)
        rotate_matrix = np.array([[np.cos(self.angle), np.sin(self.angle)], [-np.sin(self.angle), np.cos(self.angle)]])
        return curve @ rotate_matrix

    def polygon_area(self,curve):
        curve = np.array(curve)
        x = curve [:,0]
        y = curve[:,1]
        correction = x[-1] * y[0] - y[-1]* x[0]
        main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
        return 0.5*np.abs(main_area + correction)

    def punition(self):
        # Si on est en dessous de l'aire min => punition 
        if self.area < self.area_min :
            return np.exp((self.area_min/self.area) -1) - 1 # vaut 0 au début 
        else : 
            return 0. 

