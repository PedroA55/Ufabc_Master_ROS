import numpy as np
from quat_utils import QuatProd, Quat2Euler, Euler2Quat
#from numpy.core.numeric import NaN
#from quat_utils import DerivQuat
from scipy.linalg import solve_continuous_are as solve_lqr, inv


class Controller:

    """
    This class contains the position and attitude control design and trajectory generator as well. 
    """
    
    #Constants
    Ixx = 16.83*10**-3
    Iyy = 16.83*10**-3
    Izz = 28.34*10**-3
    Ir  = 5*10**-5
    I1 = (Iyy-Izz)/Ixx
    I2 = (Izz-Ixx)/Iyy
    I3 = (Ixx-Iyy)/Izz
    
    #Mass and Gravity
    M, G = 1.03, 9.82

    KT = 1.435*10**-5
    KD = 2.4086*10**-7
    L = 0.26
    # Output Matrix
    C = np.eye(6)
    #Matrices Q and R
    Qo = np.diag([1, 1, 1, 1, 1, 1])*3.0
    Ro = np.diag([1, 1, 1])*2.0
    
    def __init__(self):

        #Some constants for control design
        self.J = np.array([[self.Ixx, 0, 0],
                           [0, self.Iyy, 0],
                           [0, 0, self.Izz]])
        self.ang_ant_des = np.zeros((3,1))
        self.theta_des_ant = np.zeros((3,1))
        self.i_error = np.zeros((3,1))
        #self.wd = 0 #condição inicial

    def f2w(self,f,m):
        """""
        Translates F (Thrust) and M (Body x, y and z moments) into eletric motor angular velocity (rad/s)
        input:
            f - thrust 
            m - body momentum in np.array([[mx, my, mz]]).T
        outputs:
            F - Proppeler Thrust - engine 1 to 4
            w - Proppeler angular velocity - engine 1 to 4
            F_new - clipped thrust (if control surpasses maximum motor thrust)
            M_new - clipped momentum (same as above)     
            
            Numa futura versão essa função pode ser um control allocator.       
        """""
        T = np.array([[self.KT, self.KT, self.KT, self.KT],
                      [-self.L*self.KT, 0, self.L*self.KT, 0],
                      [0, -self.L*self.KT, 0, self.L*self.KT],
                      [self.KD, -self.KD, self.KD, -self.KD]])
        
        u = np.array([f, float(m[0]), float(m[1]), float(m[2])]).T
        
        """
         'T' é a matriz de transformação, tal que 
         u = T*w
         w é o vetor dos quadrados das velocidades angulares,
         w = [w1**2, w2**2, w3**2, w4**2]

         w = T^(-1)*u         
        """

        w = np.linalg.solve(T, u) # T*w = u
        
        w1_rad = np.sqrt(np.abs(w[0]))
        w2_rad = np.sqrt(np.abs(w[1]))
        w3_rad = np.sqrt(np.abs(w[2]))
        w4_rad = np.sqrt(np.abs(w[3]))
        
        w_1 = w1_rad*(-1)*2*np.pi/60 # rad/seg to RPM e sentido do giro
        w_2 = w2_rad*2*np.pi/60      # rad/seg to RPM
        w_3 = w3_rad*(-1)*2*np.pi/60 # rad/seg to RPM e sentido do giro    
        w_4 = w4_rad*2*np.pi/60      # rad/seg to RPM
            
        """
        w é o vetor das velocidades angulares dos motores
        """
        
        W = np.array([[w_1,w_2,w_3,w_4]]).T
        wd = - w1_rad + w2_rad - w3_rad + w4_rad
        """
        Não está sendo utilizado clipping. 
        Isto é, não está sendo verificado o máximo esforço requerido.
        
        FM_new = np.dot(x, u)   # Nao está sendo utilizado        
        F_new = FM_new[0]       # Nao está sendo utilizado
        M_new = FM_new[1:4]     # Nao está sendo utilizado
        
        # step_effort = (u*K_F/(T2WR*M*G/4)*2)-1
        
        """
        F_new = 0
        M_new = 0
        return W, F_new, M_new, wd
    
    def pos_control_PD(self, pos_atual, pos_des, vel_atual, vel_des, accel_des, psi):

        #PD gains Real States
        Kp = np.array([[2, 0 ,0],
                       [0, 2, 0],
                       [0, 0, 8]])*1.0
        Kd = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 3]])*1.0

        # Kp = np.array([[2, 0 ,0],
        #                [0, 2, 0],
        #                [0, 0, 1.5]])*1.5
        # Kd = np.array([[3, 0, 0],
        #                [0, 2, 0],
        #                [0, 0, 1]])*1.5

        dpos_error = pos_des - pos_atual

        vel_error = vel_des - vel_atual

        n = accel_des/np.linalg.norm(accel_des)
        t = vel_des/np.linalg.norm(vel_des)
        b = np.cross(t, n, axis=0)

        if np.isnan(b).any:
            pos_error = dpos_error
        else:
            pos_error = (dpos_error.T@n)@n + (dpos_error.T@b)@b


        # u = Kp@dpos_error + Kd@vel_error

        # theta_des = np.arctan2(u[0], (u[2]+9.82))

        # phi_des = np.arctan2(-u[1], (u[2]+9.82)*np.cos(theta_des))

        # T = 1.03*(u[2] + 9.82)/(np.cos(theta_des)*np.cos(phi_des))

        
        # alo = 1
        rddot_c = accel_des + Kd@vel_error + Kp@pos_error

        T = self.M*(self.G + rddot_c[2])

        phi_des = (rddot_c[0]*np.sin(psi) - rddot_c[1]*np.cos(psi))/self.G
        theta_des = (rddot_c[0]*np.cos(psi) + rddot_c[1]*np.sin(psi))/self.G

        euler_out = np.array([
            phi_des[0],
            theta_des[0], 
            psi])  
        q_erro = Euler2Quat(euler_out)
        
        return T, q_erro, euler_out

    def pos_control_Quat(self, pos_atual, pos_des, vel_atual, vel_des, q_des):
        
        """
        Function that computes the desired thrust and quaternion for quadrotor
        based on desired trajectory.
        """

        # Compute position and velocity error
        error_pos = pos_atual - pos_des
        error_vel = vel_atual - vel_des

        #Gains for Estimated States
        #Proportional gain
        Kp = np.diag([4, 4, 20])*2.3
        # #Derivative gain
        Kd = np.diag([3.5, 3.5, 7])*1.3

        #Control force in inertial frame
        Fu = -Kp@error_pos - Kd@error_vel - 1.05*np.array([[0, 0, -9.8]]).T

        #Desired quaternion
        z = np.array([[0, 0, 1]]).T
        Fu_norm = np.linalg.norm(Fu)
        q = np.zeros((4,1))
        q[0] = (np.vdot(z, Fu) + Fu_norm)
        q[1:4] =  np.cross(z, Fu, axis=0)
        q_norm = np.linalg.norm(q)
        q_p = q/q_norm
        q_erro = QuatProd(q_p, q_des)        
        euler_out = Quat2Euler(q_erro)
        return Fu_norm, q_erro, euler_out
    
    def att_control_PD(self, ang_atual, ang_vel_atual, ang_des, freq):
        
        phi = float(ang_atual[0])
        theta = float(ang_atual[1])
        psi = float(ang_atual[2])

        #PID gains Real States
        Kp = np.array([[30, 0 ,0],
                       [0, 30, 0],
                       [0, 0, 1.4]])*3.5
        Kd = np.array([[8, 0, 0],
                       [0, 8, 0],
                       [0, 0, 1]])*0.8

        angle_error = ang_des - ang_atual
        #ang_vel_des = (ang_des - self.ang_ant_des)/0.01
        #
        # TODO Implementar filtro PB para essa derivada.
        #
        ang_vel_des = (ang_des - self.ang_ant_des)*freq

        ang_vel_error = ang_vel_des - ang_vel_atual

        T = np.array([[1/self.Ixx, np.sin(phi)*np.tan(theta)/self.Iyy, np.cos(phi)*np.tan(theta)/self.Izz],
                      [0, np.cos(phi)/self.Iyy, -np.sin(phi)/self.Izz],
                      [0, np.sin(phi)/np.cos(theta)/self.Iyy, np.cos(phi)/np.cos(theta)/self.Izz]])

        u = np.linalg.inv(T)@(Kp@angle_error + Kd@ang_vel_error)
        # u = (Kp@angle_error + Kd@ang_vel_error)

        #Optimal input
        tau_x = float(u[0])
        tau_y = float(u[1])
        tau_z = float(u[2])

        self.ang_ant_des = ang_des

        tau = np.array([tau_x, tau_y, tau_z])
        
        error = np.array([angle_error, ang_vel_error]).reshape((6,1))
        
        return tau, error

    def att_control_Quat(self, q_atual, q_des, ang_vel_atual):
        
        """
        Computes the desired torques for quadrotor based on
        desired quaternion and angular velocities.
        """

        #Compute quaternion's log mapping
        ln_q = self.log_mapping(q_atual)
        ln_q_des = self.log_mapping(q_des)

        #Vector with axis-angle representation
        theta = 2*ln_q
        theta_des = 2*ln_q_des

        # Attitude states
        x_att = np.zeros((6,1))
        x_att[0:3] = theta
        x_att[3:6] = ang_vel_atual

        #Desired attitude states
        x_att_des = np.zeros((6,1))
        x_att_des[0:3] = theta_des
        # derivative of theta_des
        x_att_des[3:6] = (theta_des - self.theta_des_ant)/0.01

        #Attitude states error
        error = x_att - x_att_des
    
        # #Gains for real states
        #Gain matrix
        # K = np.zeros((3,6))
        # #Proportional gain
        # K[0:3, 0:3] = np.diag([6, 6, 2.5])*30
        # #Derivative gain
        # K[0:3, 3:6] = np.diag([1, 1, .1])*10

        #Gains for Estimated States
        #Gain matrix
        K = np.zeros((3,6))
        #Proportional gain
        K[0:3, 0:3] = np.diag([7, 7, 2.5])*40
        #Derivative gain
        K[0:3, 3:6] = np.diag([1.5, 1.5, .1])*15
        #PD control law
        u = -K@error

        #Save current desired theta to compute the next desired angular velocity
        self.theta_des_ant = theta_des

        #Compute the desired torques
        tau = self.J@u + np.cross(ang_vel_atual, self.J@ang_vel_atual, axis=0)

        return tau, error    

    def att_control_LQR(self, euler_atual, euler_desejado, ang_vel_atual, freq,wd):
        '''
        Function that computes the desired torques for quadrotor based on
        desired Euler angles and angular velocities.
        The vector state is x = [phi, theta, psi, p, q, r]
        '''
        x = np.array([[euler_atual[0],
                       euler_atual[1],
                       euler_atual[2],
                       ang_vel_atual[0],
                       ang_vel_atual[1],
                       ang_vel_atual[2]]]).T

        # TODO Implementar filtro PB para essa derivada.
        vel_ang_desejada = (euler_desejado - self.ang_ant_des)*freq
        
        x_des = np.array([[euler_desejado[0],
                           euler_desejado[1],
                           euler_desejado[2],
                           vel_ang_desejada[0],
                           vel_ang_desejada[1],
                           vel_ang_desejada[2]]]).T
        error = np.array([x- x_des]).reshape((6,1))
        # 
        # TODO Implementar perturbação Ir*Omega_r
        A = np.zeros((6,6))
        A[0,3] = 1
        A[1,4] = 1
        A[2,5] = 1
        A[4,5] = -(self.Ir/self.Ixx)*wd
        A[5,4] = -(self.Ir/self.Iyy)*wd
        B = np.zeros((6,3))
        B[3,0] = 1/self.Ixx
        B[4,1] = 1/self.Iyy
        B[5,2] = 1/self.Izz
        Q = np.diag([1, 1, 1, 1, 1, 1])*5.0
        R = np.diag([1, 1, 1])*5.0
        P = solve_lqr(A, B, Q, R)
        K = inv(R)@B.T@P
        u = -K@error
        self.ang_ant_des = euler_desejado
        #self.wd = wd
        return np.array(u).reshape((3,1)), np.array(error).reshape((6,1))
    
    def att_control_SDRE(self, euler_atual, euler_desejado, ang_vel_atual, freq,wd):
        '''
        Function that computes the desired torques for quadrotor based on
        desired Euler angles and angular velocities.
        The vector state is x = [phi, theta, psi, p, q, r]
        '''
        x = np.array([[euler_atual[0],
                       euler_atual[1],
                       euler_atual[2],
                       ang_vel_atual[0],
                       ang_vel_atual[1],
                       ang_vel_atual[2]]]).T

        # TODO Implementar filtro PB para essa derivada.
        vel_ang_desejada = (euler_desejado - self.ang_ant_des)*freq
        
        x_des = np.array([[euler_desejado[0],
                           euler_desejado[1],
                           euler_desejado[2],
                           vel_ang_desejada[0],
                           vel_ang_desejada[1],
                           vel_ang_desejada[2]]]).T
        error = np.array([x- x_des]).reshape((6,1))
        x = x.reshape((6,1))
        x_des = x_des.reshape((6,1))
        
        #Parâmetros que serão utilizados
        Jr = self.Ir; L = self.L
        Ixx = self.Ixx; Iyy = self.Iyy; Izz = self.Izz
        I1 = (Iyy-Izz)/Ixx;  I2 = (Izz-Ixx)/Iyy;  I3 = (Ixx-Iyy)/Izz
        Qo = self.Qo; Ro = self.Ro; C = self.C
        
        
        #Passo 1 - Atualizando os termos das matrizes subótimas
        Sphi = np.sin(euler_atual[0]); Cphi = np.cos(euler_atual[0])
        Stheta = np.sin(euler_atual[1]); Ctheta = np.cos(euler_atual[1]); Ttheta = np.tan(euler_atual[1])
        
        #coeficientes
        a1 = Ttheta*(1-(Sphi**2)*I2 + (Cphi**2)*I3)
        a2 = Stheta*Sphi*Cphi*(I2+I3)
        a3 = 1/Ctheta + Ctheta*(Cphi**2 - Sphi**2)*I1 + Stheta*(Sphi**2)*Ttheta*I2 - Stheta*(Cphi**2)*Ttheta*I3
        a4 = -Sphi*Cphi*I1
        a5 = Sphi*Cphi*((Ctheta**2)*I1-(Stheta**2)*(I2+I3))
        a6 = -Sphi*Cphi*(I3+I2); a7 = Ctheta*(-1 + (Cphi**2)*I2 - (Sphi**2)*I3)
        a8 = Stheta*Sphi*Cphi*(I2+I3)
        a9 = -Stheta*Ctheta*((Cphi**2)*I2-(Sphi**2)*I3)
        a10 = (1/Ctheta)*(1 -(Sphi**2)*I2 + (Cphi**2)*I3)
        a11 = Sphi*Cphi*(I2+I3); a12 = Ttheta*(1+(Sphi**2)*I2-(Cphi**2)*I3)
        a13 = -Stheta*Sphi*Cphi*(I2+I3)
        b1 = L/Ixx; b2 = (L*Ttheta*Sphi)/Iyy; b3 = (L*Ttheta*Cphi)/Izz
        b4 = (L*Cphi)/Iyy; b5 = -(L*Sphi)/Izz; b6 = (L*Sphi)/(Ctheta*Iyy)
        b7 = (L*Cphi)/(Ctheta*Izz)
        c1 = -(Cphi*Jr*wd)/Ixx; c2 = -((Ctheta*Sphi/Ixx)+(Ttheta*Stheta*Sphi/Iyy))*Jr*wd
        c3 = (Ttheta*Sphi*Jr*wd)/Iyy
        c4 = -(Stheta*Cphi*Jr*wd)/Iyy
        c5 = (Cphi*Jr*wd)/Iyy
        c6 = -(Stheta*Sphi*Jr*wd)/(Ctheta*Iyy)
        c7 = (Sphi*Jr*wd)/(Ctheta*Iyy)
        A88 = a1*x[4]+ a2*x[5] +c3
        A810 = a4*x[4] + c1
        A812 = a3*x[4] + a5*x[5] + c2
        A108 = a6*x[4]+ a7*x[5] +c5
        A1010 = a8*x[5]
        A1012 = a9*x[5] + c4
        A128 = a10*x[4]+ a11*x[5] +c7
        A1210 = a12*x[5]
        A1212 = a13*x[5] + c6
        # Matriz Ao
        Ao = np.zeros((6,6))
        Ao[0,3]= 1; Ao[1,4]= 1; Ao[2,5]= 1
        Ao[3,3]= A88; Ao[3,4]= A810; Ao[3,5]= A812
        Ao[4,3]= A108; Ao[4,4]= A1010; Ao[4,5]= A1012
        Ao[5,3]= A128; Ao[5,4]= A1210; Ao[5,5]= A1212
        # Matriz Bo
        Bo = np.zeros((6,3))
        Bo[3,0]=b1; Bo[3,1]= b2; Bo[3,2]=b3
        Bo[4,1]=b4; Bo[4,2]= b5
        Bo[5,1]=b6; Bo[5,2]= b7
        
        # Passo 2 - Com as matrizes Q e R redefinir as matrizes E(x), V(x) e W(x)
        E = Bo@inv(Ro)@np.transpose(Bo)
        V = np.transpose(C)@Qo@C
        W = np.transpose(C)@Qo
        # Passo 3 - Resolver a Eq. Algébrica de Ricatti
        P = solve_lqr(Ao, Bo, Qo, Ro)
        # Passo 4 - Ganhos K(x) e Kz(x)
        K = inv(Ro)@np.transpose(Bo)@P
        Kz =inv(Ro)@np.transpose(Bo)@inv(P@E-np.transpose(Ao))@W; # Tem que arrumar aqui
        # Passo 5 - Sinal de controle Final
        u = -K@x + Kz@x_des
        M2 = u[0]
        M3 = u[1]
        M4 = u[2]
        
        return np.array(u).reshape((3,1)), np.array(error).reshape((6,1))
    
    #################################### TRAJECTORY PLANNER FUNCTIONS ######################################################
    
    def log_mapping(self, q):

        """
        Function that compute the quaternion's log mapping
        """

        # Quaternion's scalar part
        q_s = q[0,0]
        #Quaternion's vector part
        q_vec = q[1:4]
        #Quaternion vector norm
        q_vec_norm = np.linalg.norm(q_vec)

        #Logarithimic mapping for unit quaternion
        if q_vec_norm != 0:
            ln_q = (q_vec/q_vec_norm)*np.arccos(q_s)
        else:
            ln_q = np.zeros((3,1))
        
        return ln_q
    
    #Returns the derivatives
    def polyT(self, n, k, t):

        T = np.zeros((n,1))
        D = np.zeros((n,1))

        for i in range(1, n+1):
            D[i-1] = i - 1
            T[i-1] = 1

        for j in range(1, k+1):
            for i in range(1, n+1):
                T[i-1] = T[i-1]*D[i-1]

                if D[i-1]>0:
                    D[i-1] = D[i-1] - 1
                

        for i in range(1, n+1):
            T[i-1] = T[i-1]*t**D[i-1]
        
        T = T.T

        return T

    #Get the optimal snap functions coefficients 
    def getCoeff_snap(self, waypoints, t):

        n = len(waypoints) - 1
        A = np.zeros((8*n, 8*n))
        b = np.zeros((8*n, 1))

        # print(b.T)

        row = 0
        #Initial constraints
        for i in range(0, 1):
            A[row, 8*(i):8*(i+1)] = self.polyT(8, 0, t[0])
            b[i, 0] = waypoints[0]
            row = row + 1
        
        for k in range(1, 4):
            A[row, 0:8] = self.polyT(8, k, t[0])
            row = row + 1
        
        if n == 1:
            #Last P constraints
            for i in range(0, 1):
                A[row, 8*(i):8*(i+1)] = self.polyT(8, 0, t[-1])
                b[row, 0] = waypoints[1]
                row = row + 1  
            
            for k in range(1, 4):
                A[row, 8*(n) - 8:8*(n)] = self.polyT(8, k, t[-1])
                row = row + 1



        elif n>1:


            #Pi constraints
            shift = 0
            for j in range(1, n):
                
                
                
                for i in range(0, 2):
                    A[row, 8*(i+shift):8*(i+1+shift)] = self.polyT(8, 0, t[j])
                    b[row, 0] = waypoints[j]

                    row = row + 1

                for k in range(1, 7):
                    A[row, 8*(j-1):8*(j)] = self.polyT(8, k, t[j])
                    A[row, 8*(j):8*(j+1)] = -self.polyT(8, k, t[j])
                    row = row + 1
                
                shift += 1
            
            
            #Last P constraints
            for i in range(0, 1):
                A[row, 8*(n) - 8:8*(n)] = self.polyT(8, 0, t[-1])
                b[row, 0] = waypoints[n]
                row = row + 1
            
            for k in range(1, 4):
                A[row, 8*(n) - 8:8*(n)] = self.polyT(8, k, t[-1])
                row = row + 1


        coeff = np.linalg.inv(A)@b
        
        c_matrix = coeff.reshape(n, 8)

        return A, b, c_matrix

    #Compute the snap trajectory equations at time 't'
    def equation_snap(self, t, c_matrix, eq_n):
        x = self.polyT(8, 0, t)
        v = self.polyT(8, 1, t)
        a = self.polyT(8, 2, t)
        j = self.polyT(8, 3, t)
        s = self.polyT(8, 4, t)
        

        P = np.sum(x*c_matrix[eq_n,:])
        V = np.sum(v*c_matrix[eq_n,:])
        A = np.sum(a*c_matrix[eq_n,:])
        J = np.sum(j*c_matrix[eq_n,:])
        S = np.sum(s*c_matrix[eq_n,:])
        

        return P, V, A, J, S
    
    #Storage the values of any equations at time 't' in lists
    def evaluate_equations_snap(self, t, step, c_matrix):
            
        skip = 0

        x_list = []
        v_list = []
        a_list = []
        j_list = []
        s_list = []

        for i in np.arange(0, t[-1], step):

            if skip == 0:

                if i >= t[skip] and i<=t[skip+1]:
                
                    p, v, a, j, s = self.equation_snap(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)
                    j_list.append(j)
                    s_list.append(s)

                else:

                    skip += 1

                    p, v, a, j, s = self.equation_snap(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)
                    j_list.append(j)
                    s_list.append(s)

            elif skip > 0 and skip < len(t):

                if i > t[skip] and i <= t[skip+1]:

                    p, v, a, j, s = self.equation_snap(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)
                    j_list.append(j)
                    s_list.append(s)

                else:

                    skip += 1

                    p, v, a, j, s = self.equation_snap(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)
                    j_list.append(j)
                    s_list.append(s)
        
        return x_list, v_list, a_list, j_list, s_list

    #Get the optimal acceleration functions coefficients 
    def getCoeff_accel(self, waypoints, t):

        n = len(waypoints) - 1
        A = np.zeros((4*n, 4*n))
        b = np.zeros((4*n, 1))

        # print(b.T)

        row = 0
        #Initial constraints
        for i in range(0, 1):
            A[row, 4*(i):4*(i+1)] = self.polyT(4, 0, t[0])
            b[i, 0] = waypoints[0]
            row = row + 1
        
        for k in range(1, 2):
            A[row, 0:4] = self.polyT(4, k, t[0])
            row = row + 1
        
        if n == 1:
            #Last P constraints
            for i in range(0, 1):
                A[row, 4*(i):4*(i+1)] = self.polyT(4, 0, t[-1])
                b[row, 0] = waypoints[1]
                row = row + 1  
            
            for k in range(1, 2):
                A[row, 4*(n) - 4:4*(n)] = self.polyT(4, k, t[-1])
                row = row + 1



        elif n>1:


            #Pi constraints
            shift = 0
            for j in range(1, n):
                
                
                
                for i in range(0, 2):
                    A[row, 4*(i+shift):4*(i+1+shift)] = self.polyT(4, 0, t[j])
                    b[row, 0] = waypoints[j]

                    row = row + 1

                for k in range(1, 3):
                    A[row, 4*(j-1):4*(j)] = self.polyT(4, k, t[j])
                    A[row, 4*(j):4*(j+1)] = -self.polyT(4, k, t[j])
                    row = row + 1
                
                shift += 1
            
            
            #Last P constraints
            for i in range(0, 1):
                A[row, 4*(n) - 4:4*(n)] = self.polyT(4, 0, t[-1])
                b[row, 0] = waypoints[n]
                row = row + 1
            
            for k in range(1, 2):
                A[row, 4*(n) - 4:4*(n)] = self.polyT(4, k, t[-1])
                row = row + 1


        coeff = np.linalg.inv(A)@b
        
        c_matrix = coeff.reshape(n, 4)

        return A, b, c_matrix

    #Compute the acceleration trajectory equations at time 't'
    def equation_accel(self, t, c_matrix, eq_n):
        x = self.polyT(4, 0, t)
        v = self.polyT(4, 1, t)
        a = self.polyT(4, 2, t)

        P = np.sum(x*c_matrix[eq_n,:])
        V = np.sum(v*c_matrix[eq_n,:])
        A = np.sum(a*c_matrix[eq_n,:])
        

        return P, V, A

    #Storage the values of any equations at time 't' in lists
    def evaluate_equations_accel(self, t, step, c_matrix):
            
        skip = 0

        x_list = []
        v_list = []
        a_list = []

        for i in np.arange(0, t[-1], step):

            if skip == 0:

                if i >= t[skip] and i<=t[skip+1]:
                
                    p, v, a = self.equation_accel(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)

                else:

                    skip += 1

                    p, v, a = self.equation_accel(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)

            elif skip > 0 and skip < len(t):

                if i > t[skip] and i <= t[skip+1]:

                    p, v, a = self.equation_accel(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)

                else:

                    skip += 1

                    p, v, a = self.equation_accel(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)
        
        return x_list, v_list, a_list
    
    def point_to_np_array(self, point):
        return np.array([[point.x, point.y, point.z]]).T
    
    def euler_to_np_array(self, euler):
        return np.array([[euler.phi, euler.theta, euler.psi]]).T
    
    def quat_to_np_array(self, quat):
        return np.array([[quat.w, quat.x, quat.y, quat.z]]).T
