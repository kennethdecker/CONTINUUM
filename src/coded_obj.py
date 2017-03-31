import numpy as np
from peewee import *
from create_database import *
import sys
import warnings
from query_obj import QueryObj
from coded_matrix import coded_matrix
from mission_profile import MissionProfile

class CodedObj(object):
    ''' This is the main object that will be used throughout the constraint analysis. This object contains all of the methods 
    necessary to apply mission analysis, constraint analysis, evaluate performance, and apply the topsis method.
    
    --- Inputs (initialization) ---
    level_array: (int numpy.array)
        This is an array that tells the number of levels that must be considered for each type of mdoule being analyzed.
        ex. If analysis is of 2 batteries, 3 motors, and 3 props, then level_array = np.array([2,3,3], dtype = np.int)
    module_list: (str list)
        This is an array that contains the names of each module class being considered. The order of the module class names MUST
        correspond to the orders given in the level_array. 
    QueryObj: (sqlite obj)
        This is an object that contains the database model objects for each module being tested. See query_obj.py for instructions

    '''
    def __init__(self, level_array, module_list, QueryObj):

        self.levels = level_array
        self.modules = module_list
        self.query = QueryObj
        self.compatibility = np.ones((np.sum(level_array), (np.sum(level_array))), dtype = np.int)
        self.mission = []
        self.coded_matrix()
        self.build_namelist()


    def coded_matrix(self):
        '''
        This function creates a full factorial coded matrix for a set of alternatives. Pulls level array from initialization of object 
        to generate coded matrix. Is called in initialization. Does not return anything, but does create attribute called "design" in
        main object.

        '''
        levels = self.levels

        m = np.prod(levels)
        n = np.sum(levels)
        design = np.zeros((m,n), dtype = np.int)

        categories = len(levels)

        ssize = m
        ncycles = ssize
        start = 0

        for i in range(categories):

            coded = np.diag(np.matrix([1]*levels[i]).A1)
            nreps = ssize/ncycles
            ncycles = ncycles/levels[i]

            coded = np.repeat(coded, nreps, axis = 1)

            coded = coded.T

            new_coded = coded
            for j in range(ncycles-1):
                new_coded = np.vstack((new_coded, coded))

            design = design.T
            design[start:start + levels[i],:] = new_coded.T
            start = start + levels[i]
            design = design.T

        # return design
        self.array = design

    def build_namelist(self):
        '''
        This method generates a dictionary that lists the names of every module for each module class. Keys of dictionary are module class names,
        contents are the names of each module of that class. This namelist is used to apply incompatibilities in order to eliminate incompatible 
        modules. Does not return anything, but does create "namelist" attribute in main object. 
         '''
        
        namelist = {}
        module_list = self.modules
        for i in module_list:
            names = []
            if i == 'Battery':
                for j in self.query.battery:
                    names.append(j.name)

            elif i == 'Motor':
                for j in self.query.motor:
                    names.append(j.name)

            elif i == 'Prop':
                for j in self.query.prop:
                    names.append(j.name)

            elif i == 'ESC':
                for j in self.query.esc:
                    names.append(j.name)

            elif i == 'Frame':
                for j in self.query.frame:
                    names.append(j.name)

            namelist[i] = names

        self.namelist = namelist 

    def add_mission(self, missionObj):
        ''' 
        Adds mission object to the object attribute "mission." Mission objects are stored as a list in order to account for multiple possible
        mission profiles. See mission_profile.py for mission setup instructions.
        '''
        self.mission.append(missionObj)

    def evaluate_cases(self, TR_static, initial_weight = 0, initial_current = 0, num_props = 4):
        ''' 
        This module evaluates important performance parameters for the entire full factorial design space. Pulls design attribute from object, 
        identifies modules in each design, and evaluates performance.

        --- Assumptions ---
        All motors draw equal current
        All motors carry weight evenly during hover
        Payload current is known and drawn off battery for duration of flight

        --- Inputs ---
        TR_static: (float)
            This is the static thrust to weight ratio of the vehicle set by the user.
        initial_weight: (float, default = 0)
            This is the initial weight of the vehicle. The modules being considered by the design only account for a portion of the weight of 
            the vehicle, and are computed in this method. This value accounts for the weight of modules and payload not considered in the analysis.
            Units of weight are in grams.
        initial_current: (float, default = 0)
            This is the current draw for anything aboard the vehicle besides the propulsion system.[A]
        num_props: (int, default = 4)
            This is the number of propellers on the vehicle. This determines how much thrust each propeller must generate.

        --- Added attributes ---
        results: (float np.array)
            This is an array of performance values that will be fed into run_topsis(). Rows correspond to each designand are taken from the 
            design array. The columns correspond to performance values. Order is set in attribute_list
        attribute_list: (str list)
            This list tells which performance values correspond to each column of the results array.

        --- Raised Exceptions ---
        ValueError:
            Raised if sum of level_array supplied by user does not correpond to total number of columns in design array

        --- Handled Exceptions ---
        RuntimeError:
            compute_current and run_mission methods are called and contain implicit solvers. Each raises a RuntimeError if values do not converge.
            Exceptions are handled by countin the total number of failed cases, reporting their number, and printing the final number of failed cases 
            at the end.
        '''

        m,n = self.array.shape
        attribute_list = ['Weight', 'Efficiency', 'Current', 'Endurance at max thrust', 'Charge']
        results = np.zeros((m, len(attribute_list)), dtype = np.float64)
        modules = len(self.levels)

        if np.sum(self.levels) != n:
            raise ValueError('Sum of level array must equal number of clumns of coded matrix')
        b = 0
        for i in range(m):

            row = self.array[i]

            weight = initial_weight
            col = 0

            for j in range(modules):

                num = self.levels[j]

                for k in range(num):

                    if self.modules[j] == 'Battery':
                        weight = weight + self.array[i][col] * self.query.battery[k].weight

                        if self.array[i][col] == 1:
                            battery = self.query.battery[k]

                    elif self.modules[j] == 'Motor':
                        weight = weight + self.array[i][col] * self.query.motor[k].weight * num_props
                        
                        if self.array[i][col] == 1:
                            motor = self.query.motor[k]

                    elif self.modules[j] == 'Prop':
                        weight = weight + self.array[i][col] * self.query.prop[k].weight * num_props
                        
                        if self.array[i][col] == 1:
                            prop = self.query.prop[k]

                    elif self.modules[j] == 'ESC':
                        weight = weight + self.array[i][col] * self.query.esc[k].weight

                    elif self.modules[j] == 'Frame':
                        weight = weight + self.array[i][col] * self.query.frame[k].weight

                    col = col + 1

            efficiency = self.compute_efficiency(battery, motor, prop)

            try:
                I = self.compute_current(battery, motor, prop, TR_static*weight/num_props)
                
            except RuntimeError:
                b += 1
                for j in range(len(attribute_list)):
                    results[i][j] = np.nan
                warnings.warn('Case %d failed to converge' % i)
                continue

            if np.isnan(I):
                b += 1
                for j in range(len(attribute_list)):
                    results[i][j] = np.nan
                warnings.warn('Case %d failed to converge' % i)
                continue

            I = I + initial_current/num_props
            endurance = self.compute_endurance(battery, I)
            try:
                Q = self.run_mission(battery, motor, prop, TR_static, weight, initial_current = initial_current)
            except RuntimeError:
                b += 1
                for j in range(len(attribute_list)):
                    results[i][j] = np.nan
                warnings.warn('Case %d failed to converge' % i)
                continue
            # print Q
            if (np.isnan(Q) or Q < 0):
                b += 1
                for j in range(len(attribute_list)):
                    results[i][j] = np.nan
                warnings.warn('Case %d failed to converge' % i)
                continue


            # results[i][0] = weight
            # results[i][1] = efficiency
            for j in range(len(attribute_list)):
                if attribute_list[j] == 'Weight':
                    results[i][j] = weight
                elif attribute_list[j] == 'Efficiency':
                    results[i][j] = efficiency
                elif attribute_list[j] == 'Current':
                    results[i][j] = I
                elif attribute_list[j] == 'Endurance at max thrust':
                    results[i][j] = endurance
                elif attribute_list[j] == 'Charge':
                    results[i][j] = Q
            

        nans = []
        for i in range(m):
            if np.isnan(results[i,1]):
                nans.append(i)

        np.delete(results, nans, axis = 0)
        np.delete(self.array, nans, axis = 0)
        self.attribute_list = attribute_list
        self.results = results
        print '%d Current computations failed to converge' % b


    def compute_efficiency(self, batteryObj, motorObj, propObj):

        ''' 
        
        '''

        Kv = motorObj.kv
        Rm = motorObj.Rm
        I0 = motorObj.I0
        I_max = motorObj.max_current

        V = batteryObj.volts
        Q = batteryObj.charge

        d = propObj.diameter
        pitch = propObj.pitch

        rho = .002377
        Kt = 1352.4/Kv


        I = np.linspace(1,20)

        torque = Kt*(I-I0)
        RPM = (V-I*Rm)*Kv
        P_out = np.multiply(torque, RPM)/1352.4
        P_in = V*I
        eta = np.divide(P_out,P_in)

        return np.max(eta)

    def compute_current(self, batteryObj, motorObj, propObj, T, RPMout = False):

        '''
        This module computes the current draw necessary to generate a ceratin amout of thrust. Thrust must be given for each propeller,
        NOT for the entire vehicle. Pulls experimental data from text files in prop_data folder. Directories to the appropriate file 
        found in the database.

        --- Assumptions ---
        All losses accounted for in database values
        Ct and Cq values lie on regression lines
        Ct and Cq vary approximately linearly vs. RPM (ideally would be constant)
        Thrust given is static thrust at sea level in standard atmosphere

        --- Inputs ---
        batteryObj: (sqlite object)
            object from database for a battery query.
        motorObj: (sqlite object)
            object from database for a motor query.
        propObj: (sqlite object)
            object from database for a prop query.
        T: (float)
            value of thrust per propeller. [g]
        RPMout: (bool, default = False)
            switch that determines whether or not function returns RPM

        --- Returns ---
        I: (float)
            Current necessary to produce a given thrust per propeller. [A]
        RPM: (float)
            Necessary RPM to produce provided thrust [RPM]

        --- Raise Exceptions ---
        RuntimeError:
            raised if implicit solver does not converge when computing RPM
        '''

        #Unpack Prop Info
        D = propObj.diameter
        pitch = propObj.pitch
        data_folder = propObj.data

        #Unpack Motor Info
        kv = motorObj.kv
        motor_I_max = motorObj.max_current
        Rm = motorObj.Rm
        I0 = motorObj.I0
        kt = 1352.4/kv
        rho = .002378

        #Unpack Battery info
        V_in = batteryObj.volts

        if data_folder != 'null':
            n_list, Ct_list, Cq_list = np.loadtxt(data_folder, delimiter = ',', skiprows = 1, unpack = True) #Unpack info from raw data
            fit1 = np.polyfit(n_list, Ct_list, 1)
            p = np.poly1d(fit1)

            n_new = 200.0
            Ct = 1.0

            eps = 1.

            i = 1
            while eps > 1e-4:
                
                n_old = n_new
                # Ct = p(n_old)
                
                # n_new = np.sqrt(T/(rho*Ct*((D/12.)**4.0)))

                # lam = .8
                # n_new = lam*n_new + (1-lam)*n_old

                h = 1e-4
                f1 = p(n_old) - (T/(rho*(n_old**2.)*((D/12.)**4.)))
                n_h = n_old + h
                f2 =  p(n_h) - (T/(rho*(n_h**2.)*((D/12.)**4.)))
                f_prime = (f2-f1)/h

                n_new = n_old - (f1/f_prime)                

                eps = np.abs((n_new - n_old)/np.maximum(n_old, 1e-5))

                i += 1

                if i > 200:
                    print 'max iter reached'
                    print 'n = %f ' % n_new
                    print 'eps = %f' % eps
                    raise RuntimeError('RPM computation did not converge')

                    # sys.exit()


            # n_list2, Cq_list = np.loadtxt(data_folder + 'n_vs_cq.txt', skiprows = 1, unpack = True)
            fit2 = np.polyfit(n_list, Cq_list, 1)
            p2 = np.poly1d(fit2)

            Cq = p2(n_new)
            tau = Cq*rho*(n_new**2.0)*((D/12.)**5.)
            rpm = n_new*60.
            I_req = (tau/kt) + I0
            V_req = (rpm/kv) + I_req*Rm
            
            if RPMout:
            	return (I_req, n_new*60.0)
            else:
            	return I_req

    def compute_endurance(self, batteryObj, I, SF = .71, num_bat = 1, num_props = 4):
        ''' '''
        Q = batteryObj.charge*(60.0/1000.0)
        t = (num_bat*Q*SF)/(I*num_props)

        return t

    def compute_max_speed(self, batteryObj, motorObj, propObj, TR_static, weight, num_props = 4):
        ''' 
        This computes the maximum horizontal speed of the vehicle based simple equations of motion

        --- Assumptions ---
        Vertical component of thrust is equal to weight
        Velocity entering propeller is equal to Vx*cos(alpha) where alpha is bank angle
        exit velocity is equal to pitch velocity
        exit velocity is constant
        Assumes a CD = 1, Sref = .01 [m^2] (this is the value of the frame hub planform area)

        --- Inputs ---
        batteryObj: (sqlite object)
            object from database for a battery query
        motorObj: (sqlite object)
            object from database for a motor query
        propObj: (sqlite object)
            object from database for a prop query
        TR_static: (float)
            static thrust to weight ratio set by user
        weight: (float)
            weight of the vehicle [g]
        num_props: (int, default = 4)
            number of propellers on vehicle

        --- Returns ---
        u_max: (float)
            max horizontal velocity of vehicle given the stated assumptions [ft/s]

        --- Exceptions ---
        RuntimeError:
            raised if solver fails to converge when computing velocity 
        '''

        pitch = propObj.pitch
        rho = .002378
        CD = 1.
        S = .01*10.7639 #[sq ft]
        weight_lbs = weight*0.00220462 #[lbs]
        I, RPM = self.compute_current(batteryObj, motorObj, propObj, weight/num_props, RPMout = True)

        ue = RPM*pitch*(1./12.)*(1./5280.)*60.

        u_new = 0
        eps = 1
        step = 1e-4

        ii = 0
        while eps > 1e-6:
            u_old = u_new
            u_step = u_new + step
            ii += 1

            TR_new = 1
            eps2 = 1
            jj = 0
            while eps2 > 1e-6:
                TR_old = TR_new
                TR_step = TR_new + step
                jj += 1

                f1 = (TR_old/TR_static) + ((u_old/ue)*(np.sqrt(TR_old -1.)/TR_old)) - 1.0
                f2 = (TR_step/TR_static) + ((u_old/ue)*(np.sqrt(TR_step -1.)/TR_step)) - 1.0
                f_prime = (f2-f1)/step

                TR_new = TR_old - (f1/f_prime)
                eps2 = np.abs((TR_new - TR_old)/np.maximum(TR_old, 1e-5))
                if jj > 100:
                     raise RuntimeError('Max horizontal speed did not converge')

            g1 = (np.sqrt((TR_new**2.) - 1.)*weight_lbs) - .5*rho*(u_old**2.)*CD*S

            eps3 = 1
            kk = 0
            while eps3 > 1e-6:
                TR_old = TR_new
                TR_step = TR_new + step
                kk += 1

                f1 = (TR_old/TR_static) + ((u_step/ue)*(np.sqrt(TR_old -1.)/TR_old)) - 1.0
                f2 = (TR_step/TR_static) + ((u_step/ue)*(np.sqrt(TR_step -1.)/TR_step)) - 1.0
                f_prime = (f2-f1)/step

                TR_new = TR_old - (f1/f_prime)
                eps3 = np.abs((TR_new - TR_old)/np.maximum(TR_old, 1e-5))


                if kk > 100:
                    raise RuntimeError('Max horizontal speed did not converge')

            g2 = (np.sqrt((TR_new**2.)-1.)*weight_lbs) - .5*rho*(u_step**2.)*CD*S
            g_prime = (g2 - g1)/step

            u_new = u_old - g1/g_prime
            eps = np.abs((u_new - u_old)/np.maximum(1e-5, u_old))

            # print ii > 100

            if ii > 100:
                
                raise RuntimeError('Max horizontal speed did not converge')

        return u_new

    def compute_max_climb(self, batteryObj, motorObj, propObj, TR_static, weight, num_props = 4):
        ''' 
        This method computes the max climb rate of the vehicle based on simple equations of motion

        --- Assumptions ---
        Vertical component of thrust is equal to weight plus drag
        Velocity entering propeller is equal to climb rate
        exit velocity is equal to pitch velocity
        exit velocity is constant
        Assumes a CD = 1, Sref = .01 [m^2] (this is the value of the frame hub planform area)

        --- Inputs ---
        batteryObj: (sqlite object)
            object from database for a battery query
        motorObj: (sqlite object)
            object from database for a motor query
        propObj: (sqlite object)
            object from database for a prop query
        TR_static: (float)
            static thrust to weight ratio set by user
        weight: (float)
            weight of the vehicle [g]
        num_props: (int, default = 4)
            number of propellers on vehicle

        --- Returns ---
        u_new: (float)
            max vertical velocity of vehicle given the stated assumptions [ft/s]

        --- Exceptions ---
        RuntimeError:
            raised if solver fails to converge when computing velocity 
        '''

        pitch = propObj.pitch
        rho = .002378
        CD = 1.0
        S = .01*10.7639 #[sq ft]
        weight_lbs = weight*0.00220462 #[lbs]
        I, RPM = self.compute_current(batteryObj, motorObj, propObj, TR_static*weight/num_props, RPMout = True)
        ue = RPM*pitch*(1./12.)*(1./5280.)*60.

        u_new = 0.
        eps = 1.
        step = 1e-4

        ii = 0
        while eps > 1e-6:
            u_old = u_new
            u_step = u_new + step
            f1 = TR_static*weight_lbs*(1-(u_old/ue)) - .5*rho*(u_old**2.0)*CD*S
            f2 = TR_static*weight_lbs*(1-(u_step/ue)) - .5*rho*(u_step**2.0)*CD*S
            f_prime = (f2 - f1)/step

            u_new = u_old - (f1/f_prime)

            eps = np.abs((u_new - u_old)/np.maximum(u_old, 1e-5))

            ii += 1
            if ii > 100:
                raise RuntimeError('Max climb did not converge')

        return u_new

    def endurance_constraint(self, endurance):

        m, n = self.array.shape
        endurance_index = self.attribute_list.index('Endurance')
        endurance_array = self.results[:, endurance_index]

        remove = []
        for i in range(len(endurance_array)):
            if endurance_array[i] < (endurance):
                remove.append(i)

        new_array = np.delete(self.array, remove, axis = 0)
        new_results = np.delete(self.results, remove, axis = 0)

        self.array = new_array
        self.results = new_results

    def run_mission(self, batteryObj, motorObj, propObj, TR_static, weight, initial_current = 0, num_bat = 1., num_props = 4.):
        '''
        Performs mission analysis based on missions stored in mission_list attribute. Method analyzes each mission to compute the necessary
        charge of the battery to execute each mission. Returns the highest value of charge required to perform mission.

        --- Assumptions ---
        Includes all assumptions made in compute_max_speed and compute_max_climb
        Assumes that cruise sections and climb sections are done at max speed

        --- Inputs ---
        batteryObj: (sqlite object)
            object from database for a battery query
        motorObj: (sqlite object)
            object from database for a motor query
        propObj: (sqlite object)
            object from database for a prop query
        TR_static: (float)
            static thrust to weight ratio set by user
        weight: (float)
            weight of the vehicle [g]
        num_props: (int, default = 4)
            number of propellers on vehicle

        --- Returns ---
        Q: (float)
            Required charge battery must have to complete mission. [mAh]

         '''
        
        Q_vec = []

        for l in range(len(self.mission)):

            loiter_segs = self.mission[l].loiter 
            cruise_segs = self.mission[l].cruise
            climb_segs = self.mission[l].climb

            Q_req = 0.0 #[mAh]

            for time in loiter_segs:
                T_req = weight*1.1
                I_req = self.compute_current(batteryObj, motorObj, propObj, T_req/num_props, RPMout = False)
                Q_loiter = ((I_req + initial_current/4.0)*num_props)*time*(1000.0/60.0)

                Q_req = Q_req + Q_loiter

            for dist in cruise_segs:
                u_max = self.compute_max_speed(batteryObj, motorObj, propObj, TR_static, weight)
                t = dist/u_max/60.
                I = self.compute_current(batteryObj, motorObj, propObj, TR_static*weight/num_props, RPMout = False)
                Q_cruise = ((I + initial_current/4.0)*num_props)*t*(1000/60.)

                Q_req = Q_req + Q_cruise

            for dist in climb_segs:
                max_climb = self.compute_max_climb(batteryObj, motorObj, propObj, TR_static, weight)
                t = dist/max_climb/60.
                I = self.compute_current(batteryObj, motorObj, propObj, TR_static*weight/num_props, RPMout = False)
                Q_climb = ((I + initial_current/4.0)*num_props)*t*(1000.0/60.0)

                Q_req = Q_req + Q_climb

            Q_vec.append(Q_req)

        return np.amax(Q_vec)

    def mission_constraint(self, SF = .71):
        ''' 
        Applies the battery charge constraint that was computed in run_mission. Scans through results array and pulls charge required for each
        design. Then identifies the battery used in each design and compares the charges. If the required charge is greater than the charge of the
        battery multiplied by a safety factor, then that design is eliminated

        --- Inputs ---
        SF: (float, default = .71)
            Safety factor multplied by battery charge
        '''

        m,n = self.array.shape
        i = self.modules.index('Battery')
        bat_start = np.sum(self.levels[:i])
        bat_cols = range(bat_start, bat_start + self.levels[i])

        charge_index = self.attribute_list.index('Charge')
        charge_array = self.results[:, charge_index]

        bat_reduced = self.array[:, bat_cols]

        remove = []
        for i in range(len(bat_reduced)):
            row = bat_reduced[i]
            for j in range(len(row)):
                if row[j] == 1:
                    bat_num = j
                    break

            Q = charge_array[i]
            Q_bat = self.query.battery[bat_num].charge
            if ((Q_bat*SF) < Q):
                remove.append(i)

        self.array = np.delete(self.array, remove, axis = 0)
        self.results = np.delete(self.results, remove, axis = 0) 


    def thrust_constraint(self, TR_static, num_props = 4):
        ''' 
        Applies the thrust constraint that was computed in run_mission. Tests every possible compination of battery, motor, and prop from the 
        query attribute of the object. Each combination is analyzed to compute the voltage and current necessary to produce the desired thrust
        for the entire vehicle. If the required current exceeds the max current of the motor, or if the required voltage exceeds the maximum
        voltage of the battery, then the design is eliminated

        --- Inputs ---
        T: (float)
            Thrust that must be generated by entire vehcile
        num_props: (int, default = 4)
            number of propellers on vehcile

        --- Exceptions ---
        RuntimeError:
            raised if convergence fails when computing RPM
        '''

        m,n = self.array.shape
        i = self.modules.index('Motor')
        motor_start = np.sum(self.levels[:i])
        motor_cols = range(motor_start, motor_start + self.levels[i])
        i = self.modules.index('Prop')
        prop_start = np.sum(self.levels[:i])
        prop_cols = range(prop_start, prop_start + self.levels[i])
        i = self.modules.index('Battery')
        bat_start = np.sum(self.levels[:i])
        bat_cols = range(bat_start, bat_start + self.levels[i])

        weight_index = self.attribute_list.index('Weight')
        weight_array = self.results[:, weight_index]

        # T = T/num_props
        # omit = []
        # i = 0
        # j = 0
        # k = 0

        # for bat in self.query.battery:
        #     i = 0

        #     for mot in self.query.motor:
        #         j = 0

        #         for prop in self.query.prop:

        #             #Unpack Prop Info
        #             D = prop.diameter
        #             pitch = prop.pitch
        #             data_folder = prop.data

        #             #Unpack Motor Info
        #             kv = mot.kv
        #             I_max = mot.max_current
        #             I0 = mot.I0
        #             Rm = mot.Rm
        #             kt = 1352.4/kv
        #             rho = .002378

        #             #Unpack Battery Info
        #             V_in = bat.volts

        #             #Evaluate Required current
        #             if data_folder != 'null':
        #                 n_list, Ct_list = np.loadtxt(data_folder + 'n_vs_ct.txt', skiprows = 1, unpack = True) #Unpack info from raw data
        #                 fit1 = np.polyfit(n_list, Ct_list, 1)
        #                 p = np.poly1d(fit1)

        #                 n_new = 200.0
        #                 Ct = 1.0

        #                 eps = 1.

        #                 ii = 1
        #                 while eps > 1e-6:
                            
        #                     n_old = n_new
        #                     # Ct = p(n_old)
        #                     # n_new = np.sqrt(T/(rho*Ct*((D/12.)**4.0)))

        #                     h = 1e-4
        #                     f1 = p(n_old) - (T/(rho*(n_old**2.)*((D/12.)**4.)))
        #                     n_h = n_old + h
        #                     f2 =  p(n_h) - (T/(rho*(n_h**2.)*((D/12.)**4.)))
        #                     f_prime = (f2-f1)/h

        #                     n_new = n_old - (f1/f_prime)

        #                     eps = np.abs((n_new - n_old)/n_old)
        #                     ii += 1

        #                     if ii > 100:
        #                         print 'max iter reached'
        #                         print 'n = %f ' % n_new
        #                         print 'esps = %f' % eps
        #                         raise RuntimeError('RPM computation did not converge')


        #                 n_list2, Cq_list = np.loadtxt(data_folder + 'n_vs_cq.txt', skiprows = 1, unpack = True)
        #                 fit2 = np.polyfit(n_list2, Cq_list, 1)
        #                 p2 = np.poly1d(fit2)

        #                 Cq = p2(n_new)
        #                 tau = Cq*rho*(n_new**2.0)*((D/12.)**5.)
        #                 rpm = n_new*60.

        #                 I_req = (tau/kt) + I0
        #                 V_req = (rpm/kv) + I_req*Rm

        #                 if not ((I_req< I_max) and (V_req < V_in)):
        #                     omit.append([i,j,k])

        #             j += 1

        #         i += 1

        #     k += 1

        # k = 0

        motor_reduced = self.array[:, motor_cols]
        prop_reduced = self.array[:, prop_cols]
        bat_reduced = self.array[:, bat_cols]
        
        remove = []
        for i in range(m):
            for j in range(len(motor_cols)):
                if motor_reduced[i][j] == 1:
                    mot = self.query.motor[j]
                    break
            for j in range(len(bat_cols)):
                if bat_reduced[i][j] == 1:
                    bat = self.query.battery[j]
                    break
            for j in range(len(prop_cols)):
                if prop_reduced[i][j] == 1:
                   prop = self.query.prop[j]
                   break

            weight = weight_array[i]
            T = (TR_static*weight)/num_props

            #Unpack Prop Info
            D = prop.diameter
            pitch = prop.pitch
            data_folder = prop.data

            #Unpack Motor Info
            kv = mot.kv
            I_max = mot.max_current
            I0 = mot.I0
            Rm = mot.Rm
            kt = 1352.4/kv
            rho = .002378

            #Unpack Battery Info
            V_in = bat.volts

            #Evaluate Required current
            if data_folder != 'null':
                n_list, Ct_list, Cq_list = np.loadtxt(data_folder, delimiter = ',', skiprows = 1, unpack = True) #Unpack info from raw data
                fit1 = np.polyfit(n_list, Ct_list, 1)
                p = np.poly1d(fit1)

                n_new = 200.0
                Ct = 1.0

                eps = 1.

                ii = 1
                while eps > 1e-6:
                    
                    n_old = n_new
                    # Ct = p(n_old)
                    # n_new = np.sqrt(T/(rho*Ct*((D/12.)**4.0)))

                    h = 1e-4
                    f1 = p(n_old) - (T/(rho*(n_old**2.)*((D/12.)**4.)))
                    n_h = n_old + h
                    f2 =  p(n_h) - (T/(rho*(n_h**2.)*((D/12.)**4.)))
                    f_prime = (f2-f1)/h

                    n_new = n_old - (f1/f_prime)

                    eps = np.abs((n_new - n_old)/n_old)
                    ii += 1

                    if ii > 100:
                        print 'max iter reached'
                        print 'n = %f ' % n_new
                        print 'esps = %f' % eps
                        raise RuntimeError('RPM computation did not converge')


                # n_list2, Cq_list = np.loadtxt(data_folder + 'n_vs_cq.txt', skiprows = 1, unpack = True)
                fit2 = np.polyfit(n_list, Cq_list, 1)
                p2 = np.poly1d(fit2)

                Cq = p2(n_new)
                tau = Cq*rho*(n_new**2.0)*((D/12.)**5.)
                rpm = n_new*60.

                I_req = (tau/kt) + I0
                V_req = (rpm/kv) + I_req*Rm

                if not ((I_req< I_max) and (V_req < V_in)):
                    remove.append(i)


        # for a in range(m):
        #     for ii in range(len(motor_cols)):
        #         if motor_reduced[a][ii] == 1:
        #             i = ii

        #     for jj in range(len(prop_cols)):
        #         if prop_reduced[a][jj] == 1:
        #             j = jj

        #     for kk in range(len(bat_cols)):
        #         if bat_reduced[a][kk] == 1:
        #             k = kk

        #     if ([i,j,k] in omit):
        #         remove.append(a)

        
        self.array = np.delete(self.array, remove, axis = 0)
        self.results = np.delete(self.results, remove, axis = 0)

        # return self

    def esc_constraint(self, SF = 1.2):
        ''' '''
        m, n = self.array.shape
        i = self.modules.index('ESC')
        esc_start = np.sum(self.levels[:i])
        esc_cols = range(esc_start, esc_start + self.levels[i])
        i = self.modules.index('Motor')
        mot_start = np.sum(self.levels[:i])
        mot_cols = range(mot_start, mot_start + self.levels[i])

        omit = []
        i = 0
        j = 0
        for esc in self.query.esc:
            j = 0
            for mot in self.query.motor:
                esc_max_current = esc.max_constant_current
                mot_max_current = mot.max_current

                if esc_max_current > (mot_max_current * SF):
                    omit.append([i,j])
                j += 1

            i += 1

        esc_reduced = self.array[:, esc_cols]
        mot_reduced = self.array[:, mot_cols]

        remove = []
        for k in range(m):
            for ii in range(len(esc_cols)):
                if esc_reduced[k][ii]:
                    i = ii

            for jj in range(len(mot_cols)):
                if mot_reduced[k][jj]:
                    j = jj

            if ([i,j] in omit):
                remove.append(k)

        self.array = np.delete(self.array, remove, axis = 0)
        self.results = np.delete(self.results, remove, axis = 0)


    def frame_constraint(self, SF = 1.1):
        ''' '''

        m,n = self.array.shape
        i = self.modules.index('Frame')
        frame_start = np.sum(self.levels[:i])
        frame_cols = range(frame_start, frame_start + self.levels[i])
        i = self.modules.index('Prop')
        prop_start = np.sum(self.levels[:i])
        prop_cols = range(prop_start, prop_start + self.levels[i]) 

        omit = []
        i = 0
        j = 0

        for fram in self.query.frame:
            j = 0
            for prop in self.query.prop:

                length = fram.length    #Unpack frame length
                D = prop.diameter       #Unpack prop diameter

                if (D*SF > length):
                    omit.append([i,j])

                j += 1

            i += 1

        frame_reduced = self.array[:, frame_cols]
        prop_reduced = self.array[:, prop_cols]

        remove = []

        for k in range(m):
            for ii in range(len(frame_cols)):
                if frame_reduced[k][ii] == 1:
                    i = ii

            for jj in range(len(prop_cols)):
                if prop_reduced[k][jj] == 1:
                    j = jj

            if ([i,j] in omit):
                remove.append(k)

        self.array = np.delete(self.array, remove, axis = 0)
        self.results = np.delete(self.results, remove, axis = 0)

    def prop_motor_incompatibility(self):
        ''' '''

        propObj = self.query.prop
        motorObj = self.query.motor

        incompatible_list = []

        for prop in propObj:

            if prop.shaft_diameter == 0.0:
                continue

            for mot in motorObj:

                if mot.shaft_diameter != prop.shaft_diameter:
                    incompatible_list.append([mot.name, prop.name])

        self.incompatible_list = incompatible_list



    def apply_compatibility(self, incompatible, self_compatible = True):
        '''Removes incompatible cases from consideration '''

        modules = self.modules
        levels = self.levels
        namelist = self.namelist
        main_matrix = self.compatibility
        design = self.array

        if self_compatible:
            for i in range(len(levels)):
                base = np.sum(levels[:i])
                box_len = levels[i]
                main_diag = [1]*box_len
                diag_mat = np.zeros((box_len, box_len), dtype = np.int)
                np.fill_diagonal(diag_mat, main_diag)

                main_matrix[base:base + box_len, base:base + box_len] = diag_mat

        for i in range(len(incompatible)):

            name1, name2 = incompatible[i]

            for mod1 in modules:
                if name1 in namelist[mod1]:
                    j1 = namelist[mod1].index(name1)

                    break

            for mod2 in modules:
                if name2 in namelist[mod2]:
                    j2 = namelist[mod2].index(name2)
                    break

            mod_num1 = modules.index(mod1)
            mod_num2 = modules.index(mod2)

            base1 = np.sum(levels[:mod_num1])
            base2 = np.sum(levels[:mod_num2])

            main_matrix[base1 + j1][base2 + j2] = 0
            main_matrix[base2 + j2][base1 + j1] = 0

        self.compatibility = main_matrix

        index_list = []
        for i in range(len(main_matrix)):
            for j in range(i, len(main_matrix[i])):
                val = main_matrix[i][j]
                if val == 0:
                    index_list.append([i,j])

        remove = []
        for pair in index_list:
            val1, val2 = pair
            i = 0
            for row in design:
                if (row[val1] == 1 and row[val2] == 1):
                    remove.append(i)
                i += 1

        new_design = np.delete(design, remove, axis = 0)
        new_results = np.delete(design, remove, axis =0)

        self.array = new_design
        # self.results = new_results

    def run_topsis(self, scaling_array, decision_array):

        data_array = self.results

        m,n = data_array.shape

        if m == 0:
            print 'No feasible design space, check constraints'
            sys.exit()
        elif m == 1:
            print 'Constraint analysis produced 1 feasible design'
            self.print_results(0)
            sys.exit()
        else:
            print 'Constraint analysis produced %d feasible designs' % m

        norm_array = np.zeros((1,n), dtype = np.float64)
        pos_ideal = np.zeros((1,n), dtype = np.float64)
        neg_ideal = np.zeros((1,n), dtype = np.float64)
        pos_dist = np.zeros((1,n), dtype = np.float64)
        neg_dist = np.zeros((1,n), dtype = np.float64)
        closeness = np.zeros((1,m), dtype = np.float64)

        trans = data_array.T
        for i in range(n):
            norm_array[0][i] = np.linalg.norm(trans[i])
            
        norm_data = data_array/norm_array

        if np.sum(scaling_array) != 1.0:
            raise ValueError('Sum of scaling array must equal 1')

        norm_data = norm_data*scaling_array
        trans = norm_data.T

        for i in range(n):
            if decision_array[i] == 1:

                pos_ideal[0][i] = np.max(trans[i])
                neg_ideal[0][i] = np.min(trans[i])

            elif decision_array[i] == 0:

                pos_ideal[0][i] = np.min(trans[i])
                neg_ideal[0][i] = np.max(trans[i])

            else:
                raise ValueError('Decision Array can only contain 1 (maximize) or 0 (minimize)')

        for j in range(len(norm_data)):
            for i in range(n):

                pos_dist[0][i] = np.sqrt(np.sum((norm_data[j] - pos_ideal)**2.0))
                neg_dist[0][i] = np.sqrt(np.sum((norm_data[j] - neg_ideal)**2.0))
                closeness[0][j] = neg_dist[0][i]/(neg_dist[0][i] + pos_dist[0][i])

        ideal = np.argmax(closeness)

        print 'Run Successful!'
        print '--- TOPSIS Results ---'
        self.print_results(ideal)
        # return (ideal, data_array[ideal])

    def print_results(self, row_num):
        
        row = self.array[row_num]
        results_row = self.results[row_num]
        namelist = self.namelist
        modules = self.modules
        attribute_list = self.attribute_list

        full_list = []
        for i in range(len(modules)):
            mod_list = []
            mod = modules[i]
            mod_start = np.sum(self.levels[:i])
            mod_cols = range(mod_start, mod_start + self.levels[i])
            mod_reduced = row[mod_cols]

            for k in range(len(mod_cols)):
                if mod_reduced[k] == 1:
                    mod_list.append(namelist[mod][k])

            full_list.append(mod_list)

        for i in range(len(modules)):
            print '%s : %s' % (modules[i], full_list[i])

        for i in range(len(attribute_list)):
            print '%s : %f' % (attribute_list[i], results_row[i])


if __name__ == '__main__':

    db = SqliteDatabase('../database/continuum.db')
    db.connect()

    motor_query = Motor.select()#.where(Motor.name == 'MT-1806').get()
    battery_query = Battery.select()#.where(Battery.name == 'Zippy1').get()
    prop_query = Prop.select()#.where(Prop.name == 'Gemfan1').get()

    data_set = QueryObj(battery_query, motor_query, prop_query)

    level_array = np.array([len(battery_query), len(motor_query), len(prop_query)], dtype = np.int)
    # coded_array = coded_matrix(level_array)
    module_list = ['Battery', 'Motor', 'Prop']

    scaling_array = np.array([.4, .6])
    decision_array = np.array([1,0])

    test_array = CodedObj(level_array, module_list, data_set)

    # test_array.evaluate_cases()
    # test_array.endurance_constraint(10.)
    # test_array.thrust_constraint(200.)
    # test_array.run_topsis(scaling_array, decision_array)
    # a = (('Gemfan 6x3', 'Zippy2'), ('Gemfan 6x3', 'Zippy1'))
    # a = [['Gemfan 6x3', 'Zippy2'], ['Gemfan 6x3', 'Zippy1']]
    # test_array.apply_compatibility(a, self_compatible = True)
    # print test_array.compatibility
    # print test_array.results
    # test_array.prop_motor_incompatibility()
    # print test_array.incompatible_list

    test_mission = MissionProfile(name = 'Test1')
    test_mission.add_element('Climb', 50.)
    test_mission.add_element('Cruise', 500.)
    test_mission.add_element('Loiter', 5.0)
    test_mission.add_element('Cruise', 500.)
    test_array.add_mission(test_mission)
    test_array.evaluate_cases(TR_static = 2.0, initial_weight = 200.)
    test_array.mission_constraint()
    m,n = test_array.array.shape
    print m
    test_array.thrust_constraint(2.0)
    m,n = test_array.array.shape



    # mot = Motor.select().where(Motor.name == 'MT-1806').get()
    # bat = Battery.select().where(Battery.name == 'Zippy1').get()
    # prop = Prop.select().where(Prop.name == 'Gemfan 5x3').get()

    # u =  test_array.compute_max_speed(bat, mot, prop, 2.0, 465.)
    # I, RPM = test_array.compute_current(bat, mot, prop, 465./4, RPMout = True)
    # print I
    # print RPM/60.
    
    # print u

    
    # test_mission1 = MissionProfile(name = 'Test1')
    # test_mission2 = MissionProfile(name = 'Test2')
    # test_array.add_mission(test_mission1)
    # test_array.add_mission(test_mission2)
    # for i in test_array.mission:
    #     print i.name

    # test_array.compute_current(bat, mot, prop, 300.)



