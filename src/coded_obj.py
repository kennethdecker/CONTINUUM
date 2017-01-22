import numpy as np
from peewee import *
from create_database import *
import sys
from query_obj import QueryObj
from coded_matrix import coded_matrix

class CodedObj(object):
    def __init__(self, level_array, module_list, QueryObj):

        self.levels = level_array
        self.modules = module_list
        self.query = QueryObj
        self.coded_matrix()


    def coded_matrix(self):
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



    def evaluate_cases(self):

        m,n = self.array.shape
        attribute_list = ['Weight', 'Efficiency', 'Current', 'Endurance']
        results = np.zeros((m, len(attribute_list)), dtype = np.float64)
        modules = len(self.levels)

        if np.sum(self.levels) != n:
            raise ValueError('Sum of level array must equal number of clumns of coded matrix')
        b = 0
        for i in range(m):

            row = self.array[i]

            weight = 0
            col = 0

            for j in range(modules):

                num = self.levels[j]

                for k in range(num):

                    if self.modules[j] == 'Battery':
                        weight = weight + self.array[i][col] * self.query.battery[k].weight

                        if self.array[i][col] == 1:
                            battery = self.query.battery[k]

                    elif self.modules[j] == 'Motor':
                        weight = weight + self.array[i][col] * self.query.motor[k].weight
                        
                        if self.array[i][col] == 1:
                            motor = self.query.motor[k]

                    elif self.modules[j] == 'Prop':
                        weight = weight + self.array[i][col] * self.query.prop[k].weight
                        
                        if self.array[i][col] == 1:
                            prop = self.query.prop[k]

                    elif self.modules[j] == 'ESC':
                        weight = weight + self.array[i][col] * self.query.esc[k].weight

                    elif self.modules[j] == 'Frame':
                        weight = weight + self.array[i][col] * self.query.frame[k].weight

                    col = col + 1

            efficiency = self.compute_efficiency(battery, motor, prop)
            I = self.compute_current(battery, motor, prop, weight)
            endurance = self.compute_endurance(battery, I)

            if np.isnan(I):
                b += 1

            # results[i][0] = weight
            # results[i][1] = efficiency
            for j in range(len(attribute_list)):
                if attribute_list[j] == 'Weight':
                    results[i][j] = weight
                elif attribute_list[j] == 'Efficiency':
                    results[i][j] = efficiency
                elif attribute_list[j] == 'Current':
                    results[i][j] = I
                elif attribute_list[j] == 'Endurance':
                    results[i][j] = endurance

        self.attribute_list = attribute_list
        self.results = results
        print '%d Current computaions failed to converge' % b

        # return results

    def compute_efficiency(self, batteryObj, motorObj, propObj):

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

    def compute_current(self, batteryObj, motorObj, propObj, T):
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
            n_list, Ct_list = np.loadtxt(data_folder + 'n_vs_ct.txt', skiprows = 1, unpack = True) #Unpack info from raw data
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

                eps = np.abs((n_new - n_old)/n_old)
                i += 1

                if i > 200:
                    print 'max iter reached'
                    print 'n = %f ' % n_new
                    print 'eps = %f' % eps
                    # sys.exit()


            n_list2, Cq_list = np.loadtxt(data_folder + 'n_vs_cq.txt', skiprows = 1, unpack = True)
            fit2 = np.polyfit(n_list2, Cq_list, 1)
            p2 = np.poly1d(fit2)

            Cq = p2(n_new)
            tau = Cq*rho*(n_new**2.0)*((D/12.)**5.)
            rpm = n_new*60.

            I_req = (tau/kt) + I0
            V_req = (rpm/kv) + I_req*Rm
            
            return I_req

    def compute_endurance(self, batteryObj, I, SF = .71):
        Q = batteryObj.charge*(60.0/1000.0)
        t = (Q*SF)/I

        return t

    def endurance_constraint(self, endurance, power, SF = .71):

        m,n = self.array.shape
        i = self.modules.index('Battery')
        start_col = np.sum(self.levels[:i])
        cols = range(start_col, start_col + self.levels[i])

        omit = [];
        i = 0
        for bat in self.query.battery:
            V = bat.volts
            Q = bat.charge
            I = power/V

            if (endurance/SF) > (Q/I):
                omit.append(i)

            i += 1

        reduced = self.array[:,cols]
        remove = []
        i = 0
        for j in range(m):
            if (1 in reduced[j][omit]):
                remove.append(i)
            i += 1

        self.array = np.delete(self.array, remove, axis = 0)

        # return self

    def thrust_constraint(self, T):

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

        omit = []
        i = 0
        j = 0
        k = 0

        for bat in self.query.battery:

            for mot in self.query.motor:

                for prop in self.query.prop:

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
                        n_list, Ct_list = np.loadtxt(data_folder + 'n_vs_ct.txt', skiprows = 1, unpack = True) #Unpack info from raw data
                        fit1 = np.polyfit(n_list, Ct_list, 1)
                        p = np.poly1d(fit1)

                        n_new = 200.0
                        Ct = 1.0

                        eps = 1.

                        i = 1
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
                            i += 1

                            if i > 100:
                                print 'max iter reached'
                                print 'n = %f ' % n_new
                                print 'esps = %f' % eps
                                sys.exit()


                        n_list2, Cq_list = np.loadtxt(data_folder + 'n_vs_cq.txt', skiprows = 1, unpack = True)
                        fit2 = np.polyfit(n_list2, Cq_list, 1)
                        p2 = np.poly1d(fit2)

                        Cq = p2(n_new)
                        tau = Cq*rho*(n_new**2.0)*((D/12.)**5.)
                        rpm = n_new*60.

                        I_req = (tau/kt) + I0
                        V_req = (rpm/kv) + I_req*Rm

                        if not ((I_req< I_max) and (V_req < V_in)):
                            omit.append([i,j,k])

                    j += 1

                i += 1

            k += 1

        # k = 0
        motor_reduced = self.array[:, motor_cols]
        prop_reduced = self.array[:, prop_cols]
        bat_reduced = self.array[:, bat_cols]
        remove = []

        for a in range(m):
            for ii in range(len(motor_cols)):
                if motor_reduced[a][ii] == 1:
                    i = ii

            for jj in range(len(prop_cols)):
                if prop_reduced[a][jj] == 1:
                    j = jj

            for kk in range(len(bat_cols)):
                if bat_reduced[a][kk] == 1:
                    k = kk

            if ([i,j,k] in omit):
                remove.append(a)


        self.array = np.delete(self.array, remove, axis = 0)

        # return self

    def frame_constraint(self, SF = 1.1):

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

        

    def run_topsis(self, scaling_array, decision_array):

        data_array = self.results

        m,n = data_array.shape

        norm_array = np.zeros((1,n), dtype = np.float64)
        pos_ideal = np.zeros((1,n), dtype = np.float64)
        neg_ideal = np.zeros((1,n), dtype = np.float64)
        pos_dist = np.zeros((1,n), dtype = np.float64)
        neg_dist = np.zeros((1,n), dtype = np.float64)
        closeness = np.zeros((1,n), dtype = np.float64)

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

        for i in range(n):

            pos_dist[0][i] = np.sqrt(np.sum((norm_data[i] - pos_ideal)**2.0))
            neg_dist[0][i] = np.sqrt(np.sum((norm_data[i] - neg_ideal)**2.0))
            closeness[0][i] = neg_dist[0][i]/(neg_dist[0][i] + pos_dist[0][i])

        ideal = np.argmax(closeness)

        return (ideal, data_array[ideal])








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
    # test_array.endurance_constraint(10., 100.)
    # test_array.thrust_constraint(200.)
    # test_array.run_topsis(scaling_array, decision_array)
    # print test_array.results

    # mot = Motor.select().where(Motor.name == 'NTM PropDrive 28-36').get()
    # bat = Battery.select().where(Battery.name == 'Zippy3').get()
    # prop = Prop.select().where(Prop.diameter == 12.0).get()

    # test_array.compute_current(bat, mot, prop, 300.)



