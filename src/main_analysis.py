import numpy as np
from peewee import *
from create_database import *
import sys
from coded_obj import CodedObj
from query_obj import QueryObj
from mission_profile import MissionProfile


def main():
    #Define design variables and mission elements
    TqW = 1.5
    payload_weight = 121. #[g]
    payload_current = 0. #[A]
    mission = MissionProfile()
    mission.add_element('Climb', 50.)
    mission.add_element('Cruise', 100.)
    mission.add_element('Loiter', 10.)
    mission.add_element('Cruise', 100.)

    #Query Database
    db = SqliteDatabase('../database/continuum.db')
    db.connect()

    motor_query = Motor.select()#.where(Motor.name == 'MT-1806').get()
    battery_query = Battery.select()#.where(Battery.name == 'Zippy1').get()
    prop_query = Prop.select()#.where(Prop.name == 'Gemfan1').get()
    # frame_query = Frame.select()

    data_set = QueryObj(battery_query, motor_query, prop_query)

    #Set full factorial study values
    level_array = np.array([len(battery_query), len(motor_query), len(prop_query)], dtype = np.int)
    module_list = ['Battery', 'Motor', 'Prop']

    design_array = CodedObj(level_array, module_list, data_set)

    #Set TOPSIS parameters
    attribute_list = ['Weight', 'Efficiency', 'Current', 'Endurance', 'Charge']
    decision_array = np.array([0, 1, 0, 1, 0])
    scaling_array = np.array([.5, .1, .2, .1, .1])

    #Set incompatibility List
    incompatible = (
        ('NTM PropDrive 28-26', 'Gemfan 5x3'),
        ('NTM PropDrive 28-36', 'Gemfan 5x3')
        )

    #Apply constraints and perform topsis
    dead_weight = 418.
    initial_weight = payload_weight + dead_weight

    design_array.add_mission(mission)
    design_array.apply_compatibility(incompatible)
    design_array.evaluate_cases(TqW, initial_weight = initial_weight, initial_current = payload_current, num_props =  4)
    design_array.mission_constraint(SF = .8)
    # design_array.esc_constraint()
    # design_array.frame_constraint()
    design_array.thrust_constraint(TqW)
    design_array.run_topsis(scaling_array, decision_array)


if __name__ == '__main__':

    main()