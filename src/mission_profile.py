'''Mission profile object stores all legs of the mission. The mission allows loiter, horizontal cruie, and vertical climb elements of the mission.
These legs will be considered in the main analysis when sizing the vehicle '''

class MissionProfile(object):

    def __init__(self, vehicle_type = 'QuadCopter', name = None):
        self.vehicle_type = vehicle_type
        self.name = name
        self.loiter = []
        self.climb = []
        self.cruise = []

    def add_element(self, phase_name, level):
        ''' This method adds phases to the mission profile. 
        --- Inputs ---
            phase_name (str):
                phase name corresponds to the name of the segment. Allowable values are 'Cruie', 'Climb', and 'Loiter.'
            level (float):
                level corresponds to the level of the mission phase. For loiter, level is the loiter time in minutes. For Climb, level is the 
                climb distance in ft. For Cruise, the level is cruise distance in ft.

         '''

        if self.vehicle_type == 'QuadCopter':

            if phase_name == 'Loiter':
                self.loiter.append(level)
            elif phase_name == 'Climb':
                self.climb.append(level)
            elif phase_name == 'Cruise':
                self.cruise.append(level)
            else:
                raise ValueError('Not an acceptable mission phase name')

if __name__ == '__main__':
    test_mission = MissionProfile()
    test_mission.add_element('Loiter', 5.0)
    test_mission.add_element('Climb', 100.0)
    test_mission.add_element('Cruise', 1000.0)
    test_mission.add_element('Loiter', 3.0)

    print test_mission.loiter
    print test_mission.climb
    print test_mission.cruise

