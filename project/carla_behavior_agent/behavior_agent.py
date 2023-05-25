# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import random
import numpy as np
import carla
from basic_agent import BasicAgent
from local_planner import RoadOption
from behavior_types import Cautious, Aggressive, Normal

from misc import get_speed, positive, is_within_distance, compute_distance

class BehaviorAgent(BasicAgent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment such as tailgating avoidance.
    Adding to these are possible behaviors, the agent can also keep safety distance
    from a car in front of it by tracking the instantaneous time to collision
    and keeping it in a certain range. Finally, different sets of behaviors
    are encoded in the agent, from cautious to a more aggressive ones.
    """

    def __init__(self, vehicle, behavior='normal', opt_dict={}, map_inst=None, grp_inst=None):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param behavior: type of agent to apply
        """

        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._look_ahead_steps = 0

        # Vehicle information
        self._speed = 0
        self._speed_limit = 0
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = None
        self._sampling_resolution = 2

        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()

    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

        self._overake_coverage-=1
        
        """self._behavior.braking_distance = (self._speed/10)**2
        print("Current velocity: ",self._speed,", Security distance: ", self._behavior.braking_distance)"""

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected

    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        
        Gestisce i cambi di corsia tenendo un considerazione i veicoli che vengono da dietro e uscita dal parcheggio
        """

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        behind_vehicle_state, behind_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max(
            self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, low_angle_th=160) # cambiano gli angolo rispetto alle altre chiamate perchè si cercano veicoli in questa determinata direzione -> angolo posteriore sinistro
        if behind_vehicle_state and self._speed < get_speed(behind_vehicle):    #se il veicolo si trova dietro e la velocità è minore della nostra (non sopraggiungono veicoli)
            if (right_turn == carla.LaneChange.Right or right_turn ==
                    carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving: 
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1) # viene fatto un controllo su una possibile collisione con altri veicoli (curva a destra)
                if not new_vehicle_state:
                    print("Tailgating, moving to the right!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         right_wpt.transform.location)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the left!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         left_wpt.transform.location)

    def collision_and_car_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """

        vehicle_list = self._world.get_actors().filter("*vehicle*")
        static_ob_list = self._world.get_actors().filter("*static*")

        def dist(v): return v.get_location().distance(waypoint.transform.location)

        safety_distance = max(45, 2*(get_speed(self._vehicle)/10)**2)

        ob_list = [v for v in vehicle_list if dist(v) < safety_distance and v.id != self._vehicle.id]
        for v in static_ob_list:
            if dist(v) < safety_distance:
                ob_list.append(v)

        ob_list.sort(key=lambda x:dist(x))
        
        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected( #object_obstacle_detected in realtà
                ob_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                ob_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                ob_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)
            
            # Check for tailgating: utile quando si parte da bordo strada e ci si deve intromettere
            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self._speed > 10 \
                    and self._behavior.tailgate_counter == 0:
                print("tailgating from collision and car avoidance manager")
                self._tailgating(waypoint, ob_list)

        return vehicle_state, vehicle, distance, ob_list

    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        def dist(w): return w.get_location().distance(waypoint.transform.location)
        walker_list = [w for w in walker_list if dist(w) < 10]      #considera solo i pedoni a distanza minore di 10

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)

        return walker_state, walker, distance

    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """

        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)   #differenza di velocità tra noi e il viecolo
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)   #tempo per raggiungere il veicolo (time to collision)

        # Under safety time distance, slow down.
        if self._behavior.safety_time > ttc > 0.0:
            print("Under safety time distance")
            target_speed = min([
                positive(vehicle_speed - self._behavior.speed_decrease), # riduco la velocità di self._behavior.speed_decrease
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            #control = self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            print("Actual safety time distance")
            target_speed = min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            #control = self._local_planner.run_step(debug=debug)

        # Normal behavior.
        else:
            print("Normal behavior")
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
        
        print("Target speed after car_following_manager: ", target_speed)
        control = self._local_planner.run_step(debug=debug)

        return control

    def overtake_manager(self, lane_offset = 1 ):

        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())
        
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        static_ob_list = self._world.get_actors().filter("*static*")

        horizon = 100

        # list of object to overtake
        ob_list = [v for v in static_ob_list if is_within_distance(v.get_transform(), self._vehicle.get_transform(), horizon, [0,60])     # se l'ggetto è avanti a noi ad una distanza massima
                                                and v.id != self._vehicle.id
                                                and ego_wpt.lane_id == self._map.get_waypoint(v.get_transform().location, lane_type=carla.LaneType.Any).lane_id]      # se l'oggetto è nella nostra stessa corsia
        
        def dist(v): return v.get_location().distance(self._vehicle.get_transform().location)
        ob_list.sort(key=lambda x:dist(x))

        # search for the first location in which perform the reentry
        search_for_reentry = True
        safety_distance_for_reentry = self._vehicle.bounding_box.extent.x*2 + self._behavior.safety_space_reentry
        i = 0
        while search_for_reentry:
            if i == len(ob_list)-1:
                break
            ob1_length = ob_list[i].bounding_box.extent.x
            ob2_length = ob_list[i+1].bounding_box.extent.x
            distance_between_objects = ob_list[i].get_location().distance(ob_list[i+1].get_location()) - (ob1_length+ob2_length) 
            if distance_between_objects > safety_distance_for_reentry:
                break
            i+=1
            
        other_line_distance = ob_list[i].get_location().distance(self._vehicle.get_location()) + ob_list[i].bounding_box.extent.x + self._behavior.safety_space_reentry/2
        
        print("distance from the last obstacle: ", dist(ob_list[-1]))
        
        print("Distance for overtake: ", other_line_distance)

        target_line_id =  ego_wpt.lane_id + lane_offset
        target_line_id = target_line_id if target_line_id != 0 else target_line_id + 1      # 0 is the central lane

        # list of vehicle on the lane in wihch we have to move
        ob_list = [v for v in vehicle_list if is_within_distance(v.get_transform(), self._vehicle.get_transform(), horizon, [0,90])     # se l'ggetto è avanti a noi ad una distanza massima
                                                and v.id != self._vehicle.id
                                                and target_line_id == self._map.get_waypoint(v.get_transform().location, lane_type=carla.LaneType.Any).lane_id]      # se l'oggetto è nella corsia desiderata
                        
        ob_list.sort(key=lambda x:dist(x))

        other_line_time = other_line_distance/(self._behavior.max_speed/2)  #time spent on the other line is estimed to be calculated on target velocity/2
        
        if len(ob_list) == 0:
            return True, other_line_time
        target_vehicle = ob_list[0]
        target_vehicle_distance = dist(target_vehicle)

        if target_vehicle_distance > other_line_distance:
            target_vehicle_velocity = get_speed(target_vehicle)
            target_vehicle_time = (target_vehicle_distance-other_line_distance)/target_vehicle_velocity
            print("target_vehicle_time: ", target_vehicle_time)
            print("other_line_time: ", other_line_time)
            if target_vehicle_time > other_line_time:
                return True, other_line_distance

        return False, 0
    def overtake_manager_ver2(self, lane_offset = 1 ):

        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())
        
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        static_ob_list = self._world.get_actors().filter("*static*")

        horizon = 100

        # list of object to overtake
        ob_list = [v for v in static_ob_list if is_within_distance(v.get_transform(), self._vehicle.get_transform(), horizon, [0,60])     # se l'ggetto è avanti a noi ad una distanza massima
                                                and v.id != self._vehicle.id
                                                and ego_wpt.lane_id == self._map.get_waypoint(v.get_transform().location, lane_type=carla.LaneType.Any).lane_id]      # se l'oggetto è nella nostra stessa corsia
        
        def dist(v): return v.get_location().distance(self._vehicle.get_transform().location)
        ob_list.sort(key=lambda x:dist(x))

        # search for the first location in which perform the reentry
        search_for_reentry = True
        safety_distance_for_reentry = self._vehicle.bounding_box.extent.x*2 + self._behavior.safety_space_reentry
        i = 0
        while search_for_reentry:
            if i == len(ob_list)-1:
                break
            ob1_length = ob_list[i].bounding_box.extent.x
            ob2_length = ob_list[i+1].bounding_box.extent.x
            distance_between_objects = ob_list[i].get_location().distance(ob_list[i+1].get_location()) - (ob1_length+ob2_length) 
            if distance_between_objects > safety_distance_for_reentry:
                break
            i+=1
            
        other_line_distance = ob_list[i].get_location().distance(self._vehicle.get_location()) + ob_list[i].bounding_box.extent.x + self._behavior.safety_space_reentry/2
        
        print("distance from the last obstacle: ", dist(ob_list[-1]))
        
        print("Distance for overtake: ", other_line_distance)

        target_line_id =  ego_wpt.lane_id + lane_offset
        target_line_id = target_line_id if target_line_id != 0 else target_line_id + 1      # 0 is the central lane

        # list of vehicle on the lane in wihch we have to move
        ob_list = [v for v in vehicle_list if is_within_distance(v.get_transform(), self._vehicle.get_transform(), horizon, [0,90])     # se l'ggetto è avanti a noi ad una distanza massima
                                                and v.id != self._vehicle.id
                                                and target_line_id == self._map.get_waypoint(v.get_transform().location, lane_type=carla.LaneType.Any).lane_id]      # se l'oggetto è nella corsia desiderata

        if len(ob_list) == 0:
            return True, other_line_time
        
        ob_list.sort(key=lambda x:dist(x))
        target_vehicle = ob_list[0]
        target_vehicle_distance = dist(target_vehicle)

        if target_vehicle_distance > other_line_distance:
            target_vehicle_velocity = get_speed(target_vehicle)
            target_vehicle_time = (target_vehicle_distance-other_line_distance)/target_vehicle_velocity
            """
            a = 0.1
            a_max = 1
            v0 = self._vehicle.get_velocity()
            s = other_line_distance
            other_line_time = target_vehicle_time + 1
            while a <= a_max and other_line_time >= target_vehicle_time:
                delta = (-2*v0/a)**2 + 8*s/a #delta non può essere <= 0
                t1,t2 = (-(2*v0/a) + delta**0.5)/2, (-(2*v0/a) - delta**0.5)/2
                other_line_time = max(t1,t2)
                vf = v0 + a*other_line_time
                a+=0.1
                if vf > self.max_speed:
                    other_line_time = target_vehicle_time"""
            
            
            a = 0.1 ### dobbiamo stimare l'acelerazione
            v0 = self._vehicle.get_velocity()
            v_max = self._behavior.max_speed
            t_acc = (v_max-v0)/a
            s_acc = v0*t_acc + 0.5*a*(t_acc**2)
            if s_acc >= other_line_distance:
                print("controlla il tempo")
                delta = (-2*v0/a)**2 + 8*other_line_distance/a #delta non può essere <= 0
                t1,t2 = (-(2*v0/a) + delta**0.5)/2, (-(2*v0/a) - delta**0.5)/2
                other_line_time = max(t1,t2)
                if target_vehicle_time > other_line_time:
                    return True, other_line_time
            elif t_acc < target_vehicle_time:
                t_const =(other_line_distance-s_acc)/v_max
                other_line_time = t_acc+t_const
                if target_vehicle_time > other_line_time:
                    return True, other_line_time
        return False, 0
        


    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        self._update_information()

        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        print(list(self._local_planner._waypoints_queue)[:20], "length: ", len(self._local_planner._waypoints_queue))

        # 1: Red lights and stops behavior
        if self.traffic_light_manager():
            print("RED LIGHTS")
            return self.emergency_stop()

        
        
        # 2.1: Pedestrian avoidance behaviors
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)

        if walker_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = w_distance - max(
                walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance:
                print("TOO CLOSE to pedestrian, distance = ", distance, "ID: ", walker.id)
                return self.emergency_stop()

        # 2.2: Car following behaviors and static object avoidance behaviors
        vehicle_state, vehicle, distance, ob_list = self.collision_and_car_avoid_manager(ego_vehicle_wp)

        if vehicle_state:
            print(vehicle.type_id, distance)
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = distance - max(
                vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # static object management
            if "vehicle" not in vehicle.type_id and distance < self._behavior.overtake_distance and self._overake_coverage <= 0:
                # overtake
                print("START LANE CHANGE")
                overtake_possibile, other_lane_time = self.overtake_manager()
                if overtake_possibile:
                    self.lane_change('left', 0, other_lane_time, 1)
                    target_speed = min([
                        self._behavior.max_speed,
                        self._speed_limit - 5])
                    self._local_planner.set_speed(target_speed)
                    control = self._local_planner.run_step(debug=debug)             
        
            # vehicle management
            if distance < self._behavior.braking_distance:
                # Emergency brake if the car is very close.
                print("TOO CLOSE to vechicle, distance = ", distance, "ID: ", vehicle.id)
                return self.emergency_stop()
            else:
                control = self.car_following_manager(vehicle, distance)

        # 3: Intersection behavior (non c'è molta differenza con il normal behavior perchà e gestito in _vehicle_obstacle_detected)
        elif self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - 5])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # 4: Normal behavior
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)


        return control

    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        print("EMERGENCY STOP")
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control



