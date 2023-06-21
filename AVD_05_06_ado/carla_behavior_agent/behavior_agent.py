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
from basic_agent import *
from local_planner import RoadOption
from behavior_types import Cautious, Aggressive, Normal
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from utils import *
import math
from controller import *


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
        self._sampling_resolution = 4.5
        self._lane_b = False
        self._behind_bug = False
        self._lane_bycicle_left = False
        self._lane_bycicle_right = False
        self._lane_bycicle_c = True
        self._lane_vehicle_left = False
        self._lane_vehicle_right = False
        self._lane_vehicle_c = True
        self._lane_c = True
        self._lane_right = False
        self._stops_map = {}

       
        self._check_behind_bicycle = []
        self._behind_bicycle = False
        self._count_stop = 0 
        self._count_vehicles=0
        self._bicycle_id = None
        self._vehicle_id = None
        self._vehicle_bug = None
        self._lane_id = None
        self._update_speed = False
        self._path = None
        self._control_dict = dict()
        self._current_stop = None
        self._ignore_junction = False

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
        print("speed limit ",self._speed_limit)
        print("speed attuale: ", self._speed)
        self._local_planner.set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected
    
    def traffic_light_manager_junction(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, traffic_light = self._affected_by_traffic_light_junction(lights_list, max_distance=16)#vede se si trova un semaforo ad un 
                                                                                                        #max di 16 metri e se c'è me lo ritorna

        if traffic_light is not None:
            self._control_dict["traffic_light"] = traffic_light
            self._ignore_junction = True  #c'è un semaforo e non c'è uno stop e quindi l'intersezione viene gestita con il semaforo e non con la lane obstacle

        return affected
    
    def compute_safe_distance(self, speed):#calcolo della distanza di sicurezza,vista su google

        safe_distance = (speed * speed) / 152
        return (safe_distance)

    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        behind_vehicle_state, behind_vehicle, _, _, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
            self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, low_angle_th=160)
        if behind_vehicle_state and self._speed < get_speed(behind_vehicle):
            if (right_turn == carla.LaneChange.Right or right_turn ==
                    carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _, _, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the right!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         right_wpt.transform.location)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _, _, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
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
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id]
        bicycle_list_names = ['vehicle.bh.crossbike', 'vehicle.diamondback.century', 'vehicle.gazelle.omafiets']
        safe_distance = self.compute_safe_distance(self._speed)

        if(len(vehicle_list) != 0):
            for vehicle in vehicle_list:
                if getattr(vehicle,'type_id') in bicycle_list_names:
                    vehicle_list = [i for i in vehicle_list if i != vehicle]
        
        if(len(vehicle_list) != 0):
            if self._direction == RoadOption.CHANGELANELEFT:
                print("change lane left collision and car")
                vehicle_state, vehicle, distance, _, _, _ = self._vehicle_obstacle_detected(
                    vehicle_list, max(
                        self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
            elif self._direction == RoadOption.CHANGELANERIGHT:
                print("change lane right collision and car")
                vehicle_state, vehicle, distance, _, _, _ = self._vehicle_obstacle_detected(
                    vehicle_list, max(
                        self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
            else:
                vehicle_state, vehicle, distance, vehicle_id, vehicle_lane_type, count = self._vehicle_obstacle_detected(
                    vehicle_list, 20, up_angle_th=90)
            
                
                     
            if(safe_distance + 5.5 > distance):             #calcolo la distanza di sicurezza in maniera dinamica e in base alla velocità del veicolo e non più di default
                return vehicle_state, vehicle, distance
            else:
                return(False, None, -1)
        else:
            return(False, None, -1)
        
    
    
            
        
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

        walker_list = self._world.get_actors().filter("walker.pedestrian.*")
        def dist(w): return w.get_location().distance(waypoint.transform.location)
        walker_list = [w for w in walker_list if dist(w) < 60 and w.id != self._vehicle.id]
        safe_distance = self.compute_safe_distance(self._speed)

        for walker in walker_list:
            walker_location = walker.get_location()
            walker_wp = self._map.get_waypoint(walker_location)
            #print("Road id, lane id e lane type del walker a 40 metri: ", walker_wp.road_id, walker_wp.lane_id, walker_wp.lane_type, getattr(walker, "type_id"))

        if(len(walker_list) != 0):
            if self._direction == RoadOption.CHANGELANELEFT:
                walker_state, walker, distance, _, _, _ = self._vehicle_obstacle_detected(walker_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
            elif self._direction == RoadOption.CHANGELANERIGHT:
                walker_state, walker, distance, _, _, _ = self._vehicle_obstacle_detected(walker_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
            else:
                walker_state, walker, distance, _, _, _ = self._vehicle_obstacle_detected(walker_list, 35, up_angle_th=180)

                #print("valore di ritorno della predestrian detect: ", walker_state, walker, distance)
                
                if(safe_distance + 5.5 > distance):
                    return walker_state, walker, distance
                else:
                    return(False, None, -1)
                
            return walker_state, walker, distance
        else:
            return (False, None, -1)
    

    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """

        print("Sono in car following")

        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        # Under safety time distance, slow down.
        if self._behavior.safety_time > ttc > 0.0:
            target_speed = min([
                positive(vehicle_speed - self._behavior.speed_decrease),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Normal behavior.
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control
        
    def detect_lane_obstacle(self,extension_factor_actor = 2.3, extension_factor_adv = 2.8, margin = 1.02, ego_vehicle_wp = None):
        
        actor = self._vehicle
        world = CarlaDataProvider.get_world()
        world_actors = world.get_actors().filter('*vehicle*')
        actor_bbox = actor.bounding_box
        actor_transform = actor.get_transform()
        actor_location = actor_transform.location
        actor_vector = actor_transform.rotation.get_forward_vector()
        actor_vector = np.array([actor_vector.x, actor_vector.y])
        actor_vector = actor_vector / np.linalg.norm(actor_vector)
        actor_vector = actor_vector * (extension_factor_actor - 1) * actor_bbox.extent.x
        actor_location = actor_location + carla.Location(actor_vector[0], actor_vector[1])
        actor_yaw = actor_transform.rotation.yaw

        if(len(world_actors) != 0):
            is_hazard = False
            for adversary in world_actors:
                adversary_wp = self._map.get_waypoint(adversary.get_location())
                if adversary.id != actor.id and actor_transform.location.distance(adversary.get_location()) < 40 and adversary_wp.road_id != ego_vehicle_wp.road_id:
                    adversary_bbox = adversary.bounding_box
                    adversary_transform = adversary.get_transform()
                    adversary_loc = adversary_transform.location
                    adversary_yaw = adversary_transform.rotation.yaw
                    if self._incoming_direction in [RoadOption.STRAIGHT]:
                        overlap_adversary = RotatedRectangle(
                            adversary_loc.x, adversary_loc.y,
                            2 * margin * adversary_bbox.extent.x * extension_factor_adv, adversary_bbox.extent.y, adversary_yaw)
                        overlap_actor = RotatedRectangle(
                            actor_location.x, actor_location.y,
                            2 * margin * actor_bbox.extent.x * 2, actor_bbox.extent.y, actor_yaw)
                    else:
                        overlap_adversary = RotatedRectangle(
                            adversary_loc.x, adversary_loc.y,
                            1.75 * margin * adversary_bbox.extent.x * extension_factor_adv, 2 * margin * adversary_bbox.extent.y, adversary_yaw)
                        overlap_actor = RotatedRectangle(
                            actor_location.x, actor_location.y,
                            2 * margin * actor_bbox.extent.x * 2, 2 * margin * actor_bbox.extent.y*extension_factor_actor, actor_yaw)
                    overlap_area = overlap_adversary.intersection(overlap_actor).area
                    speed_adv = get_speed(adversary)
                    if overlap_area > 0 and  speed_adv > 5:
                        is_hazard = True
                        return  (is_hazard, adversary)
                    
            return (is_hazard, None)
        return(False, None)
    
    
    def stop_manager(self):#stessa cosa di traffic light ma si occupa degli stop

        actor_list = self._world.get_actors()
        stop_list = actor_list.filter("*stop*")

        max_distance = self.compute_safe_distance(self._speed + 4)
        affected, stop, distance = self.affected_by_stop(stop_list, max_distance)

        self._control_dict["stop"] = stop

        if(affected):
            print("Control dict: ", self._control_dict["stop"])

        return affected, distance


    def run_step(self, debug=False):#viene chimata ogni 0.05 secondi dal programma
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        self._update_information()
        if self._count_stop > 0:
            self._count_stop = self._count_stop + 1
            if(self._count_stop < 200): #se il count arriva a 200 significa che la macchina allo stop si è fermata per 10 secondi
                return self.emergency_stop()
            else:
                
                self._current_stop = self._control_dict["stop"]
                self._count_stop = 0

        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
        print("Ego wp lane id: ", ego_vehicle_wp.lane_id)

        print("Ignore junction: ", self._ignore_junction)

        # 1: Red lights and stops behavior
        if self.traffic_light_manager(): #richiama la gestione del semaforo
            return self.emergency_stop() #fa  la frenata di emergenza 
        self.traffic_light_manager_junction()
        
        affected , distance = self.stop_manager() #gestione degli stop
        if affected:
            if self._current_stop is None:
                self._count_stop = 1 #ha visto lo stop
                print("ho visto lo stop")
                return self.emergency_stop()
            else:
                if self._current_stop.id != self._control_dict["stop"].id:#mi salvo l'id dello stop per evitare che il sistema veda 2 volte lo stesso stop
                    self._count_stop = 1
                    print("ho visto lo stop")
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
                print("emergency nel pedestrian avoid")
                return self.emergency_stop()
            
        
        if self._control_dict.get("traffic_light") is not None:
            traffic_light = self._control_dict["traffic_light"]
            distance = ego_vehicle_loc.distance(traffic_light.get_location())
            if not ego_vehicle_wp.is_junction and self.check_behind(traffic_light, distance):
                self._ignore_junction = False
                del self._control_dict["traffic_light"]

        # 2.2: Car following behaviors
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)
        #print("Valore di ritorno della colision and car nel run step: ", vehicle_state, vehicle, distance)

        if vehicle_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = distance - max(
                vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
        
            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance:
                print("sono in emergency nella collision and car, vehicle_state, vehicle, distance", vehicle_state, vehicle, distance)
                return self.emergency_stop()
            else:
                control = self.car_following_manager(vehicle, distance)    

        # 3: Intersection behavior

        elif ego_vehicle_wp.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT]) and not self._ignore_junction:# la self.ignor.... è una variabile che viene settata a true
            is_hazard, junction_vehicle = self.detect_lane_obstacle(ego_vehicle_wp=ego_vehicle_wp)                                  #quando l'incrocio è gestito dal semaforo
            if is_hazard:
                print("mi interseco con il veicolo",junction_vehicle)
                return self.emergency_stop_junction()
            else:
                print("sono in un incrocio ma nessun veicolo mi rompe il cazzo")
                target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - 5])
                
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)

        # 4: Normal behavior
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        control.steer = self._local_planner._vehicle_controller.past_steering
        return control

    def emergency_stop_junction(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 0.75
        control.hand_brake = False
        control.steer = self._local_planner._vehicle_controller.past_steering
        return control
