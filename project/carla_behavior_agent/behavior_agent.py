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
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from utils import *

from misc import get_speed, positive, is_within_distance, compute_distance, compute_angle

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
        self._sampling_resolution = 1

        self._max_acc = 3         # max acceleration estimation
        self._lane_width = 3.5
        self._lane_change_distance = 2.75
        self._overtake_counter = 0
        self._finish_overtake_margin = 3
        self._emergency_stop_counter = 0
        self._default_offset = 0

        self._junction_wpt = None
        self._count_stop = 0 

        self._control_dict = dict()
        self._current_stop = None
        self._ignore_junction = False
        self._stops_map = {}

        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()

        self._safety_time = self._behavior.safety_time
        self._braking_distance = self._behavior.braking_distance
        self._max_speed = self._behavior.max_speed
        self._safety_space_reentry = self._behavior.safety_space_reentry

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

        self._look_ahead_steps = int((self._speed) / 10)       #scenario1: self._speed

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW


        # emergency stop for a long time
        if self._emergency_stop_counter > 200:
            self._behavior.safety_time = -1
            self._behavior.braking_distance = 0.5
        if self._speed > 10:
            self._behavior.safety_time = self._safety_time
            self._behavior.braking_distance = self._braking_distance
            self._emergency_stop_counter = 0

        # set when overtake is finished
        if self._len_waypoints_queue_before_overatke is not None and (self._len_waypoints_queue_before_overatke - len(self._local_planner._waypoints_queue)) == (self._overake_coverage + self._finish_overtake_margin):
            self._len_waypoints_queue_before_overatke = None
            self._overake_coverage = 0
            self._overtake = False
            self._behavior.max_speed = self._max_speed
            self._overtake_counter += 1
            self._local_planner.set_lateral_offset(0)
            print("\n\nOvertake done\n\n")

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        if self._junction_wpt is None:
                self._junction_wpt, far_dir = self._local_planner.get_incoming_waypoint_and_direction(steps=15)
                if not self._junction_wpt.is_junction:
                    self._junction_wpt = None
        
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

        safety_distance = max(55, 2*(get_speed(self._vehicle)/10)**2)

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
                    self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=65)
            
            # Check for tailgating: utile quando si parte da bordo strada e ci si deve intromettere
            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self._speed > 10 \
                    and self._behavior.tailgate_counter == 0:
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
        walker_list = [w for w in walker_list if dist(w) < 20]      #considera solo i pedoni a distanza minore di 10

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1, from_walker = True)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1, from_walker = True)
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=90, from_walker = True)

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
            print("Normal behavior in car following")
            target_speed = self._behavior.max_speed
            """target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])"""
            self._local_planner.set_speed(target_speed)

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
        if self._junction_wpt is not None:
            dist = compute_distance(ego_vehicle_wp.transform.location,self._junction_wpt.transform.location)
            if dist >= 1:
                target_speed = self._behavior.arriving_at_junction_speed
                self._local_planner.set_speed(target_speed)
            else:
                self._junction_wpt = None
    
        print("Target speed after car_following_manager: ", target_speed)
        control = self._local_planner.run_step(debug=debug, overtake=self._overtake)

        return control

    def overtake_manager(self, lane_offset = 1):

        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())
        
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        static_ob_list = self._world.get_actors().filter("*static*")

        horizon = 150

        # list of objects to overtake
        ob_list = [v for v in static_ob_list if is_within_distance(v.get_transform(), self._vehicle.get_transform(), horizon, [0,60])     # se l'ggetto è avanti a noi ad una distanza massima
                                                and v.id != self._vehicle.id
                                                and ego_wpt.lane_id == self._map.get_waypoint(v.get_transform().location, lane_type=carla.LaneType.Any).lane_id]      # se l'oggetto è nella nostra stessa corsia

        
        for v in vehicle_list:
            if is_within_distance(v.get_transform(), self._vehicle.get_transform(), horizon, [0,60]) and v.id != self._vehicle.id:
                vehicle_lane = self._map.get_waypoint(v.get_transform().location, lane_type=carla.LaneType.Any).lane_id
                if (v.attributes["base_type"]!="bicycle" and vehicle_lane == ego_wpt.lane_id + 1 and get_speed(v)==0):
                    ob_list.append(v)
                
        def dist(v): return v.get_location().distance(self._vehicle.get_transform().location)
        ob_list.sort(key=lambda x:dist(x))

        if len(ob_list)==0:
            return False, 0

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
            
        other_line_distance = ob_list[i].get_location().distance(self._vehicle.get_location()) + ob_list[i].bounding_box.extent.x + self._behavior.safety_space_reentry/4
        total_overtake_distance = other_line_distance - 2*self._lane_change_distance + 2*(self._lane_change_distance**2 + self._lane_width**2)**(0.5)
                
        print("Distance for overtake: ", total_overtake_distance)

        print("Object to vertake: ", i+1)

        if ego_wpt.lane_id < 0:
            target_line_id =  ego_wpt.lane_id + lane_offset
            target_line_id = target_line_id if target_line_id != 0 else target_line_id + 1      # 0 is the central lane
        else:
            target_line_id =  ego_wpt.lane_id - lane_offset
            target_line_id = target_line_id if target_line_id != 0 else target_line_id - 1      # 0 is the central lane

        up_angle_th = 30
        # list of vehicle on the lane in wihch we have to move
        ob_list = []
        overtake_wpts = self._local_planner.get_incoming_waypoints(int(total_overtake_distance))        
        for wpt in overtake_wpts:
            angle = compute_angle(ego_wpt.transform, wpt.transform)
            if angle < 174.5:       # misurato sperimentalmente
                up_angle_th = 90
                break


        print("up angle th: ", up_angle_th)

        for v in vehicle_list:
            if (is_within_distance(v.get_transform(), self._vehicle.get_transform(), horizon, [0,up_angle_th])    # se l'ggetto è avanti a noi ad una distanza massima
                    and v.id != self._vehicle.id
                    and target_line_id == self._map.get_waypoint(v.get_transform().location, lane_type=carla.LaneType.Any).lane_id):      # se l'oggetto è nella corsia desiderata
                ob_list.append(v)

        if len(ob_list) == 0:
            print("overtake (NO vehicles on th other line)")
            return True, other_line_distance
        
        ob_list.sort(key=lambda x:dist(x))
        target_vehicle = ob_list[0]
        target_vehicle_distance = dist(target_vehicle)

        if target_vehicle_distance > other_line_distance:
            target_vehicle_velocity = get_speed(target_vehicle)/3.6

            if target_vehicle_velocity < 19.44:     #70 km/h in m/s
                print(target_vehicle_velocity)
                target_vehicle_velocity = 19.44   

            target_vehicle_time = (target_vehicle_distance-other_line_distance)/target_vehicle_velocity
            
            
            a = self._max_acc
            v0 = get_speed(self._vehicle)/3.6
            v_max = self._behavior.overtake_velocity/3.6
            t_acc = (v_max-v0)/a
            s_acc = v0*t_acc + 0.5*a*(t_acc**2)
            if s_acc >= total_overtake_distance:
                delta = (2*v0/a)**2 + 8*total_overtake_distance/a #delta non può essere <= 0
                t1,t2 = (-(2*v0/a) + delta**0.5)/2, (-(2*v0/a) - delta**0.5)/2
                other_line_time = max(t1,t2)
                if target_vehicle_time > other_line_time:

                    print("Overtake in acc mot: ",other_line_time,", ", target_vehicle_time)
                    return True, other_line_distance
            elif t_acc < target_vehicle_time:
                t_const =(total_overtake_distance-s_acc)/v_max
                other_line_time = t_acc+t_const
                if target_vehicle_time > other_line_time:
                    print("Overtake in acc + uniform mot: ",other_line_time,", ", target_vehicle_time)
                    return True, other_line_distance
        return False, 0
        
    def invading_vehicles(self, lane_offset = 1):

        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        def dist(v): return v.get_location().distance(self._vehicle.get_transform().location)
        horizon = 20          

        if ego_wpt.lane_id < 0:
            target_line_id =  ego_wpt.lane_id + lane_offset
            target_line_id = target_line_id if target_line_id != 0 else target_line_id + 1      # 0 is the central lane
        else:
            target_line_id =  ego_wpt.lane_id - lane_offset
            target_line_id = target_line_id if target_line_id != 0 else target_line_id - 1      # 0 is the central lane

        ob_list = []
        # list of vehicle on the lane in wihch we have to move
        for v in vehicle_list:
            if (is_within_distance(v.get_transform(), self._vehicle.get_transform(), horizon, [0,90])    # se l'ggetto è avanti a noi ad una distanza massima
                    and v.id != self._vehicle.id
                    and target_line_id == self._map.get_waypoint(v.get_transform().location, lane_type=carla.LaneType.Any).lane_id):      # se l'oggetto è nella corsia desiderata
                ob_list.append(v)

        self._local_planner.set_lateral_offset(self._default_offset)
        self._behavior.max_speed = self._max_speed

        if len(ob_list) != 0:
            ob_list.sort(key=lambda x:dist(x))
            target_vehicle = ob_list[0]

            exact_location = target_vehicle.get_transform().location
            projected_waypoint = self._map.get_waypoint(target_vehicle.get_transform().location, lane_type=carla.LaneType.Any)

            disalignment = compute_distance(exact_location, projected_waypoint.transform.location)
            
            self._local_planner.set_lateral_offset(-2*disalignment)              
            if disalignment > 0.5:                                                 
                self._behavior.max_speed = self._behavior.invading_velocity
    

    def detect_lane_obstacle(self,extension_factor_actor = 2.0, extension_factor_adv = 2.5, margin = 1.02, ego_vehicle_wp = None):
        
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
        affected, traffic_light = self._affected_by_traffic_light_junction(lights_list, max_distance=16)
                                                                                                        

        if traffic_light is not None:
            self._control_dict["traffic_light"] = traffic_light
            self._ignore_junction = True  #c'è un semaforo e quindi l'intersezione viene gestita con il semaforo e non con la gestione degli stop

        return affected
        
    def compute_safe_distance(self, speed):     #calcolo della distanza di sicurezza,vista su google
        safe_distance = (speed * speed) / 152
        return (safe_distance)
        
    def stop_manager(self):         #stessa cosa di traffic light ma si occupa degli stop

        actor_list = self._world.get_actors()
        stop_list = actor_list.filter("*stop*")

        max_distance = self.compute_safe_distance(self._speed + 4)
        affected, stop, distance = self.affected_by_stop(stop_list, max_distance)

        self._control_dict["stop"] = stop

        if(affected):
            print("Control dict: ", self._control_dict["stop"])

        return affected, distance

    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        self._update_information()

        if self._count_stop > 0:
            self._count_stop = self._count_stop + 1
            if(self._count_stop < 200): #se il count arriva a 200 significa che la macchina allo stop si è fermata per 10 secondi
                print("STOP ",self._count_stop)
                return self.emergency_stop()
            else:
                self._current_stop = self._control_dict["stop"]
                self._count_stop = 0

        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        print("My lane id and road id: ", ego_vehicle_wp.lane_id,  ego_vehicle_wp.road_id)
        wpt_right_lane = ego_vehicle_wp.get_right_lane()
        if wpt_right_lane is not None:
            print("Right lane: ", wpt_right_lane.lane_id, wpt_right_lane.road_id)
        else:
            print("Right lane: None")

        print("Ignore junction: ", self._ignore_junction)

        # 0: check for invading vehicles
        if not self._overtake:
            self.invading_vehicles()


        # 1: Red lights and stops behavior
        if self.traffic_light_manager(): #richiama la gestione del semaforo
            print("\nRed Light\n")
            return self.emergency_stop() #fa  la frenata di emergenza 
        self.traffic_light_manager_junction()
        
        affected , distance = self.stop_manager() #gestione degli stop
        if affected:
            if self._current_stop is None or (self._current_stop.id != self._control_dict["stop"].id):
                self._count_stop = 1 #ha visto lo stop
                print("\nSTOP\n")
                return self.emergency_stop()

        
        # 2.1: Pedestrian avoidance behaviors
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)


        if walker_state:
            print("\n\nWALKER\n\n")
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = w_distance - max(
                walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close.
            if distance < self._behavior.pedestrian_braking_distance:
                print("TOO CLOSE to pedestrian, distance = ", distance, "ID: ", walker.id)
                return self.emergency_stop(brake=1)
        
        if self._control_dict.get("traffic_light") is not None:
            traffic_light = self._control_dict["traffic_light"]
            distance = ego_vehicle_loc.distance(traffic_light.get_location())
            if not ego_vehicle_wp.is_junction and self.check_behind(traffic_light, distance):
                self._ignore_junction = False
                del self._control_dict["traffic_light"]

        # 2.2: Car following behaviors and static object avoidance behaviors
        vehicle_state, vehicle, distance, ob_list = self.collision_and_car_avoid_manager(ego_vehicle_wp)

        # entra solo se non stai già sorpassando
        if vehicle_state and not self._overtake:
            #print(vehicle.type_id, distance)
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = distance - max(
                vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # static object and bycicle management
            if (("vehicle" not in vehicle.type_id or vehicle.attributes["base_type"] =="bicycle" or ("vehicle" in vehicle.type_id and self._map.get_waypoint(vehicle.get_transform().location, lane_type=carla.LaneType.Any).lane_id == self._map.get_waypoint(self._vehicle.get_location()).lane_id + 1 and get_speed(vehicle)==0)) and 
                    distance < self._behavior.overtake_distance):
                # overtake
                print("TRYNG TO OVERTAKE ", vehicle)
                overtake_possibile, other_line_distance = self.overtake_manager()
                if overtake_possibile:
                    print("\nSTART OVERTAKE\n")
                    self._behavior.max_speed = self._behavior.overtake_velocity
                    self._local_planner.set_lateral_offset(-0.6)
                    self.lane_change('left', 0, other_line_distance-self._lane_change_distance, self._lane_change_distance) 
                    target_speed = self._behavior.max_speed
                    self._local_planner.set_speed(target_speed)
                    control = self._local_planner.run_step(debug=debug, overtake=self._overtake)  
        
                
            if distance < self._behavior.braking_distance:
                # Emergency brake if the car is very close.
                print("TOO CLOSE to vechicle, distance = ", distance, "ID: ", vehicle.id)
                return self.emergency_stop()
            else:
                print("Attention: ", vehicle.type_id, " distance: ", distance)
                control = self.car_following_manager(vehicle, distance)

        # 3: Intersection behavior
        elif self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT]) and not self._ignore_junction:
            print("INTERSECTION BEHAVIOR")
            if self._junction_wpt is not None:
                self._junction_wpt = None
            is_hazard, junction_vehicle = self.detect_lane_obstacle(ego_vehicle_wp=ego_vehicle_wp)                                 
            if is_hazard:
                return self.emergency_stop(brake=0.75)
            else:
                target_speed = self._behavior.arriving_at_junction_speed
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug, overtake=self._overtake)

        # 4: Normal behavior
        else:
            print("NORMAL BEHAVIOR")
            target_speed = self._behavior.max_speed

            if self._junction_wpt is not None:
                dist = compute_distance(ego_vehicle_wp.transform.location,self._junction_wpt.transform.location)
                print("Junction wpt not None, ",dist)
                if dist >= 1:
                    target_speed = self._behavior.arriving_at_junction_speed
                    print("JUNCTION speed ", dist)
                else:
                    self._junction_wpt = None

            print("Target speed:", target_speed)
            """target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])"""
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug, overtake=self._overtake)


        return control

    def emergency_stop(self, brake=None):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        print("EMERGENCY STOP")
        if get_speed(self._vehicle) < 0.1:
            self._emergency_stop_counter += 1
            print(self._emergency_stop_counter)
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        if brake is not None:
            control.brake = brake
            control.steer = self._local_planner._vehicle_controller.past_steering
        
        control.hand_brake = False
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



