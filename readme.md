# AVD
The repo contains code and report for the university exam *Autonomous Vehicle Driving*.

The purpose of this project is to deisgn, implement and evaluate an autonomous driving system that respects as much as possible the road rules.
The realization of the project is made possible through the utilization of the simulator CARLA, starting from a baseline of reference. The baseline already implements the main components of an autonomous driving system, that are the global planner, the behavior planner, the local planner, along with PID controller for longitudinal control and Stanley controller for lateral control.
Our goal was to analyze the main features of the baseline, highlighting critical issues and trying to improve them, integrating the management of some situations or behaviors that the baseline doesn’t handle or handles badly.

The system of autonomous driving is applied on a car and tested in five routes, each of which with its weather conditions, environmental context and road scenarios. Each route consists of a series of events/scenarios that test the robustness of the autonomous driving system.


The final project implements the main components that an autonomous driving system needs, that are:
- **mission planner**, able to provide the mission planning, in order to properly navigate each route, finding the route through the A* algorithm;
- **behavior planner**, that implements a rule based approach for managing different situation, but not all that a vehicle can found on the road. The hierarchy of the rules is the following:
- - *Roadway narrowing management* - takes effect when the ego vehicle encounters a vehicle coming from the other lane invading the ego vehicle lane;
- - *Traffic light management* - takes effect when the ego vehicle encounters a red traffic light;
- - *Stop management* - takes effect when the ego vehicle encounters stop signal;
- - *Pedestrian avoidance* - allows the ego vehicle to stop when it is too close to a pedestrian;
- - *Vehicle and static object management* - takes effect when the ego vehicle encounter a vehi- cle (car, bicycle and motorcycle) or a static object. This point is further divided in three situations:
- - - *Vehicle and static object avoidance* - allows the ego vehicle to stop when it is too close to a vehicle or static object;
- - - *Vehicle following* - the ego vehicle follows the leading vehicle if it is at a safety distance, otherwise it starts decelerate;
- - - *Overtake management* - takes effect when the ego vehicle encounters one of these objects: cars that are on the roadside, bicycle, motorcycle or static object;
- - *Intersection behavior* - takes effect when the ego vehicle is in proximity of intersections not regulated by a traffic light (described in section 6.2);
- - *normal behavior* - the ego vehicle follows the preset max speed or the speed limit when none of the listed situations happen.
- **local planner**, that provides the interface to set a target speed or to modify the current path, setting a new global plan or adding new waypoint to the existing queue;
- **controllers**, a PID for longitudinal control and a Stanley Controller for lateral control.


All the various steps and experiments that led to the develop of the final system are in the **Report_AVD.pdf** file. The outline is in the  **Projects_AVD.pdf** file the code, instead, is in the **project** folder.