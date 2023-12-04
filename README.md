# Deep RL based MPC for Safety Critical Planning in Multi Agent System 

In the problem of optimal unsignalized intersection management for continual streams of randomly arriving robots. This problem involves repeatedly solving different instances of a mixed integer program, for which the computation time using a naive optimization algorithm scales exponentially with the number of robots and lanes. Hence, such an approach is not suitable for real-time implementation. The use of end-to-end RL with formal safety guarantees for a multi-agent system


This work propose a solution framework that combines learning and sequential optimization. In particular, we propose an algorithm for learning a shared policy that given the traffic state information, determines the crossing order of the robots. Then, we optimize the trajectories of the robots sequentially according to that crossing order. This approach inherently guarantees safety at all times. 



![](https://github.com/Mowbray-R-V/DRL-based-MPC-for-safety-critical-planning/edit/main/archi-2.png)

![](https://github.com/Mowbray-R-V/DRL-based-MPC-for-safety-critical-planning/edit/main/archi-RL.png)

#To-DO

RL for learning the combinatorial problem, model image, CNN kernel, MA-DDPG
algorithm images
