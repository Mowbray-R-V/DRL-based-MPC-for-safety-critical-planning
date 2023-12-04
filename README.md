# Deep RL based MPC for Safety Critical Planning in Multi Agent System 

The problem of optimal unsignalized intersection management for continual streams of randomly arriving robots is considered. This problem involves repeatedly solving different instances of a mixed integer program, for which the computation time using a naive optimization algorithm scales exponentially with the number of robots and lanes. Hence, such an approach is not suitable for real-time implementation. On the other hand the use of end-to-end RL with formal safety guarantees for a fast moving multi-agent system is still a open are of research. This work proposes a hybrid framework that use Deep RL methods for learning a combinatorial problem and MPC for trajectory plannig. This way we can guarantee a formal safety for collision avoidance and a siginificant reduction in computational time. 





![](https://github.com/Mowbray-R-V/DRL-based-MPC-for-safety-critical-planning/edit/main/archi-2.png)

![](https://github.com/Mowbray-R-V/DRL-based-MPC-for-safety-critical-planning/edit/main/archi-RL.png)

#To-DO

RL for learning the combinatorial problem, model image, CNN kernel, MA-DDPG
algorithm images
