# DRL-based-MPC-for-safety-critical-planning
## Work Abstract

We consider the problem of optimal unsignalized intersection management for continual streams of randomly arriving robots. This problem involves repeatedly solving different instances of a mixed integer program, for which the computation time using a naive optimization algorithm scales exponentially with the number of robots and lanes. Hence, such an approach is not suitable for real-time implementation. In this paper, we propose a solution framework that combines learning and sequential optimization. In particular, we propose an algorithm for learning a shared policy that given the traffic state information, determines the crossing order of the robots. Then, we optimize the trajectories of the robots sequentially according to that crossing order. This approach inherently guarantees safety at all times. We validate the performance of this approach using extensive simulations. Our approach, on average, significantly outperforms the heuristics from the literature. We also show through simulations that the computation time for our approach scales linearly with the number of robots. We further implement the learnt policies on physical robots with a few modifications to the solution framework to address real-world challenges and establish its real-time implementability.



![](https://github.com/Mowbray-R-V/DRL-based-MPC-for-safety-critical-planning/edit/main/archi-2.png)
![](https://github.com/Mowbray-R-V/DRL-based-MPC-for-safety-critical-planning/edit/main/archi-RL.png)

#To-DO

RL for learning the combinatorial problem, model image, CNN kernel, MA-DDPG
algorithm images
