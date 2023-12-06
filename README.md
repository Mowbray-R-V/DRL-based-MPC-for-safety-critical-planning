# Deep RL based MPC for Safety Critical Planning in Multi Agent system (Warehouse Robots)  

The problem of optimal unsignalized intersection management for continual streams of randomly arriving robots is considered. This problem involves repeatedly solving different instances of a mixed integer program, for which the computation time using a naive optimization algorithm scales exponentially with the number of robots and lanes. Hence, such an approach is not suitable for real-time implementation. On the other hand the use of end-to-end RL with formal safety guarantees for a fast moving multi-agent system is still a open area of research. This work proposes a hybrid framework that use Deep RL methods for learning a combinatorial problem and MPC for trajectory plannig. This way we can guarantee a formal safety for collision avoidance and a siginificant reduction in computational time. 

In this work we use the DDPG algorithm

explain the architecture
The actor model structure:



![](https://github.com/Mowbray-R-V/DRL-based-MPC-for-safety-critical-planning/edit/main/archi-2.png)

![](https://github.com/Mowbray-R-V/DRL-based-MPC-for-safety-critical-planning/edit/main/archi-RL.png)

# Installation

1. clone the repository
   
 ``` 
 git clone https://github.com/Mowbray-R-V/DRL-based-MPC-for-safety-critical-planning.git
 ``` 
 
2. Create a conda environment and install

 ``` cd DRL-based-MPC-for-safety-critical-planning
     conda env create -f environment.yml
     conda activate sample
     run run_tr-te.sh 
 ```


# Results


RL for learning the combinatorial problem, model image, CNN kernel, MA-DDPG
algorithm images
