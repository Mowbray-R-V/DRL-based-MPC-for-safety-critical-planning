# Deep RL based MPC for Safety Critical Planning in Multi Agent system (Warehouse Robots)  

The problem of optimal unsignalized intersection management for continual streams of randomly arriving robots is considered. This problem involves repeatedly solving different instances of a mixed integer program, for which the computation time using a naive optimization algorithm scales exponentially with the number of robots and lanes. Hence, such an approach is not suitable for real-time implementation. On the other hand the use of end-to-end RL with formal safety guarantees for a fast moving multi-agent system is still a open area of research. This work proposes a hybrid framework that use Deep RL methods for learning a combinatorial problem and MPC for trajectory plannig. This way we can guarantee a formal safety for collision avoidance and a siginificant reduction in computational time. 

 MA-DDPG actor model
![](https://github.com/Mowbray-R-V/DRL-based-MPC-for-safety-critical-planning/blob/main/model.png)


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

# Simulation results
https://github.com/Mowbray-R-V/DRL-based-MPC-for-safety-critical-planning/assets/106313820/cf509cbe-3f54-4a64-8c77-d4d6ddfc1130

