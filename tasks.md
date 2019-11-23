# Tasks
## 1. Error Mitigation with DNNs:

   1. Develop deep AutoEncoders (denoising and sparse) for error mitigation of quantum circuits
   
       a. Implement and Test on state and process tomography (in simulation and hardware) to quatify efficacy of AE.

          Potential Outcome: higher reconstruction fidelities and faster convergence for process tomography

       b. Implement and Test on quantum circuit learning research. 

          Potential Outcome: representing deeper quantum circuits 

   2. Develop Deep Generative Stochastic Networks for error mitigation during sampling
   
       a. Do (a,b) in **1**.

       b. Benchmark against other (shot-noise sensitive) techniques such as Richardson interpolation, etc...

          Potential Outcome: shot-noise robust and learnable error mitigation model.  
---           
## 2. Accelerated VQE:
   i. Use models developped within (i) and (ii) for on-the-fly error mitigation on VQE quantum chemistry benchmarks.  

    Potential Outcome: faster convergence rates from the classical optimizer and higher sample efficiency
---
## 3. DNN-based Readout Models:
   i. Develop DNN-based models for readout of SC-based quantum circuits.
   ii. Develop DNN-based models for readout of trapped-ion qubits