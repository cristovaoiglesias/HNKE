# HNKE
Hybrid nonlinear Kalman estimator (HNKE)


This study introduces the hybrid nonlinear Kalman estimator (HNKE), a novel hybrid Gaussian filter designed for fast and low-cost bioprocess monitoring. It auto-initializes the state error covariance matrix P(0) and iteratively estimates the parameters of a generic unstructured mechanistic model (UMM)  in real time. 
Traditional methods often require extensive tuning and can fail under common biomanufacturing conditions characterized by limited data availability and the use of generic UMMs. HNKE addresses these challenges by integrating a hybrid dynamic model with uncertainty quantification (HMuq) with unscented transformation and cubature rule in a hybrid Gaussian filter framework.
 Our empirical evaluation using synthetic bioprocess data representing monoclonal antibody (mAb) production demonstrates that HNKE requires a small amount of data to be trained and only the definition of measurement  R and process Q noise covariance matrices to outperform the baseline models in monitoring highly nonlinear bioprocesses.
These results suggest that HNKE provides a practical, robust, adaptable, and cost-effective solution for bioprocess monitoring that is aligned with the goals of the biopharmaceutical industry to enhance efficiency and reduce costs. 
