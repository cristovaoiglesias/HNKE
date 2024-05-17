using  Plots,Distributions,DifferentialEquations
using BSON#: @load
using Statistics
using DataFrames
using Flux
#using Measurements,
using StaticArrays
using LinearAlgebra
using StatsBase

# Colors
# https://github.com/JuliaGraphics/Colors.jl/blob/master/src/names_data.jl



include("mAb_synthetic_dataset.jl")
# using .mAb_synthetic_dt

#path="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/ensemble/trained_Ensemble_MLP/"
path="/home/bolic/cris/RQ3/ensemble_of_MLP2/"
#
# #load all trained sub-MLP that compose the ensemble
# ensemble_MLP = Dict()
# ensemble_size=100
# for i=1:ensemble_size
#     pt=path*"$(i)_sub_MLP.bson"
#     m = BSON.load(pt, @__MODULE__)
#     ensemble_MLP[i]=m[:model]
# end

function RMSE(observed_data,prediction)
    t=observed_data
    y=prediction
    se = (t - y).^2
    mse = mean(se)
    rmse = sqrt(mse)
    return rmse
end

function RMSPE(ground_truth,estimation)
    # https://s2.smu.edu/tfomby/eco5385_eco6380/lecture/Scoring%20Measures%20for%20Prediction%20Problems.pdf
    # https://stats.stackexchange.com/questions/413249/what-is-the-correct-definition-of-the-root-mean-square-percentage-error-rmspe
    # https://www.sciencedirect.com/topics/earth-and-planetary-sciences/root-mean-square-error#:~:text=The%20RMSE%20statistic%20provides%20information,the%20better%20the%20model's%20performance.
# https://www.researchgate.net/profile/Adriaan-Brebels/publication/281718517_A_survey_of_forecast_error_measures/links/56f43b2408ae81582bf0a1a9/A-survey-of-forecast-error-measures.pdf
    mspe=mean(((ground_truth-estimation)./ground_truth).^2)
    rmspe=sqrt(mspe)
    return rmspe*100
end

function normalize_data(x,min,max)
    y=(x-min)/(max-min)
    return y
end

function inverse_normalizetion(y,min,max)
    x=(y*(max-min))+min
    return x
end

function normalize_input(u0)
    #the max and min values were obtained from the dataset used to train the Ensemble, see file dataset_to_train_EnMLP.jl
    min_values_state_variables=[1.1247355623692597e8, 1.0876934495531811, 0.0016493210047543638, 0.0, 0.002675178467605843, 13.163672825054604, 0.0]
    max_values_state_variables=[3.9843740973885384e9, 100.72750841246663, 25.078825296718655, 124.32540101966985, 16.566031188582603, 4632.744925522347, 91.0]
    for i=1:7
        u0[i]=normalize_data(u0[i],min_values_state_variables[i],max_values_state_variables[i])
    end
    return u0
end

function unnormalize_output(p)
    #the max and min values were obtained from the dataset used to train the Ensemble, see file dataset_to_train_EnMLP.jl
    min_values_estimated_params=[-0.13240732597403193, -5.0243094334252736e-9, -7.242819067849792e-10, -2.075801104192634e-9, -4.1946626959321127e-10, -7.189692896472483e-8]
    max_values_estimated_params=[0.2247201064149725, 6.382248486023288e-9, 9.252830910761743e-10, 3.860459201515955e-9, 1.0219287122168513e-9, 1.2732423754124306e-7]
    for i=1:6
        p[i]=inverse_normalizetion(p[i],min_values_estimated_params[i],max_values_estimated_params[i])
    end
    return p
end


function ensemble_prediction(ensemble,inpt) # make predictions with MLP ensemble where input is states(t) and output params(t)
    #the input is normalized and the output is unnromalized.
    row_prediction=[]
    u0=normalize_input(inpt)
    for j=1:length(ensemble)
        p=ensemble[j](u0)
        # println(p)
        p=unnormalize_output(p)
        push!(row_prediction,p)
    end
    rp=vcat(map(x->x', row_prediction)...)
    rp_mean= [mean(rp[:,1]),mean(rp[:,2]),mean(rp[:,3]),mean(rp[:,4]),mean(rp[:,5]),mean(rp[:,6])]
    rp_std = [std(rp[:,1]),std(rp[:,2]),std(rp[:,3]),std(rp[:,4]),std(rp[:,5]),std(rp[:,6])]
    return rp_mean,rp_std #return mean and std
end




## making predictions with MLP ensemble where input is states(t) and the output is params(t) that enable to predict the states(t+1).

tstart=0.0
tend=103.0
sampling= 7.0
tgrid=tstart:sampling:tend
sampling= 0.125
tgrid_large=tstart:sampling:tend
tgrid_opt=[0.0, 7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, 56.0, 63.0, 70.0, 77.0, 84.0, 91.0, 98.0]

# # function ode_system(u, p, t)
#     Xv, GLC, GLN, LAC, AMM, mAb = u
#     μ_Xv, μ_GLC, μ_GLN, μ_LAC, μ_AMM, μ_mAb = p
#     # du[1] = μ_Xv*Xv  #dXv
#     # du[2] = -μ_GLC*Xv #dGLC
#     # du[3] = -μ_GLN*Xv #dGLN
#     # du[4] = μ_LAC*Xv #+ klac1*GLC + klac2*GLN   #dLAC
#     # du[5] = μ_AMM*Xv  #- klac1*GLC  #μ_AMM*Xv - kdeg*du[3] #dAMM #(eq10: dAMM)
#     # du[6] = μ_mAb*Xv
#     du1 = μ_Xv*Xv;  #dXv
#     du2 = -μ_GLC*Xv; #dGLC
#     du3 = -μ_GLN*Xv; #dGLN
#     du4 = μ_LAC*Xv; #+ klac1*GLC + klac2*GLN   #dLAC
#     du5 = μ_AMM*Xv;  #- klac1*GLC  #μ_AMM*Xv - kdeg*du[3] #dAMM #(eq10: dAMM)
#     du6 = μ_mAb*Xv;
#     return SVector(du1,du2,du3,du4,du5,du6)
# # end



function ode_system(u, p, t)
    Xv, GLC, GLN, LAC, AMM, mAb = u
    μ_Xv, μ_GLC, μ_GLN, μ_LAC, μ_AMM, μ_mAb = p
    # du[1] = μ_Xv*Xv  #dXv
    # du[2] = -μ_GLC*Xv #dGLC
    # du[3] = -μ_GLN*Xv #dGLN
    # du[4] = μ_LAC*Xv #+ klac1*GLC + klac2*GLN   #dLAC
    # du[5] = μ_AMM*Xv  #- klac1*GLC  #μ_AMM*Xv - kdeg*du[3] #dAMM #(eq10: dAMM)
    # du[6] = μ_mAb*Xv
    du1 = μ_Xv*Xv;  #dXv
    du2 = -μ_GLC*Xv; #dGLC
    du3 = -μ_GLN*Xv; #dGLN
    du4 = μ_LAC*Xv; #+ klac1*GLC + klac2*GLN   #dLAC
    du5 = μ_AMM*Xv;  #- klac1*GLC  #μ_AMM*Xv - kdeg*du[3] #dAMM #(eq10: dAMM)
    du6 = μ_mAb*Xv;
    return Array([du1,du2,du3,du4,du5,du6])
end







# ODE system for mAb production used in "Bioprocess optimization under uncertainty using ensemble modeling (2017)"
function ode_system2!(du, u, p, t)
    Xv, Xt, GLC, GLN, LAC, AMM, MAb = u
    mu_max, Kglc, Kgln, KIlac, KIamm, mu_dmax, Kdamm, Klysis, Yxglc, mglc, Yxgln, alpha1, alpha2, Kdgln, Ylacglc, Yammgln, r1, r2 ,lambda = p

    mu = mu_max*(GLC/(Kglc+GLC))*(GLN/(Kgln+GLN))*(KIlac/(KIlac+LAC))*(KIamm/(KIamm+AMM));
    mu_d = mu_dmax/(1+(Kdamm/AMM)^2);

    du[1] = mu*Xv-mu_d*Xv;  #viable cell density XV
    du[2] = mu*Xv-Klysis*(Xt-Xv); #total cell density Xt
    du[3] = -(mu/Yxglc+mglc)*Xv;
    du[4] = -(mu/Yxgln+alpha1*GLN/(alpha2+GLN))*Xv - Kdgln*GLN;
    du[5] = Ylacglc*(mu/Yxglc+mglc)*Xv;
    du[6] = Yammgln*(mu/Yxgln+alpha1*GLN/(alpha2+GLN))*Xv+Kdgln*GLN;
    du[7] = (r2-r1*mu)*lambda*Xv;
end
#parameters from the paper "Bioprocess optimization under uncertainty using ensemble modeling (2017)"
p = [5.8e-2, 0.75, 0.075, 171.756, 28.484, 3e-2, 1.76, 0.05511, 1.061e8, 4.853e-14, 5.57e8, 3.4e-13, 4, 9.6e-3, 1.399, 4.27e-1, 0.1, 2, 7.21e-9 ]

u0 = [2e8   2e8   29.1   4.9  0.0  0.310  80.6; #SPN initial condition from "Bioprocess optimization under uncertainty using ensemble modeling(2017)"
      2e8   2e8   100    4.9  0.0  0.310  80.6; #SP1 In-silico Optimization of a Batch Bioreactor for mAbs Production in Relationship to the Net Evolution of the Hybridoma Cell Culture (2019)
      2e8   2e8   29.1   9.0  0.0  0.310  80.6; #SP2 In-silico Optimization of a Batch Bioreactor for mAbs Production in Relationship to the Net Evolution of the Hybridoma Cell Culture (2019)
      2e8   2e8   45.0   10   0.0  0.310  80.6; #SP3 In-silico Optimization of a Batch Bioreactor for mAbs Production in Relationship to the Net Evolution of the Hybridoma Cell Culture (2019)
      2e9   2e8   100    25   0.0  0.310  80.6] #SP4  In-silico Optimization of a Batch Bioreactor for mAbs Production in Relationship to the Net Evolution of the Hybridoma Cell Culture (2019)

prob =  ODEProblem(ode_system2!, u0[1,:], (tstart,tend), p)
sol_SPN_gt = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid)
sol_SPN_gt_7min = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid_large)
prob =  ODEProblem(ode_system2!, u0[2,:], (tstart,tend), p)
sol_SP1_gt = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid)
sol_SP1_gt_7min = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid_large)
prob =  ODEProblem(ode_system2!, u0[3,:], (tstart,tend), p)
sol_SP2_gt = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid)
sol_SP2_gt_7min = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid_large)
prob =  ODEProblem(ode_system2!, u0[4,:], (tstart,tend), p)
sol_SP3_gt = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid)
sol_SP3_gt_7min = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid_large)
prob =  ODEProblem(ode_system2!, u0[5,:], (tstart,tend), p)
sol_SP4_gt = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid)
sol_SP4_gt_7min = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid_large)



xdt=[(mAb_synthetic_dt.sol_SPN_trainingset1[:,:],sol_SPN_gt,"SPN_trainingset1"),
(mAb_synthetic_dt.sol_SPN_trainingset2[:,:],sol_SPN_gt,"SPN_trainingset2"),
(mAb_synthetic_dt.sol_SPN_trainingset3[:,:],sol_SPN_gt,"SPN_trainingset3"),
(mAb_synthetic_dt.sol_SPN_testingset1[:,:],sol_SPN_gt,"SPN_testingset1"),
(mAb_synthetic_dt.sol_SPN_testingset2[:,:],sol_SPN_gt,"SPN_testingset2"),
# (mAb_synthetic_dt.sol_SPN_testingset3[:,:],sol_SPN_gt_7min,"SPN_testingset3"),
# (mAb_synthetic_dt.sol_SPN_testingset4[:,:],sol_SPN_gt_7min,"SPN_testingset4"),

(mAb_synthetic_dt.sol_SP1_trainingset1[:,:],sol_SP1_gt,"SP1_trainingset1"),
(mAb_synthetic_dt.sol_SP1_trainingset2[:,:],sol_SP1_gt,"SP1_trainingset2"),
(mAb_synthetic_dt.sol_SP1_trainingset3[:,:],sol_SP1_gt,"SP1_trainingset3"),
(mAb_synthetic_dt.sol_SP1_testingset1[:,:],sol_SP1_gt,"SP1_testingset1"),
(mAb_synthetic_dt.sol_SP1_testingset2[:,:],sol_SP1_gt,"SP1_testingset2"),
# (mAb_synthetic_dt.sol_SP1_testingset3[:,:],sol_SP1_gt_7min,"SP1_testingset3"),
# (mAb_synthetic_dt.sol_SP1_testingset4[:,:],sol_SP1_gt_7min,"SP1_testingset4"),

(mAb_synthetic_dt.sol_SP2_trainingset1[:,:],sol_SP2_gt,"SP2_trainingset1"),
(mAb_synthetic_dt.sol_SP2_trainingset2[:,:],sol_SP2_gt,"SP2_trainingset2"),
(mAb_synthetic_dt.sol_SP2_trainingset3[:,:],sol_SP2_gt,"SP2_trainingset3"),
(mAb_synthetic_dt.sol_SP2_testingset1[:,:],sol_SP2_gt,"SP2_testingset1"),
(mAb_synthetic_dt.sol_SP2_testingset2[:,:],sol_SP2_gt,"SP2_testingset2"),
# (mAb_synthetic_dt.sol_SP2_testingset3[:,:],sol_SP2_gt_7min,"SP2_testingset3"),
# (mAb_synthetic_dt.sol_SP2_testingset4[:,:],sol_SP2_gt_7min,"SP2_testingset4"),

(mAb_synthetic_dt.sol_SP3_trainingset1[:,:],sol_SP3_gt,"SP3_trainingset1"),
(mAb_synthetic_dt.sol_SP3_trainingset2[:,:],sol_SP3_gt,"SP3_trainingset2"),
(mAb_synthetic_dt.sol_SP3_trainingset3[:,:],sol_SP3_gt,"SP3_trainingset3"),
(mAb_synthetic_dt.sol_SP3_testingset1[:,:],sol_SP3_gt,"SP3_testingset1"),
(mAb_synthetic_dt.sol_SP3_testingset2[:,:],sol_SP3_gt,"SP3_testingset2"),
# (mAb_synthetic_dt.sol_SP3_testingset3[:,:],sol_SP3_gt_7min,"SP3_testingset3"),
# (mAb_synthetic_dt.sol_SP3_testingset4[:,:],sol_SP3_gt_7min,"SP3_testingset4"),

(mAb_synthetic_dt.sol_SP4_trainingset1[:,:],sol_SP4_gt,"SP4_trainingset1"),
(mAb_synthetic_dt.sol_SP4_trainingset2[:,:],sol_SP4_gt,"SP4_trainingset2"),
(mAb_synthetic_dt.sol_SP4_trainingset3[:,:],sol_SP4_gt,"SP4_trainingset3"),
(mAb_synthetic_dt.sol_SP4_testingset1[:,:],sol_SP4_gt,"SP4_testingset1"),
(mAb_synthetic_dt.sol_SP4_testingset2[:,:],sol_SP4_gt,"SP4_testingset2")
# (mAb_synthetic_dt.sol_SP4_testingset3[:,:],sol_SP4_gt_7min,"SP4_testingset3"),
# (mAb_synthetic_dt.sol_SP4_testingset4[:,:],sol_SP4_gt_7min,"SP4_testingset4"),
]







function ensemble_prediction2(ensemble,inpt) # make predictions with MLP ensemble where input is states(t) and output params(t)
    #the input is normalized and the output is unnromalized.
    row_prediction=[]
    u0=normalize_input(inpt)
    for j=1:length(ensemble)
        p=ensemble[j](u0)
        # println(p)
        p=unnormalize_output(p)
        push!(row_prediction,p)
    end
    rp=vcat(map(x->x', row_prediction)...)
    # rp_mean= [mean(rp[:,1]),mean(rp[:,2]),mean(rp[:,3]),mean(rp[:,4]),mean(rp[:,5]),mean(rp[:,6])]
    # rp_std = [std(rp[:,1]),std(rp[:,2]),std(rp[:,3]),std(rp[:,4]),std(rp[:,5]),std(rp[:,6])]
    return rp
end







xdt=[                                                                       #HM+UKF         #UKF
# (mAb_synthetic_dt.sol_SPN_trainingset1[:,:],sol_SPN_gt,"SPN_trainingset1",1.1615191970075238e8),
# (mAb_synthetic_dt.sol_SPN_trainingset2[:,:],sol_SPN_gt,"SPN_trainingset2"),
# (mAb_synthetic_dt.sol_SPN_trainingset3[:,:],sol_SPN_gt,"SPN_trainingset3"),
# (mAb_synthetic_dt.sol_SPN_testingset1[:,:],sol_SPN_gt,"SPN_testingset1"),
# (mAb_synthetic_dt.sol_SPN_testingset2[:,:],sol_SPN_gt,"SPN_testingset2"),
(mAb_synthetic_dt.sol_SPN_testingset3[:,:],sol_SPN_gt_7min,"SPN_testingset3",0.967e8,6.2e5,0.85e8,6.2e5),
(mAb_synthetic_dt.sol_SPN_testingset4[:,:],sol_SPN_gt_7min,"SPN_testingset4",0.967e8,6.2e5,0.85e8,6.2e5),
# (mAb_synthetic_dt.sol_SPN_testingset3[:,:],sol_SPN_gt_7min,"SPN_testingset3",0.969e7,6e4,0.87e8,600000.0),
# (mAb_synthetic_dt.sol_SPN_testingset4[:,:],sol_SPN_gt_7min,"SPN_testingset4",0.969e7,6e4,0.87e8,600000.0),

# (mAb_synthetic_dt.sol_SP1_trainingset1[:,:],sol_SP1_gt,"SP1_trainingset1",1.1765909776143509e8),
# (mAb_synthetic_dt.sol_SP1_trainingset2[:,:],sol_SP1_gt,"SP1_trainingset2"),
# (mAb_synthetic_dt.sol_SP1_trainingset3[:,:],sol_SP1_gt,"SP1_trainingset3"),
# (mAb_synthetic_dt.sol_SP1_testingset1[:,:],sol_SP1_gt,"SP1_testingset1"),
# (mAb_synthetic_dt.sol_SP1_testingset2[:,:],sol_SP1_gt,"SP1_testingset2"),
(mAb_synthetic_dt.sol_SP1_testingset3[:,:],sol_SP1_gt_7min,"SP1_testingset3",1.03e8,6.2e5,1.02e8,6.2e5),
(mAb_synthetic_dt.sol_SP1_testingset4[:,:],sol_SP1_gt_7min,"SP1_testingset4",1.03e8,6.2e5,1.02e8,6.2e5),

# (mAb_synthetic_dt.sol_SP2_trainingset1[:,:],sol_SP2_gt,"SP2_trainingset1",1.2934938587251931e8),
# (mAb_synthetic_dt.sol_SP2_trainingset2[:,:],sol_SP2_gt,"SP2_trainingset2"),
# (mAb_synthetic_dt.sol_SP2_trainingset3[:,:],sol_SP2_gt,"SP2_trainingset3"),
# (mAb_synthetic_dt.sol_SP2_testingset1[:,:],sol_SP2_gt,"SP2_testingset1"),
# (mAb_synthetic_dt.sol_SP2_testingset2[:,:],sol_SP2_gt,"SP2_testingset2"),
(mAb_synthetic_dt.sol_SP2_testingset3[:,:],sol_SP2_gt_7min,"SP2_testingset3",1.12e8,6.2e5,1.02e8,6.2e5),
(mAb_synthetic_dt.sol_SP2_testingset4[:,:],sol_SP2_gt_7min,"SP2_testingset4",1.12e8,6.2e5,1.02e8,6.2e5),

# (mAb_synthetic_dt.sol_SP3_trainingset1[:,:],sol_SP3_gt,"SP3_trainingset1",1.3215926238913769e8),
# (mAb_synthetic_dt.sol_SP3_trainingset2[:,:],sol_SP3_gt,"SP3_trainingset2"),
# (mAb_synthetic_dt.sol_SP3_trainingset3[:,:],sol_SP3_gt,"SP3_trainingset3"),
# (mAb_synthetic_dt.sol_SP3_testingset1[:,:],sol_SP3_gt,"SP3_testingset1"),
# (mAb_synthetic_dt.sol_SP3_testingset2[:,:],sol_SP3_gt,"SP3_testingset2"),
(mAb_synthetic_dt.sol_SP3_testingset3[:,:],sol_SP3_gt_7min,"SP3_testingset3",0.97e8,6.2e5,0.94e8,6.2e5 ),
(mAb_synthetic_dt.sol_SP3_testingset4[:,:],sol_SP3_gt_7min,"SP3_testingset4",0.97e8,6.2e5,0.94e8,6.2e5 ),

# (mAb_synthetic_dt.sol_SP4_trainingset1[:,:],sol_SP4_gt,"SP4_trainingset1",3.1950906697139233e8),
# (mAb_synthetic_dt.sol_SP4_trainingset2[:,:],sol_SP4_gt,"SP4_trainingset2"),
# (mAb_synthetic_dt.sol_SP4_trainingset3[:,:],sol_SP4_gt,"SP4_trainingset3"),
# (mAb_synthetic_dt.sol_SP4_testingset1[:,:],sol_SP4_gt,"SP4_testingset1"),
# (mAb_synthetic_dt.sol_SP4_testingset2[:,:],sol_SP4_gt,"SP4_testingset2")
(mAb_synthetic_dt.sol_SP4_testingset3[:,:],sol_SP4_gt_7min,"SP4_testingset3",1.08e8,3.2e7,.995e8,3.2e7),
(mAb_synthetic_dt.sol_SP4_testingset4[:,:],sol_SP4_gt_7min,"SP4_testingset4",1.08e8,3.2e7,.995e8,3.2e7)
]                                                                              #R , #Q







# #test
# for Ri in [1e8,1.01e8,1.02e8,1.03e8,1.04e8,1.05e8,1.06e8,1.07e8,1.08e8,1.09e8,1.1e8]
# #Online measurements case
# #predictions with Ensemble of MLP and correction with UKF.
# for Qi in [3.1e7]
# #for Ri in [1e7,2e7,3e7,4e7,6e7 ,7e7,8e7,9e7,10e7,11e7]
# Ri=1.0e8
# Qi=3.1e7
#
# println("\n\n R=$(Ri) Q=$(Qi) \n" )

for dt_idx in [1,3,5,7,9]

RMSPE_HNKEu_EnSizes=[]

for ensemble_size=2:100

   subMLPidx = sample(1:100, ensemble_size, replace = false)

    #load all trained sub-MLP that compose the ensemble
    # ensemble_MLP = Dict()
    # for (i, e) in enumerate(subMLPidx) # =1:ensemble_size
    #     pt=path*"$(e)_sub_MLP.bson"
    #     mlps = BSON.load(pt, @__MODULE__)
    #     ensemble_MLP[i]=mlps[:model]
    # end
    ensemble_MLP = Dict()
    for i=1:ensemble_size
        pt=path*"$(i)_sub_MLP.bson"
        mlps = BSON.load(pt, @__MODULE__)
        ensemble_MLP[i]=mlps[:model]
    end

    for e in xdt[dt_idx:dt_idx]
        #e = xdt[10:10][1]
        xtest = e[1]
        xtest =hcat(xtest[:,1:1],xtest[:,3:end])
        sol_gt = e[2]
        sol_gt = hcat(sol_gt[1,:], sol_gt'[:,3:end])
        str_title = e[3]

        tgrid_opt=Array(0:0.125:103)
        steps=length(tgrid_opt)-1

        pred_EnMLP=[]
        pred_EnMLP2=[]
        pred_EnMLP2_std=[]
        # Filter parameters
        DT = 0.125;
        R1  = (e[4])^2;
        #Q1  = Diagonal([(e[5])^2, 5.01, 1.5, 8., .25, (140)^2]);
        Q1  = Diagonal([(e[5])^2, (6.02)^2, (0.61)^2, (4.29)^2, (0.70)^2, (141.98)^2]);

        R2  = (e[6])^2;
        #Q2  = Diagonal([(e[7])^2, 0.5, 0.5, 0.5, 0.5, (1)^2]);
        Q2  = Diagonal([(e[7])^2, (6.02)^2, (0.61)^2, (4.29)^2, (0.70)^2, (141.98)^2]);

        # UKF Filter
        #P0 = Diagonal([0.01, 0.01, 0.01, 0.01, 0.01, 0.01 ]);
        P0 = Diagonal([(xtest[1,1]-sol_gt[1,1]).^2, 0.01, 0.01, 0.01, 0.01, 0.01 ]);

        m0 = u0_for_pred = xtest[1,:]#[xtest[1,1:1];xtest[1,3:end]] #[2e8  ; 2e8 ;29.1 ; 4.9 ; 0.0 ; 0.310; 80.6; 7.21e-9]# %[0;0;0;0;0;0;0];
        u0_for_params=[xtest[1,:];0]
        u0_for_pred2 = xtest[1,:] #[2e8  ; 2e8 ;29.1 ; 4.9 ; 0.0 ; 0.310; 80.6; 7.21e-9]# %[0;0;0;0;0;0;0];
        u0_for_params2=[xtest[1,:];0]
        u0_for_pred3 = xtest[1,:] #[2e8  ; 2e8 ;29.1 ; 4.9 ; 0.0 ; 0.310; 80.6; 7.21e-9]# %[0;0;0;0;0;0;0];
        u0_for_params3=[xtest[1,:];0]

        m = m0
        P = P0

        m2 = m0
        P2 = P0
        # Precompute the UT weights
        # n = size(m, 1)
        # XI = sqrt(n) * [Matrix{Float64}(I, n, n) -Matrix{Float64}(I, n, n)]
        # W = ones(2 * n) / (2 * n)
        n = size(m, 1)
        alpha = 1e-3
        beta = 2.0
        kappa = 3.0 - n
        lambda = alpha^2 * (n + kappa) - n
        WM = zeros(2 * n + 1)
        WC = zeros(2 * n + 1)
        for j in 1:2*n+1
            if j == 1
                wm = lambda / (n + lambda)
                wc = lambda / (n + lambda) + (1 - alpha^2 + beta)
            else
                wm = 1 / (2 * (n + lambda))
                wc = wm
            end
            WM[j] = wm
            WC[j] = wc
        end


        # Saving the filtering corrections
        MM = zeros(size(m, 1), length(tgrid_opt[2:end]));
        PP = zeros(size(P, 1), size(P, 2), length(tgrid_opt[2:end]));

        tstart=0.0
        tend=103.0
        sampling=0.125
        tgrid=tstart:sampling:tend

        pp=[]

        MM2 = zeros(size(m, 1), length(tgrid_opt[2:end]));
        PP2 = zeros(size(P, 1), size(P, 2), length(tgrid_opt[2:end]));

        NIS_UKF=[]
        NIS_ENN_MM_UKF=[]

        NEES_UKF=[]
        NEES_ENN_MM_UKF=[]

        NEES_UKF_total=[]
        NEES_ENN_MM_UKF_total=[]

        for k=1:steps


     # #### HM
            #println(k)

            #estimate the best parameters(t) for the dynamic system based on states(t) updated by UKF using Xv(t) observed.
            p=ensemble_prediction2(ensemble_MLP,u0_for_params2) #parameters(t) with value ± std
            # estimate the states (t+1) with uncertainty based on states (t) and parameters(t) with value ± std
            tstart=tgrid_opt[k]
            tend=tgrid_opt[k+1]
            prob = ODEProblem(ode_system, u0_for_pred2, (tstart,tend),p[1,:])
            pENN2=[]
            for i=1:ensemble_size
                sol = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid,p=p[i,:]) #states (t+1) with uncertainty based on states (t) and parameters(t) with value ± std
                push!(pENN2,sol[:,end])
            end
            pENN2=vcat(map(x->x',pENN2)...)
            predm=[mean(pENN2[:,1]),mean(pENN2[:,2]),mean(pENN2[:,3]),mean(pENN2[:,4]),mean(pENN2[:,5]),mean(pENN2[:,6])]
            predstd=[std(pENN2[:,1]),std(pENN2[:,2]),std(pENN2[:,3]),std(pENN2[:,4]),std(pENN2[:,5]),std(pENN2[:,6])]
            push!(pred_EnMLP2,predm)
            push!(pred_EnMLP2_std,predstd)
            u0_for_pred2 = [xtest[k+1,1:1];predm[2:6]]
            u0_for_params2=[xtest[k+1,1:1];predm[2:6];tgrid_opt[k+1]]


     # #### HM+UKF


            #estimate the best parameters(t) for the dynamic system based on states(t) updated by UKF using Xv(t) observed.
            p=ensemble_prediction2(ensemble_MLP,u0_for_params) #parameters(t) with value ± std
            # estimate the states (t+1) with uncertainty based on states (t) and parameters(t) with value ± std
            tstart=tgrid_opt[k]
            tend=tgrid_opt[k+1]
            prob =  ODEProblem(ode_system, u0_for_pred, (tstart,tend),p[1,:])
            pENN=[]
            for i=1:ensemble_size
                sol = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid,p=p[i,:]) #states (t+1) with uncertainty based on states (t) and parameters(t) with value ± std
                push!(pENN,sol[:,end])
            end
            # Form the sigma points for dynamic model
            # SX = repeat(m, 1, 2 * n) + cholesky(Hermitian(P)).L * XI
            # Propagate through the dynamic model
            # μ_Xv, μ_GLC, μ_GLN, μ_LAC, μ_AMM, μ_mAb = p[1]
            # HX =hcat(SX[1,:] + (μ_Xv.*SX[1,:]).*DT,
            #         SX[2,:] + (-μ_GLC.*SX[1,:]).*DT,
            #         SX[3,:] + (-μ_GLN.*SX[1,:]).*DT,
            #         SX[4,:] + (μ_LAC.*SX[1,:]).*DT,
            #         SX[5,:] + (μ_AMM.*SX[1,:]).*DT,
            #         SX[6,:] + (μ_mAb.*SX[1,:]).*DT)'
            # Compute the predicted mean and covariance
            # m = zeros(size(m))
            # P = zeros(size(P))
            # for i in 1:size(HX, 2)
            #     m .+= W[i] .* HX[:, i]
            # end
            # for i in 1:size(HX, 2)
            #     P .+= W[i] .* (HX[:, i] .- m) * (HX[:, i] .- m)'
            # end
            pENN=vcat(map(x->x',pENN )...)
            P=Statistics.cov(pENN)
            m=[mean(pENN[:,1]),mean(pENN[:,2]),mean(pENN[:,3]),mean(pENN[:,4]),mean(pENN[:,5]),mean(pENN[:,6])]
            P .+= Q1

            # Form sigma points for measurement step
            # SX = repeat(m, 1, 2 * n) + cholesky(Hermitian(P)).L * XI
            # HY = SX[1, :]
            A = cholesky(Hermitian(P)).L #cholesky(P).L
            SX = [zeros(size(m)) A -A]
            SX = sqrt(n + lambda) .* SX .+ repeat(m, 1, size(SX, 2))
            HY = SX[1, :]

            # Compute the updated mean and covariance
            mu = zeros(1,1)
            S = zeros(1,1)
            C = zeros(size(SX, 1), 1)

            for i in 1:size(SX, 2)
                mu .+= WM[i] .* HY[i]'
            end

            for i in 1:size(SX, 2)
                S = S + WC[i] .* (HY[i]'.- mu) * (HY[i]'.- mu)'
                C .+= WC[i] .* (SX[:, i] .- m) * (HY[i]' .- mu)'
            end
            S .+= R1

            # Compute the gain and updated mean and covariance
            K = C / S
            Y=u0_obs=xtest[k+1,1]
            ie=Y .- mu #innovation
            m .+= K * (ie)
            P .-= K * S * K'

            MM[:, k] = m
            PP[:, :, k] = P

            u0_for_params=[m;tgrid_opt[k+1]]
            u0_for_pred=m

            append!(NIS_ENN_MM_UKF,ie*inv(S)*ie)

            e_x=[sol_gt[k+1,:]-m]
            append!(NEES_ENN_MM_UKF,[[   e_x[1][1]* inv(P[1,1])* e_x[1][1],
                                        e_x[1][2]* inv(P[2,2])* e_x[1][2],
                                        e_x[1][3]* inv(P[3,3])* e_x[1][3],
                                        e_x[1][4]* inv(P[4,4])* e_x[1][4],
                                        e_x[1][5]* inv(P[5,5])* e_x[1][5],
                                        e_x[1][6]* inv(P[6,6])* e_x[1][6]]])

            e_x=sol_gt[k+1,:]-m
            append!(NEES_ENN_MM_UKF_total, e_x'* inv(P)* e_x)

     # #### Classic UKF
            #estimate the best parameters(t) for the dynamic system based on states(t) updated by UKF using Xv(t) observed.
            p=ensemble_prediction(ensemble_MLP,u0_for_params3) #parameters(t) with value ± std

            # Form the sigma points for dynamic model
            # SX = repeat(m2, 1, 2 * n) + cholesky(Hermitian(P2)).L * XI
            A = cholesky(Hermitian(P2)).L #cholesky(P'P).L
            SX = [zeros(size(m2)) A -A]
            SX = sqrt(n + lambda) .* SX .+ repeat(m2, 1, size(SX, 2))

            # Propagate through the dynamic model
            μ_Xv, μ_GLC, μ_GLN, μ_LAC, μ_AMM, μ_mAb = p[1]

            HX =hcat(SX[1,:] + (μ_Xv.*SX[1,:]).*DT,
                    SX[2,:] + (-μ_GLC.*SX[1,:]).*DT,
                    SX[3,:] + (-μ_GLN.*SX[1,:]).*DT,
                    SX[4,:] + (μ_LAC.*SX[1,:]).*DT,
                    SX[5,:] + (μ_AMM.*SX[1,:]).*DT,
                    SX[6,:] + (μ_mAb.*SX[1,:]).*DT)'

            # Compute the predicted mean and covariance
            m2 = zeros(size(m2))
            P2 = zeros(size(P2))
            for i in 1:size(HX, 2)
                m2 .+= WM[i] .* HX[:, i]
            end
            for i in 1:size(HX, 2)
                P2 .+= WC[i] .* (HX[:, i] .- m2) * (HX[:, i] .- m2)'
            end
            P2 .+= Q2

            # Form sigma points for measurement step
            # SX = repeat(m2, 1, 2 * n) + cholesky(Hermitian(P2)).L * XI
            # HY = SX[1, :]
            A = cholesky(Hermitian(P2)).L #cholesky(P).L
            SX = [zeros(size(m2)) A -A]
            SX = sqrt(n + lambda) .* SX .+ repeat(m2, 1, size(SX, 2))
            HY = SX[1, :]
            # Compute the updated mean and covariance
            mu = zeros(1,1)
            S = zeros(1,1)
            C = zeros(size(SX, 1), 1)

            for i in 1:size(SX, 2)
                mu .+= WM[i] .* HY[i]'
            end

            for i in 1:size(SX, 2)
                S = S + WC[i] .* (HY[i]'.- mu) * (HY[i]'.- mu)'
                C .+= WC[i] .* (SX[:, i] .- m) * (HY[i]' .- mu)'
            end
            S .+= R2

            #println(WC,WM)
            # Compute the gain and updated mean and covariance
            K = C / S
            Y=u0_obs=xtest[k+1,1]
            ie=Y .- mu #innovation
            m2 .+= K * (ie)
            P2 .-= K * S * K'

            MM2[:, k] = m2
            PP2[:, :, k] = P2

            u0_for_params3=[m2;tgrid_opt[k+1]]
            u0_for_pred3=m2

            append!(NIS_UKF,ie*inv(S)*ie)

            e_x=[sol_gt[k+1,:]-m2]
            append!(NEES_UKF,[[  e_x[1][1]* inv(P2[1,1])* e_x[1][1],
                                e_x[1][2]* inv(P2[2,2])* e_x[1][2],
                                e_x[1][3]* inv(P2[3,3])* e_x[1][3],
                                e_x[1][4]* inv(P2[4,4])* e_x[1][4],
                                e_x[1][5]* inv(P2[5,5])* e_x[1][5],
                                e_x[1][6]* inv(P2[6,6])* e_x[1][6]]])

            e_x=sol_gt[k+1,:]-m2
            append!(NEES_UKF_total, e_x'* inv(P2)* e_x)


        end


        pred_EnMLP=vcat(map(x->x', pred_EnMLP)...)
        pred_EnMLP2=vcat(map(x->x', pred_EnMLP2)...)
        pred_EnMLP2_std=vcat(map(x->x', pred_EnMLP2_std)...)

          #plots
        lws=2.5
        gr( xtickfontsize=7, ytickfontsize=7, xguidefontsize=9, yguidefontsize=9, legendfontsize=6);

        plots=Plots.scatter(tgrid_opt,xtest[:,1],title=str_title,color=:indianred1, markerstrokewidth = 0, lw=lws, label = "noise dt", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
        # plot!(tgrid_opt[2:end], pred_EnMLP2, label = "ENN", grid=false,ribbon=2*sqrt.(pred_EnMLP2_std),fillalpha=.3,color=:purple, lw=lws,layout=(3,2))
        plot!(Array(0:0.125:103),sol_gt,color=:green , lw=lws, label = "true", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
        plot!(tgrid_opt[2:end], MM', label = "HNKE-U", grid=false,fillalpha=.3,color=:blue, lw=lws,layout=(3,2))
        # we=hcat(PP[1,1,:],PP[2,2,:],PP[3,3,:],PP[4,4,:],PP[5,5,:],PP[6,6,:])
        # for s = 1:6
        #     plot!(plots[s], tgrid_opt[2:end], MM'[:,s], label = "HM+UKF", grid=false,ribbon= (1/2)*1.645 .*sqrt.(PP[s,s,:]),fillalpha=.3,color=:orange, lw=lws)
        # end
        display(plots)
        savefig("HNKE-U_$(ensemble_size)")

        # plots=Plots.scatter(tgrid_opt,xtest[:,1],title=str_title,color=:indianred1, markerstrokewidth = 0, lw=lws, label = "noise dt", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
        # plot!(tgrid_opt[2:end], pred_EnMLP2, label = "ENN", grid=false,ribbon=2*sqrt.(pred_EnMLP2_std),fillalpha=.3,color=:purple, lw=lws,layout=(3,2))
        # plot!(Array(0:0.125:103),sol_gt,color=:green , lw=lws, label = "true", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
        # plot!(layout = (3, 2), size = (800, 700))
        # for s = 1:6
        #     plot!(plots[s], tgrid_opt[2:end], MM2'[:,s], label = "UKF", grid=false,ribbon= (1/2)*1.645 .*sqrt.(PP2[s,s,:]),fillalpha=.3,color=:blue, lw=lws)
        # end
        # for s = 1:6
        #     plot!(plots[s], tgrid_opt[2:end], MM'[:,s], label = "HM+UKF", grid=false,ribbon= (1/2)*1.645 .*sqrt.(PP[s,s,:]),fillalpha=.3,color=:orange, lw=lws)
        # end
        # display(plots)
        # savefig("ENN+UKF2_"*str_title)


        using CSV, DataFrames

        chain_df = DataFrame(MM', :auto)
        CSV.write(str_title*"HNKE-U.csv",chain_df)

        chain_df = DataFrame(MM2', :auto)
        CSV.write(str_title*"UKF.csv",chain_df)


        # println("RMSPE UKF for", str_title)
        # println("Xv: ",RMSPE(sol_gt[2:end,1],MM2'[:,1]))
        # println("glc: ",RMSPE(sol_gt[2:end,2],MM2'[:,2]))
        # println("gln: ",RMSPE(sol_gt[2:end,3],MM2'[:,3]))
        # println("lac: ",RMSPE(sol_gt[2:end,4],MM2'[:,4]))
        # println("amm: ",RMSPE(sol_gt[2:end,5],MM2'[:,5]))
        # println("mAb: ",RMSPE(sol_gt[2:end,6],MM2'[:,6]))

        println("-----RMSPE HNKE-U------ensemble size ", ensemble_size," - ",subMLPidx)
        println("Xv: ",RMSPE(sol_gt[2:end,1],MM'[:,1]))
        println("glc: ",RMSPE(sol_gt[2:end,2],MM'[:,2]))
        println("gln: ",RMSPE(sol_gt[2:end,3],MM'[:,3]))
        println("lac: ",RMSPE(sol_gt[2:end,4],MM'[:,4]))
        println("amm: ",RMSPE(sol_gt[2:end,5],MM'[:,5]))
        println("mAb: ",RMSPE(sol_gt[2:end,6],MM'[:,6]))


        append!(RMSPE_HNKEu_EnSizes,  [[RMSPE(sol_gt[2:end,1],MM'[:,1]),
                                        RMSPE(sol_gt[2:end,2],MM'[:,2]),
                                        RMSPE(sol_gt[2:end,3].+1.0,MM'[:,3].+1.0),
                                        RMSPE(sol_gt[2:end,4],MM'[:,4]),
                                        RMSPE(sol_gt[2:end,5],MM'[:,5]),
                                        RMSPE(sol_gt[2:end,6],MM'[:,6])]])
        # println("RMSPE ENN")
        # println("Xv: ",RMSPE(sol_gt[2:end,1],pred_EnMLP2[:,1]))
        # println("glc: ",RMSPE(sol_gt[2:end,2],pred_EnMLP2[:,2]))
        # println("gln: ",RMSPE(sol_gt[2:end,3],pred_EnMLP2[:,3]))
        # println("lac: ",RMSPE(sol_gt[2:end,4],pred_EnMLP2[:,4]))
        # println("amm: ",RMSPE(sol_gt[2:end,5],pred_EnMLP2[:,5]))
        # println("mAb: ",RMSPE(sol_gt[2:end,6],pred_EnMLP2[:,6]))



        mean_NIS=mean(NIS_UKF)
        N=length(NIS_UKF)
        # https://www.chisquaretable.net/
        # https://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm
        r1=745.3933676480418 # cquantile(Chisq(824-1), .975) # CI from table using N=824 size of noise dt 723.51	880.275
        r2=904.3945815724871 # cquantile(Chisq(824-1), .025)
        println("\n====== Normalised innovations squared Chi2 test ===NIS_UKF===", str_title)
        println("Is N*mean(NIS) inside of 95% CI ($(r1) < $(N*mean_NIS) < $(r2))?", r1 < N*mean_NIS < r2)
        # println("r1 = ",r1)
        # println("r2 = ",r2)
        # #println("mean(NIS): ",mean_NIS)
        # println("N*mean_NIS = ",N*mean_NIS)
        # #println("N = ",N)


        mean_NIS=mean(NIS_ENN_MM_UKF)
        N=length(NIS_ENN_MM_UKF)
        println("\n====== Normalised innovations squared Chi2 test ===NIS_ENN_MM_UKF===", str_title)
        println("Is N*mean(NIS) inside of 95% CI ($(r1) < $(N*mean_NIS) < $(r2))?", r1 < N*mean_NIS < r2)
        # println("r1 = ",r1)
        # println("r2 = ",r2)
        # #println("mean(NIS): ",mean_NIS)
        # println("N*mean_NIS = ",N*mean_NIS)
        # #println("N = ",N)

    end
end

RMSPE_HNKEu_EnSizes=vcat(map(x->x', RMSPE_HNKEu_EnSizes)...)
chain_df = DataFrame(RMSPE_HNKEu_EnSizes, :auto)
CSV.write("RMSPE_HNKEu_EnSizes_$(xdt[dt_idx:dt_idx][1][3]).csv",chain_df)

  #plots
lws=2.5
gr( xtickfontsize=7, ytickfontsize=7, xguidefontsize=9, yguidefontsize=9, legendfontsize=6);
plots=plot(Array(2:1:100),RMSPE_HNKEu_EnSizes,color=:green , lw=lws, label = "RMSPE HNKE-U", xlabel="Ensemble size", ylabel=["Xv(Cell/L)" "GLC(mM)" "GLN(mM)" "LAC(mM)" "AMM(mM)" "mAb(mg/L)"], layout=(3,2),size = (800,700))
display(plots)
savefig("RMSPE_HNKEu_EnSizes_$(xdt[dt_idx:dt_idx][1][3])")

end
