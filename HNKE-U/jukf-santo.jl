using  Plots,Distributions,DifferentialEquations
using BSON#: @load
using Statistics
using DataFrames
using Flux
#using Measurements,
using StaticArrays
using LinearAlgebra

# Colors
# https://github.com/JuliaGraphics/Colors.jl/blob/master/src/names_data.jl



include("mAb_synthetic_dataset.jl")
# using .mAb_synthetic_dt

path="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/ensemble/trained_Ensemble_MLP/"
# path="/home/bolic/cris/RQ3/ensemble_of_MLP2/"

#load all trained sub-MLP that compose the ensemble
ensemble_MLP = Dict()
ensemble_size=100
for i=1:ensemble_size
    pt=path*"$(i)_sub_MLP.bson"
    m = BSON.load(pt, @__MODULE__)
    ensemble_MLP[i]=m[:model]
end

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







xdt=[                                                                       #HM+UKF         #JUKF
# (mAb_synthetic_dt.sol_SPN_trainingset1[:,:],sol_SPN_gt,"SPN_trainingset1",1.1615191970075238e8),
# (mAb_synthetic_dt.sol_SPN_trainingset2[:,:],sol_SPN_gt,"SPN_trainingset2"),
# (mAb_synthetic_dt.sol_SPN_trainingset3[:,:],sol_SPN_gt,"SPN_trainingset3"),
# (mAb_synthetic_dt.sol_SPN_testingset1[:,:],sol_SPN_gt,"SPN_testingset1"),
# (mAb_synthetic_dt.sol_SPN_testingset2[:,:],sol_SPN_gt,"SPN_testingset2"),
(mAb_synthetic_dt.sol_SPN_testingset3[:,:],sol_SPN_gt_7min,"SPN_testingset3",0.967e8,6.2e5,7.5e7,6.2e5),
(mAb_synthetic_dt.sol_SPN_testingset4[:,:],sol_SPN_gt_7min,"SPN_testingset4",0.967e8,6.2e5,7.5e7,6.2e5),
# (mAb_synthetic_dt.sol_SPN_testingset3[:,:],sol_SPN_gt_7min,"SPN_testingset3",0.969e7,6e4,0.87e8,600000.0),
# (mAb_synthetic_dt.sol_SPN_testingset4[:,:],sol_SPN_gt_7min,"SPN_testingset4",0.969e7,6e4,0.87e8,600000.0),

# (mAb_synthetic_dt.sol_SP1_trainingset1[:,:],sol_SP1_gt,"SP1_trainingset1",1.1765909776143509e8),
# (mAb_synthetic_dt.sol_SP1_trainingset2[:,:],sol_SP1_gt,"SP1_trainingset2"),
# (mAb_synthetic_dt.sol_SP1_trainingset3[:,:],sol_SP1_gt,"SP1_trainingset3"),
# (mAb_synthetic_dt.sol_SP1_testingset1[:,:],sol_SP1_gt,"SP1_testingset1"),
# (mAb_synthetic_dt.sol_SP1_testingset2[:,:],sol_SP1_gt,"SP1_testingset2"),
(mAb_synthetic_dt.sol_SP1_testingset3[:,:],sol_SP1_gt_7min,"SP1_testingset3",1.03e8,6.2e5,7.53e7,6.2e5),
(mAb_synthetic_dt.sol_SP1_testingset4[:,:],sol_SP1_gt_7min,"SP1_testingset4",1.03e8,6.2e5,7.53e7,6.2e5),

# (mAb_synthetic_dt.sol_SP2_trainingset1[:,:],sol_SP2_gt,"SP2_trainingset1",1.2934938587251931e8),
# (mAb_synthetic_dt.sol_SP2_trainingset2[:,:],sol_SP2_gt,"SP2_trainingset2"),
# (mAb_synthetic_dt.sol_SP2_trainingset3[:,:],sol_SP2_gt,"SP2_trainingset3"),
# (mAb_synthetic_dt.sol_SP2_testingset1[:,:],sol_SP2_gt,"SP2_testingset1"),
# (mAb_synthetic_dt.sol_SP2_testingset2[:,:],sol_SP2_gt,"SP2_testingset2"),
(mAb_synthetic_dt.sol_SP2_testingset3[:,:],sol_SP2_gt_7min,"SP2_testingset3",1.12e8,6.2e5,8.53e7,6.2e5),
(mAb_synthetic_dt.sol_SP2_testingset4[:,:],sol_SP2_gt_7min,"SP2_testingset4",1.12e8,6.2e5,8.53e7,6.2e5),

# (mAb_synthetic_dt.sol_SP3_trainingset1[:,:],sol_SP3_gt,"SP3_trainingset1",1.3215926238913769e8),
# (mAb_synthetic_dt.sol_SP3_trainingset2[:,:],sol_SP3_gt,"SP3_trainingset2"),
# (mAb_synthetic_dt.sol_SP3_trainingset3[:,:],sol_SP3_gt,"SP3_trainingset3"),
# (mAb_synthetic_dt.sol_SP3_testingset1[:,:],sol_SP3_gt,"SP3_testingset1"),
# (mAb_synthetic_dt.sol_SP3_testingset2[:,:],sol_SP3_gt,"SP3_testingset2"),
(mAb_synthetic_dt.sol_SP3_testingset3[:,:],sol_SP3_gt_7min,"SP3_testingset3",0.97e8,6.2e5,0.9e8,6.2e5 ),
(mAb_synthetic_dt.sol_SP3_testingset4[:,:],sol_SP3_gt_7min,"SP3_testingset4",0.97e8,6.2e5,0.9e8,6.2e5 ),

# (mAb_synthetic_dt.sol_SP4_trainingset1[:,:],sol_SP4_gt,"SP4_trainingset1",3.1950906697139233e8),
# (mAb_synthetic_dt.sol_SP4_trainingset2[:,:],sol_SP4_gt,"SP4_trainingset2"),
# (mAb_synthetic_dt.sol_SP4_trainingset3[:,:],sol_SP4_gt,"SP4_trainingset3"),
# (mAb_synthetic_dt.sol_SP4_testingset1[:,:],sol_SP4_gt,"SP4_testingset1"),
# (mAb_synthetic_dt.sol_SP4_testingset2[:,:],sol_SP4_gt,"SP4_testingset2")
(mAb_synthetic_dt.sol_SP4_testingset3[:,:],sol_SP4_gt_7min,"SP4_testingset3",1.08e8,3.2e7,.995e8,3.2e7),
(mAb_synthetic_dt.sol_SP4_testingset4[:,:],sol_SP4_gt_7min,"SP4_testingset4",1.08e8,3.2e7,.995e8,3.2e7)
]                                                                              #R , #Q





indx=6
for e in xdt#[indx-1:indx]
    # e = xdt[1:1][1]
    xtest = e[1]
    xtest =hcat(xtest[:,1:1],xtest[:,3:end])
    sol_gt = e[2]
    sol_gt = hcat(sol_gt[1,:], sol_gt'[:,3:end])
    str_title = e[3]

    tgrid_opt=Array(0:0.125:103)
    steps=length(tgrid_opt)-1

    # m0 = xtest[1,:]

    u0_for_pred3 = xtest[1,:] #[2e8  ; 2e8 ;29.1 ; 4.9 ; 0.0 ; 0.310; 80.6; 7.21e-9]# %[0;0;0;0;0;0;0];
    u0_for_params3=[xtest[1,:];0]

    # prms=[0.040805906, 6.502914f-11, 8.0469596f-11, 7.729657f-10, 1.2207346f-10, 1.2906075f-8]
    # prms=ensemble_prediction(ensemble_MLP,u0_for_params3)[1] #parameters(t) with value ± std
    # println("\nparam",prms)

    # Filter parameters
    DT = 0.125;

    R2  = (e[6])^2;                                                                 #mu_Xv
    # Q2  = Diagonal([(e[7])^2, (6.02)^2, (0.61)^2, (4.29)^2, (0.70)^2, (141.98)^2, (0.001)^2, (1.)^2, (.1)^2, (0.1)^2,(0.1)^2, (10.)^2]);
    Q2  = Diagonal([(e[7])^2, (6.02)^2, (0.61)^2, (4.29)^2, (0.70)^2, (141.98)^2, (0.001)^2, (.4)^2, (.05)^2, (0.2)^2,(0.06)^2, (10.1)^2]);

    # UKF Filter
    if str_title=="SPN_testingset3"
        m0 = [2e8, 29.1, 4.9, 0.0, 0.31, 80.6] #xtest[1,:]
        prms = [0.03335603, 0.4044871f-9, 1.208975775f-10, 0.5837982f-9, 1.1768583f-10, 1.857544f-8]
        # prms = [0.020151252, 0.2848712f-9, 1.188421f-10, 0.7980118f-9, 1.6860592f-10, 1.0797752f-8]
        P0 = Diagonal([(xtest[1,1]-sol_gt[1,1]).^2, 0.01, 0.01, 0.01, 0.01, 0.01,     (0.1)^2, (.1)^2, (0.1)^2, (0.1)^2,(0.1)^2, (.1)^2 ]);
        # P0 = Diagonal([9.734512849431412e15, 0.01, 0.01, 0.01, 0.01, 0.01,     (0.1)^2, (.1)^2, (0.1)^2, (0.1)^2,(0.1)^2, (.1)^2 ]);
        # println(P0) 2.9135522937621916e16
        P0 = Matrix(P0)
        santo=0.000001
        P0[8,1]=santo
        P0[1,8]=santo
        santo=0.001
        P0[9,1]=santo
        P0[1,9]=santo
        santo=0.055
        P0[10,1]=santo
        P0[1,10]=santo
        santo=.011
        P0[11,1]=santo
        P0[1,11]=santo
        santo=.01
        P0[12,1]=santo
        P0[1,12]=santo
    elseif str_title=="SPN_testingset4"
        m0 = [2e8, 29.1, 4.9, 0.0, 0.31, 80.6] #xtest[1,:]
        # prms = [0.020151252, 0.2848712f-9, 1.188421f-10, 0.7980118f-9, 1.6860592f-10, 1.0797752f-8]
        prms = [0.03335603, 0.4044871f-9, 1.208975775f-10, 0.5837982f-9, 1.1768583f-10, 1.857544f-8]
        P0 = Diagonal([2.9135522937621916e16, 0.01, 0.01, 0.01, 0.01, 0.01,     (0.1)^2, (.1)^2, (0.1)^2, (0.1)^2,(0.1)^2, (.1)^2 ]);
        # println(P0) 9.734512849431412e15
        P0 = Matrix(P0)
        santo=0.000001
        P0[8,1]=santo
        P0[1,8]=santo
        santo=0.001
        P0[9,1]=santo
        P0[1,9]=santo
        santo=0.055
        P0[10,1]=santo
        P0[1,10]=santo
        santo=.011
        P0[11,1]=santo
        P0[1,11]=santo
        santo=.01
        P0[12,1]=santo
        P0[1,12]=santo



    elseif str_title=="SP1_testingset3"
        m0 = [2e8, 100.0, 4.9, 0.0, 0.31, 80.6] #xtest[1,:]
        # prms = [0.0042107785, 0.4498021f-9, 1.0175618f-10, 5.4629337f-10, 6.0986106f-11, 1.0797752f-8]
        prms = [0.0042107785, 0.4498021f-9, 1.2175618f-10, 7.04629337f-10, 4.5986106f-11, 1.8797752f-8]
        P0 = Diagonal([5.193859265558679e15, 0.01, 0.01, 0.01, 0.01, 0.01,     (0.1)^2, (.1)^2, (0.1)^2, (0.1)^2,(0.1)^2, (.1)^2 ]);
        # println(P0) 4.4804014956853925e14
        P0 = Matrix(P0)
        santo=0.000002
        P0[8,1]=santo
        P0[1,8]=santo
        santo=0.0011
        P0[9,1]=santo
        P0[1,9]=santo
        santo=0.00056
        P0[10,1]=santo
        P0[1,10]=santo
        santo=.00012
        P0[11,1]=santo
        P0[1,11]=santo
        santo=.02
        P0[12,1]=santo
        P0[1,12]=santo
    elseif str_title=="SP1_testingset4"
        m0 = [2e8, 100.0, 4.9, 0.0, 0.31, 80.6] #xtest[1,:]
        prms = [0.0042107785, 0.4498021f-9, 1.2175618f-10, 7.04629337f-10, 4.5986106f-11, 1.8797752f-8]
        P0 = Diagonal([(xtest[1,1]-sol_gt[1,1]).^2, 0.01, 0.01, 0.01, 0.01, 0.01,     (0.1)^2, (.1)^2, (0.1)^2, (0.1)^2,(0.1)^2, (.1)^2 ]);
        # println(P0) 5.193859265558679e15
        P0 = Matrix(P0)
        santo=0.000002
        P0[8,1]=santo
        P0[1,8]=santo
        santo=0.0011
        P0[9,1]=santo
        P0[1,9]=santo
        santo=0.00056
        P0[10,1]=santo
        P0[1,10]=santo
        santo=.00012
        P0[11,1]=santo
        P0[1,11]=santo
        santo=0.02
        P0[12,1]=santo
        P0[1,12]=santo


    elseif str_title=="SP2_testingset3"
        m0 = [2e8, 29.1, 9.0, 0.0, 0.31, 80.6] #xtest[1,:]
        prms = [0.0042107785, 0.0027621f-9, 0.7175618f-10, 0.000043337f-10, 7.3986106f-11, 1.8797752f-8]
        P0 = Diagonal([(xtest[1,1]-sol_gt[1,1]).^2, 0.01, 0.01, 0.01, 0.01, 0.01,     (0.1)^2, (.1)^2, (0.1)^2, (0.1)^2,(0.1)^2, (.1)^2 ]);
         # println(P0) 3.5999349604594725e14
         P0 = Matrix(P0)
         santo=0.008
         P0[8,1]=santo
         P0[1,8]=santo
         santo=0.0011
         P0[9,1]=santo
         P0[1,9]=santo
         santo=0.015
         P0[10,1]=santo
         P0[1,10]=santo
         santo=.000115
         P0[11,1]=santo
         P0[1,11]=santo
         santo=.000025
         P0[12,1]=santo
         P0[1,12]=santo
    elseif str_title=="SP2_testingset4"
        m0 = [2e8, 29.1, 9.0, 0.0, 0.31, 80.6] #xtest[1,:]
        # prms = [0.0042107785, 0.267621f-9, 0.2175618f-10, 0.0029337f-10, 7.3986106f-11, 1.0797752f-8]
        prms = [0.0042107785, 0.0027621f-9, 0.7175618f-10, 0.000043337f-10, 7.3986106f-11, 1.8797752f-8]
        P0 = Diagonal([3.5999349604594725e14, 0.01, 0.01, 0.01, 0.01, 0.01,     (0.1)^2, (.1)^2, (0.1)^2, (0.1)^2,(0.1)^2, (.1)^2 ]);
         # println(P0) 2.7701239461354212e14
        P0 = Matrix(P0)
        santo=0.008
        P0[8,1]=santo
        P0[1,8]=santo
        santo=0.0011
        P0[9,1]=santo
        P0[1,9]=santo
        santo=0.015
        P0[10,1]=santo
        P0[1,10]=santo
        santo=.000115
        P0[11,1]=santo
        P0[1,11]=santo
        santo=.000025
        P0[12,1]=santo
        P0[1,12]=santo

    elseif str_title=="SP3_testingset3"
        m0 = [2e8, 45.0, 10.0, 0.0, 0.31, 80.6] #xtest[1,:]
        # prms = [0.0042107785, 0.267621f-9, 1.1575618f-10,  4.498475f-10, 8.9686106f-11, 1.0797752f-8]
        prms = [0.0042107785, 0.247621f-9, 1.1575618f-10,  4.498475f-10, 8.9686106f-11, 1.7797752f-8]
        P0 = Diagonal([3.4352887631466495e15, 0.01, 0.01, 0.01, 0.01, 0.01,     (0.1)^2, (.1)^2, (0.1)^2, (0.1)^2,(0.1)^2, (.1)^2 ]);
        # println(P0) 3.0168394873719065e15
        P0 = Matrix(P0)
        santo=0.04
        P0[8,1]=santo
        P0[1,8]=santo
        santo=0.001
        P0[9,1]=santo
        P0[1,9]=santo
        santo=0.015
        P0[10,1]=santo
        P0[1,10]=santo
        santo=.00011
        P0[11,1]=santo
        P0[1,11]=santo
        santo=.000001
        P0[12,1]=santo
        P0[1,12]=santo
    elseif str_title=="SP3_testingset4"
        m0 = [2e8, 45.0, 10.0, 0.0, 0.31, 80.6] #xtest[1,:]
        prms = [0.0042107785, 0.247621f-9, 1.1575618f-10,  4.498475f-10, 8.9686106f-11, 1.7797752f-8]
        P0 = Diagonal([(xtest[1,1]-sol_gt[1,1]).^2, 0.01, 0.01, 0.01, 0.01, 0.01,     (0.1)^2, (.1)^2, (0.1)^2, (0.1)^2,(0.1)^2, (.1)^2 ]);
        # println(P0) 3.4352887631466495e15
        P0 = Matrix(P0)
        santo=0.04
        P0[8,1]=santo
        P0[1,8]=santo
        santo=0.001
        P0[9,1]=santo
        P0[1,9]=santo
        santo=0.015
        P0[10,1]=santo
        P0[1,10]=santo
        santo=.00011
        P0[11,1]=santo
        P0[1,11]=santo
        santo=.000001
        P0[12,1]=santo
        P0[1,12]=santo

    elseif str_title=="SP4_testingset3"
        m0 = [2e9, 100.0, 25.0, 0.0, 0.31, 80.6] #xtest[1,:]
        prms =[0.040805906, 0.257621f-9, 10.0469596f-11, 6.729657f-10, 1.2207346f-10, 1.2906075f-8]
        P0 = Diagonal([(xtest[1,1]-sol_gt[1,1]).^2, 0.01, 0.01, 0.01, 0.01, 0.01,     (0.1)^2, (.1)^2, (0.1)^2, (0.1)^2,(0.1)^2, (.1)^2 ]);
        # println(P0) 2.1940411208764572e16
        P0 = Matrix(P0)
        santo=0.0001
        P0[8,1]=santo
        P0[1,8]=santo
        santo=0.001
        P0[9,1]=santo
        P0[1,9]=santo
        santo=0.055
        P0[10,1]=santo
        P0[1,10]=santo
        santo=.011
        P0[11,1]=santo
        P0[1,11]=santo
        santo=.000514
        P0[12,1]=santo
        P0[1,12]=santo
    elseif str_title=="SP4_testingset4"
        m0 = [2e9, 100.0, 25.0, 0.0, 0.31, 80.6] #xtest[1,:]
        # prms =[0.040805906, 0.247621f-9, 10.0469596f-11, 10.1529657f-10, 1.80207346f-10, 1.2906075f-8]
        prms =[0.040805906, 0.257621f-9, 10.0469596f-11, 6.729657f-10, 1.2207346f-10, 1.2906075f-8]
        P0 = Diagonal([2.1940411208764572e16, 0.01, 0.01, 0.01, 0.01, 0.01,     (0.1)^2, (.1)^2, (0.1)^2, (0.1)^2,(0.1)^2, (.1)^2 ]);
        # println(P0) 1.084587716549068e16
        P0 = Matrix(P0)
        santo=0.0001
        P0[8,1]=santo
        P0[1,8]=santo
        santo=0.001
        P0[9,1]=santo
        P0[1,9]=santo
        santo=0.055
        P0[10,1]=santo
        P0[1,10]=santo
        santo=.011
        P0[11,1]=santo
        P0[1,11]=santo
        santo=.000514
        P0[12,1]=santo
        P0[1,12]=santo

    end




    m = [m0;prms]
    P = P0
    m2 = [m0;prms]
    P2 = P0
    # prms=[0.04216068, 3.070513f-10, 9.069877f-11, 4.1235845f-10, 0.6635183f-10, 1.4880681f-8]
    # m = [m0;prms]
    # P = P0
    # m2 = [m0;prms]
    # P2 = P0



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
    kalman = zeros(size(m, 1), length(tgrid_opt[2:end]));

    NIS_UKF=[]
    NIS_ENN_MM_UKF=[]

    NEES_UKF=[]
    NEES_ENN_MM_UKF=[]

    NEES_UKF_total=[]
    NEES_ENN_MM_UKF_total=[]


    for k=1:steps
    ##### Classic UKF
        #estimate the best parameters(t) for the dynamic system based on states(t) updated by UKF using Xv(t) observed.
        # p=ensemble_prediction(ensemble_MLP,u0_for_params3) #parameters(t) with value ± std

        # Form the sigma points for dynamic model
        # SX = repeat(m2, 1, 2 * n) + cholesky(Hermitian(P2)).L * XI
        A = cholesky(Hermitian(P2)).L #cholesky(P'P).L
        SX = [zeros(size(m2)) A -A]
        SX = sqrt(n + lambda) .* SX .+ repeat(m2, 1, size(SX, 2))

        # Propagate through the dynamic model
        # μ_Xv, μ_GLC, μ_GLN, μ_LAC, μ_AMM, μ_mAb = p[1]
        # println(μ_Xv, μ_GLC, μ_GLN, μ_LAC, μ_AMM, μ_mAb)

        HX =hcat(SX[1,:] .+ (SX[7,:].*SX[1,:]).*DT,
                SX[2,:] .+ (-SX[8,:].*SX[1,:]).*DT,
                SX[3,:] .+ (-SX[9,:].*SX[1,:]).*DT,
                SX[4,:] .+ (SX[10,:].*SX[1,:]).*DT,
                SX[5,:] .+ (SX[11,:].*SX[1,:]).*DT,
                SX[6,:] .+ (SX[12,:].*SX[1,:]).*DT,
                SX[7,:] .+ 0,
                SX[8,:] .+ 0,
                SX[9,:] .+ 0,
                SX[10,:] .+ 0,
                SX[11,:] .+ 0,
                SX[12,:] .+ 0)'

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

        kalman[:, k] = K
        MM2[:, k] = m2
        PP2[:, :, k] = P2

        u0_for_params3=[m2;tgrid_opt[k+1]]
        u0_for_pred3=m2

        append!(NIS_UKF,ie*inv(S)*ie)



    end


      #plots
    lws=2.5


    gr( xtickfontsize=7, ytickfontsize=7, xguidefontsize=9, yguidefontsize=9, legendfontsize=6);
    ppp=plot(tgrid_opt[2:end], MM2'[:,7:12], label = "params", grid=false,fillalpha=.3,color=:blue, lw=lws,layout=(3,2))
    display(ppp)
    ppp=plot(tgrid_opt[2:end], kalman'[:,7:12], label = "kalman", grid=false,fillalpha=.3,color=:blue, lw=lws,layout=(3,2))
    display(ppp)
    ppp=Plots.scatter(tgrid_opt,xtest[:,1],title=str_title,color=:indianred1, markerstrokewidth = 0, lw=lws, label = "noise dt", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
    plot!(Array(0:0.125:103),sol_gt,color=:green , lw=lws, label = "true", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
    plot!(tgrid_opt[2:end], MM2'[:,1:6], label = "UKF", grid=false,fillalpha=.3,color=:blue, lw=lws,layout=(3,2))
    display(ppp)
    # savefig("ENN+UKF_"*str_title)

    # plots=Plots.scatter(tgrid_opt,xtest[:,1],title=str_title,color=:indianred1, markerstrokewidth = 0, lw=lws, label = "noise dt", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
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

    # chain_df = DataFrame(MM', :auto)
    # CSV.write(str_title*"HNKE-U.csv",chain_df)

    chain_df = DataFrame(MM2', :auto)
    CSV.write(str_title*"JUKF-SANTO.csv",chain_df)


    println("RMSPE UKF for", str_title)
    println("Xv: ",RMSPE(sol_gt[2:end,1],MM2'[:,1]))
    println("glc: ",RMSPE(sol_gt[2:end,2].+10.0,MM2'[:,2].+10.0))
    println("gln: ",RMSPE(sol_gt[2:end,3].+10.0,MM2'[:,3].+10.0))
    println("lac: ",RMSPE(sol_gt[2:end,4].+10.0,MM2'[:,4].+10.0))
    println("amm: ",RMSPE(sol_gt[2:end,5],MM2'[:,5]))
    println("mAb: ",RMSPE(sol_gt[2:end,6],MM2'[:,6]))


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


    # mean_NIS=mean(NIS_ENN_MM_UKF)
    # N=length(NIS_ENN_MM_UKF)
    # println("\n====== Normalised innovations squared Chi2 test ===NIS_ENN_MM_UKF===", str_title)
    # println("Is N*mean(NIS) inside of 95% CI ($(r1) < $(N*mean_NIS) < $(r2))?", r1 < N*mean_NIS < r2)
    # # println("r1 = ",r1)
    # # println("r2 = ",r2)
    # # #println("mean(NIS): ",mean_NIS)
    # # println("N*mean_NIS = ",N*mean_NIS)
    # # #println("N = ",N)


end















#
