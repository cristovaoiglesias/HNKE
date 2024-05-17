using  Plots,Distributions,DifferentialEquations
using BSON#: @load
using Statistics
using DataFrames
using Flux
include("/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/estimating_parameters/mAb_synthetic_dataset.jl")
# using .mAb_synthetic_dt

path="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/ensemble/trained_Ensemble_MLP/"

#load all trained sub-MLP that compose the ensemble
ensemble_MLP = Dict()
ensemble_size=100
for i=1:ensemble_size
    pt=path*"$(i)_sub_MLP.bson"
    m = BSON.load(pt, @__MODULE__)
    ensemble_MLP[i]=m[:model]
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

function ode_system!(du, u, p, t)
    Xv, GLC, GLN, LAC, AMM, mAb = u
    μ_Xv, μ_GLC, μ_GLN, μ_LAC, μ_AMM, μ_mAb = p
    du[1] = μ_Xv*Xv  #dXv
    du[2] = -μ_GLC*Xv #dGLC
    du[3] = -μ_GLN*Xv #dGLN
    du[4] = μ_LAC*Xv #+ klac1*GLC + klac2*GLN   #dLAC
    du[5] = μ_AMM*Xv  #- klac1*GLC  #μ_AMM*Xv - kdeg*du[3] #dAMM #(eq10: dAMM)
    du[6] = μ_mAb*Xv
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



xtest=[[3.6320008991478646e8, 28.497291405785344, 4.873418768673552, 2.9471089418951237, 0.41053380648945065, 22.5356383543335, 0.0],
[3.9714333547558784e8, 29.497868186283487, 4.173958582884101, 4.881254750599023, 0.07245625844779557, 85.5350152123526, 7.0],
[4.380118651472328e8, 26.74829263797663, 3.728799856476061, 2.068246964408017, 1.00234821711578, 150.39795059733933, 14.0],
[2.3099172908308578e8, 19.889463215472247, 3.0601741044408395, 5.561978856925664, 1.0851387835719901, 99.99169285678953, 21.0],
[7.13729015489268e8, 27.414327631969986, 2.592863335212839, 7.902747515131178, 1.716433169077, 225.97113778652317, 28.0],
[9.20496865537368e8, 16.188369667779064, 2.615233422962437, 9.136267731320556, 2.5031432290964997, 196.78370060714204, 35.0],
[1.0878506223072157e9, 19.331820670636834, 1.2782991928819532, 19.294997645677675, 2.9614686873248535, 496.3207811284528, 42.0],
[1.3507814148902016e9, 14.203351689046015, 0.6520047831620861, 20.4120325493498, 2.843475627112798, 537.1455895540505, 49.0],
[1.307739914145754e9, 12.070407370287624, 0.02082923828446201, 24.91153824384828, 3.415373004303232, 603.8430080544445, 56.0],
[1.3419300184944696e9, 11.98399219128823, 0.5401372308555539, 22.61919630678117, 3.2764559601458436, 748.8542438122906, 63.0],
[1.012881548192765e9, 10.469088709001337, 0.2472216784346681, 22.15894188217364, 2.5557607090890215, 933.6256525443919, 70.0],
[7.72729271058243e8, 12.497223392119706, 0.19296288598514613, 28.995460645474495, 3.508573048990084, 1252.9007532825858, 77.0],
[7.227034864444705e8, 10.66755181894824, 0.2247322787263362, 20.752911616017734, 2.5225785076904126, 1106.1656253840015, 84.0],
[7.060915689415284e8, 14.343661989031641, 0.6156789301675722, 26.643767728054748, 3.8214463203881532, 1246.98942511289, 91.0]]



xdt=[(mAb_synthetic_dt.sol_SPN_trainingset1[:,:],sol_SPN_gt),
(mAb_synthetic_dt.sol_SPN_trainingset2[:,:],sol_SPN_gt),
(mAb_synthetic_dt.sol_SPN_trainingset3[:,:],sol_SPN_gt),
(mAb_synthetic_dt.sol_SPN_testingset1[:,:],sol_SPN_gt),
(mAb_synthetic_dt.sol_SPN_testingset2[:,:],sol_SPN_gt),
# (mAb_synthetic_dt.sol_SPN_testingset3[:,:],sol_SPN_gt),
# (mAb_synthetic_dt.sol_SPN_testingset4[:,:],sol_SPN_gt),

(mAb_synthetic_dt.sol_SP1_trainingset1[:,:],sol_SP1_gt),
(mAb_synthetic_dt.sol_SP1_trainingset2[:,:],sol_SP1_gt),
(mAb_synthetic_dt.sol_SP1_trainingset3[:,:],sol_SP1_gt),
(mAb_synthetic_dt.sol_SP1_testingset1[:,:],sol_SP1_gt),
(mAb_synthetic_dt.sol_SP1_testingset2[:,:],sol_SP1_gt),
# (mAb_synthetic_dt.sol_SP1_testingset3[:,:],sol_SPN_gt),
# (mAb_synthetic_dt.sol_SP1_testingset4[:,:],sol_SPN_gt),

(mAb_synthetic_dt.sol_SP2_trainingset1[:,:],sol_SP2_gt),
(mAb_synthetic_dt.sol_SP2_trainingset2[:,:],sol_SP2_gt),
(mAb_synthetic_dt.sol_SP2_trainingset3[:,:],sol_SP2_gt),
(mAb_synthetic_dt.sol_SP2_testingset1[:,:],sol_SP2_gt),
(mAb_synthetic_dt.sol_SP2_testingset2[:,:],sol_SP2_gt),
# (mAb_synthetic_dt.sol_SP2_testingset3[:,:],sol_SPN_gt),
# (mAb_synthetic_dt.sol_SP2_testingset4[:,:],sol_SPN_gt),

(mAb_synthetic_dt.sol_SP3_trainingset1[:,:],sol_SP3_gt),
(mAb_synthetic_dt.sol_SP3_trainingset2[:,:],sol_SP3_gt),
(mAb_synthetic_dt.sol_SP3_trainingset3[:,:],sol_SP3_gt),
(mAb_synthetic_dt.sol_SP3_testingset1[:,:],sol_SP3_gt),
(mAb_synthetic_dt.sol_SP3_testingset2[:,:],sol_SP3_gt),
# (mAb_synthetic_dt.sol_SP3_testingset3[:,:],sol_SPN_gt),
# (mAb_synthetic_dt.sol_SP3_testingset4[:,:],sol_SPN_gt),

(mAb_synthetic_dt.sol_SP4_trainingset1[:,:],sol_SP4_gt),
(mAb_synthetic_dt.sol_SP4_trainingset2[:,:],sol_SP4_gt),
(mAb_synthetic_dt.sol_SP4_trainingset3[:,:],sol_SP4_gt),
(mAb_synthetic_dt.sol_SP4_testingset1[:,:],sol_SP4_gt),
(mAb_synthetic_dt.sol_SP4_testingset2[:,:],sol_SP4_gt)
# (mAb_synthetic_dt.sol_SP4_testingset3[:,:],sol_SPN_gt),
# (mAb_synthetic_dt.sol_SP4_testingset4[:,:],sol_SPN_gt),
]


for e in xdt
    xtest =e[1]
    sol_gt =e[2]

    sol_gt= hcat(sol_gt[1,:], sol_gt'[:,3:end])

    tgrid_opt=[0.0, 7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, 56.0, 63.0, 70.0, 77.0, 84.0, 91.0, 98.0]
    datasetsize=14
    pred_EnMLP=[]
    for i=1:datasetsize
        u02=xtest[i,1:6]
        p=ensemble_prediction(ensemble_MLP,xtest[i,:])
        tstart=tgrid_opt[i]
        tend=tgrid_opt[i+1]

        prob =  ODEProblem(ode_system!, u02, (tstart,tend), p[1])
        sol = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid)
        push!(pred_EnMLP,sol[:,end])
    end
    pred_EnMLP=vcat(map(x->x', pred_EnMLP)...)
    lws=2.5


    gr( xtickfontsize=7, ytickfontsize=7, xguidefontsize=9, yguidefontsize=9, legendfontsize=6);
    plots=Plots.scatter(tgrid_opt,xtest[:,1:6],color=:blue , lw=lws, label = "noise dt", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
    plot!(tgrid_opt, sol_gt,color=:green , lw=lws, label = "true", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
    plot!(tgrid_opt[2:end],pred_EnMLP,color=:red , lw=lws, label = "pred", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
    display(plots)

end
