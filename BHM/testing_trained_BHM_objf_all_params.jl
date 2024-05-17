using  Plots,Distributions,DifferentialEquations
using BSON#: @load
using Statistics
using CSV, DataFrames
using Flux
using Measurements, StaticArrays




include("/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/mAb_synthetic_dataset.jl")
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
#
# function ode_system(u, p, t)
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
# end



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

#predictions with Ensemble of MLP.
for e in xdt
    xtest =e[1]
    sol_gt =e[2]

    sol_gt= hcat(sol_gt[1,:], sol_gt'[:,3:end])

    tgrid_opt=[0.0, 7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, 56.0, 63.0, 70.0, 77.0, 84.0, 91.0, 98.0]
    datasetsize=14
    pred_EnMLP=[]
    for i=1:datasetsize
        u02=xtest[i,1:6] #state (t)
        p=ensemble_prediction(ensemble_MLP,xtest[i,:]) #parameters(t) with value ± std
        tstart=tgrid_opt[i]
        tend=tgrid_opt[i+1]
        pp = Measurement{Float64}[]
        for j=1:length(p[1])
            push!(pp,measurement(p[1][j], p[2][j]))#parameters with value ± std
        end
        prob =  ODEProblem(ode_system, u02, (tstart,tend),pp)
        sol = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid) #states (t+1) with uncertainty based on states (t) and parameters(t) with value ± std
        push!(pred_EnMLP,sol[:,end])
    end

    pred_EnMLP=vcat(map(x->x', pred_EnMLP)...)

    #plots
    lws=2.5
    gr( xtickfontsize=7, ytickfontsize=7, xguidefontsize=9, yguidefontsize=9, legendfontsize=6);
    plots=plot(layout = (3, 2), size = (800, 700))
    for i = 1:6
        plot!(plots[i], tgrid_opt[2:end], Measurements.value.(pred_EnMLP[:, i]), ribbon = Measurements.uncertainty.(pred_EnMLP[:, i]), color = :blue, fillalpha=.075, lw = lws, label = "pred", ylabel = "[$(i)]")
    end
    # plot(tgrid_opt[2:end],pred_EnMLP,color=:black ,lw=lws, label = "pred", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
    Plots.scatter!(tgrid_opt,xtest[:,1:6],color=:red, markerstrokewidth = 0, lw=lws, label = "noise dt", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
    plot!(tgrid_opt, sol_gt,color=:green , lw=lws, label = "true", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
    display(plots)
end





































































println("\n ************************* Testing BHM")

## testing BHM


function unpack(nn_params::AbstractVector)
    W1 = reshape(nn_params[1:224], 32, 7);
    b1 = reshape(nn_params[225:256], 32);

    W2 = reshape(nn_params[257:1280], 32, 32);
    b2 = reshape(nn_params[1281:1312], 32);

    W3 = reshape(nn_params[1313:2336], 32, 32);
    b3 = reshape(nn_params[2337:2368], 32);

    W4 = reshape(nn_params[2369:2560], 6,32 );
    b4 = reshape(nn_params[2561:2566], 6 );

    return W1, b1, W2, b2, W3, b3, W4, b4
end

function nn_forward(xs, nn_params::AbstractVector)
    W1, b1, W2, b2, W3, b3, W4, b4 = unpack(nn_params)
    MLP = Chain(                  # Input layer: 7 features,  σ activation function
        Dense(W1, b1, softsign),     # 7 inputs×32 outputs=224 weights weights and bias=32
        Dense(W2, b2, softsign),     # 32×32=1024 weights and bias=32
        Dense(W3, b3, softsign),     # 32×32=1024 weights and bias=32
        Dense(W4, b4)                # 32×6=192 weights and bias=6
    )
    return MLP(xs)
end

function BNN_prediction(estimated_params,inpt) # make predictions with MLP ensemble where input is states(t) and output params(t)
    #the input is normalized and the output is unnromalized.
    row_prediction=[]
    u0=normalize_input(inpt)
    for j=1:length(estimated_params)
        p=nn_forward(u0,estimated_params[j])
        #p=ensemble[j](u0)
        p=unnormalize_output(p)
        push!(row_prediction,p)
    end
    rp=vcat(map(x->x', row_prediction)...)
    rp_mean= [mean(rp[:,1]),mean(rp[:,2]),mean(rp[:,3]),mean(rp[:,4]),mean(rp[:,5]),mean(rp[:,6])]
    rp_std = [std(rp[:,1]),std(rp[:,2]),std(rp[:,3]),std(rp[:,4]),std(rp[:,5]),std(rp[:,6])]
    return rp_mean,rp_std #return mean and std
end


# nn_forward(normalize_input(xdt[1][1][1,:]),BHM_params_frozen_obj[1])










#inference for all params as obj function
#bmlp.jl
pathBNN_all_params_objF="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/BHM/bnn_as_obj/chain_BayesianMLP_25 Feb 2024 17:2:469.csv"
bnn_all_params_objF = CSV.read(pathBNN_all_params_objF, DataFrame)


sz=10
N=1000
lp, maxInd = findmax(bnn_all_params_objF.lp[1:end])
bnn_all_params_OF=[]
for i in max(1, (maxInd[1]-sz)):min(N, (maxInd[1]+sz))
    push!(bnn_all_params_OF,Array(bnn_all_params_objF[1:end,:][i,1+2:2566+2]))
end


lp, maxInd = findmax(bnn_all_params_objF.lp[1:end])
bnn_all_params_OF=[Array(bnn_all_params_objF[1:end,:][maxInd,1+2:2566+2])]
#
#
# lp, maxInd = findmax(frozen_obj.lp[100:end])
# bnn_all_params_OF=[]
# for i=4980:N
#     push!(bnn_all_params_OF,Array(frozen_obj[i,1+2:192+2]))
# end
#

# tmp=[]
# for j=1:2566
#     push!(tmp,mean(bnn_all_params_objF[700:end,:][:,j+2]))
# end
# bnn_all_params_OF=[tmp]


for e in xdt
    xtest =e[1]
    sol_gt =e[2]

    sol_gt= hcat(sol_gt[1,:], sol_gt'[:,3:end])

    tgrid_opt=[0.0, 7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, 56.0, 63.0, 70.0, 77.0, 84.0, 91.0, 98.0]
    datasetsize=14
    pred_BNN=[]
    for i=1:datasetsize
        u02=xtest[i,1:6] #state (t)
        p=BNN_prediction(bnn_all_params_OF,xtest[i,:]) #parameters(t) with value ± std

        tstart=tgrid_opt[i]
        tend=tgrid_opt[i+1]
        pp = Measurement{Float64}[]
        for j=1:length(p[1])
            push!(pp,measurement(p[1][j], p[2][j]))#parameters with value ± std
        end
        prob =  ODEProblem(ode_system, u02, (tstart,tend),pp)
        sol = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid) #states (t+1) with uncertainty based on states (t) and parameters(t) with value ± std
        push!(pred_BNN,sol[:,end])
    end

    pred_BNN=vcat(map(x->x', pred_BNN)...)

    #plots
    lws=2.5
    gr( xtickfontsize=7, ytickfontsize=7, xguidefontsize=9, yguidefontsize=9, legendfontsize=6);
    # pyplot();

    plots=plot(layout = (3, 2), size = (800, 700))
    for i = 1:6
        plot!(plots[i], tgrid_opt[2:end], Measurements.value.(pred_BNN[:, i]), ribbon = Measurements.uncertainty.(pred_BNN[:, i]), color = :blue, fillalpha=.075, lw = lws, label = "pred", ylabel = "[$(i)]")
    end
    # plot(tgrid_opt[2:end],pred_BNN,color=:black ,lw=lws, label = "pred", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
    Plots.scatter!(tgrid_opt,xtest[:,1:6],color=:red, markerstrokewidth = 0, lw=lws, label = "noise dt", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
    plot!(tgrid_opt, sol_gt,color=:green , lw=lws, label = "true", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
    display(plots)
end
