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





















## testing BHM


function unpack(nn_params::AbstractVector)
    W1 = reshape(nn_params[1:50], 10, 5);
    b1 = reshape(nn_params[51:60], 10);

    W2 = reshape(nn_params[61:160], 10, 10);
    b2 = reshape(nn_params[161:170], 10);

    W3 = reshape(nn_params[171:270], 10, 10);
    b3 = reshape(nn_params[271:280], 10);

    W4 = reshape(nn_params[281:290], 1,10 );
    b4 = reshape(nn_params[291:291], 1 );
    return W1, b1, W2, b2, W3, b3 , W4, b4
end

function nn_forward(xs, nn_params::AbstractVector)
    W1, b1, W2, b2, W3, b3 , W4, b4 = unpack(nn_params)
    MLP = Chain(
        Dense(W1, b1, softsign),
        Dense(W2, b2, softsign),
        Dense(W3, b3, softsign),
        Dense(W4, b4)     #
        # Dense(W4, b4)
    )
    return MLP(xs)
end


println("\n ************************* Testing BHM")

pathBNN_pxv="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/BHM/bayesian_mlp/pXv/chain_BayesianMLP_23 Feb 2024 4:2:355.csv"
pathBNN_pglc="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/BHM/bayesian_mlp/pGLC/chain_BayesianMLP_23 Feb 2024 4:2:901.csv"
pathBNN_pgln="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/BHM/bayesian_mlp/pGLN/chain_BayesianMLP_23 Feb 2024 0:2:737.csv"
pathBNN_plac="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/BHM/bayesian_mlp/pLAC/chain_BayesianMLP_23 Feb 2024 3:2:322.csv"
pathBNN_pamm="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/BHM/bayesian_mlp/pAMM/chain_BayesianMLP_23 Feb 2024 7:2:392.csv"
pathBNN_pmab="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/BHM/bayesian_mlp/pmAb/chain_BayesianMLP_23 Feb 2024 2:2:71.csv"

chain_BNN_pxv = CSV.read(pathBNN_pxv, DataFrame)
chain_BNN_pglc = CSV.read(pathBNN_pglc, DataFrame)
chain_BNN_pgln = CSV.read(pathBNN_pgln, DataFrame)
chain_BNN_plac = CSV.read(pathBNN_plac, DataFrame)
chain_BNN_pamm = CSV.read(pathBNN_pamm, DataFrame)
chain_BNN_pmAb = CSV.read(pathBNN_pmab, DataFrame)

# getting 200 samples around the parameters that maximise the log posterior of the model
N=2000
lp, maxInd = findmax(chain_BNN_pxv.lp[100:end])
BHM_params_xv=[]
for i in max(1, (maxInd[1]-100)):min(N, (maxInd[1]+100))
    push!(BHM_params_xv,Array(chain_BNN_pxv[100:end,:][i,1+2:291+2]))
end

lp, maxInd = findmax(chain_BNN_pglc.lp[100:end])
BHM_params_glc=[]
for i in max(1, (maxInd[1]-100)):min(N, (maxInd[1]+100))
    push!(BHM_params_glc,Array(chain_BNN_pglc[100:end,:][i,1+2:291+2]))
end

lp, maxInd = findmax(chain_BNN_pgln.lp[100:end])
BHM_params_gln=[]
for i in max(1, (maxInd[1]-100)):min(N, (maxInd[1]+100))
    push!(BHM_params_gln,Array(chain_BNN_pgln[100:end,:][i,1+2:291+2]))
end

lp, maxInd = findmax(chain_BNN_plac.lp[100:end])
BHM_params_lac=[]
for i in max(1, (maxInd[1]-100)):min(N, (maxInd[1]+100))
    push!(BHM_params_lac,Array(chain_BNN_plac[100:end,:][i,1+2:291+2]))
end

lp, maxInd = findmax(chain_BNN_pamm.lp[100:end])
BHM_params_amm=[]
for i in max(1, (maxInd[1]-100)):min(N, (maxInd[1]+100))
    push!(BHM_params_amm,Array(chain_BNN_pamm[100:end,:][i,1+2:291+2]))
end

lp, maxInd = findmax(chain_BNN_pmAb.lp)
BHM_params_mab=[]
for i in max(1, (maxInd[1]-100)):min(N, (maxInd[1]+100))
    push!(BHM_params_mab,Array(chain_BNN_pmAb[i,1+2:291+2]))
end

all_BHM_params=[BHM_params_xv,BHM_params_glc,BHM_params_gln,BHM_params_lac,BHM_params_amm,BHM_params_mab]



# # Xv,GLC,GLN,    states,     time
# X = hcat(X[1:3,:]',  X[6:6,:]',  X[7:7,:]')'
# Y = hcat(Y[3:3,:])

pparm=[]
for i=1:85
    push!(pparm,mean(chain_BNN_pxv[100:end,:][:,1+3]))
end


inpt=xdt[1][1]'
inpt_a=[]
for i=1:14
    tt=normalize_input(inpt[:,i])
    tt=hcat(tt[1:3,:]',  tt[6:6,:]',  tt[7:7,:]')'
    push!(inpt_a,tt)
end

# Xv,GLC,GLN,    states,     time
# X = hcat(inpt[1:3,:]',  inpt[6:6,:]',  inpt[7:7,:]')'
# nn_forward(X, pparm)

rrr=nn_forward(hcat(inpt_a...), Array(chain_BNN_pmAb[maxInd,1+2:291+2]))
# rrr=nn_forward(hcat(inpt_a...), Array(chain_BNN_pglc[100:end,:][maxInd,1+2:85+2]))

plot(rrr')

# rrr=nn_forward(hcat(inpt_a...),pparm)
# plot(rrr')

min_values_estimated_params=[-0.13240732597403193, -5.0243094334252736e-9, -7.242819067849792e-10, -2.075801104192634e-9, -4.1946626959321127e-10, -7.189692896472483e-8]
max_values_estimated_params=[0.2247201064149725, 6.382248486023288e-9, 9.252830910761743e-10, 3.860459201515955e-9, 1.0219287122168513e-9, 1.2732423754124306e-7]
i=6
restBNN=[]
for j=1:14
push!(restBNN,inverse_normalizetion(rrr[j],min_values_estimated_params[i],max_values_estimated_params[i]))
println(inverse_normalizetion(rrr[j],min_values_estimated_params[i],max_values_estimated_params[i]))
end



yy=[[0.04697757483881504, 1.6653490479117395e-9, 3.86688766486935e-10, 9.447041138158076e-11, 4.2477928533989273e-10, 6.026716077550061e-8],
[-0.08107546324058655, 3.6556449960585604e-9, 2.791199910543287e-10, 1.9807627690834076e-9, 1.447759647650815e-10, -3.639379762511664e-8],
[0.2247201064149725, -8.76571055667918e-10, 1.2907561474663883e-10, 7.601510313262345e-10, 3.7382478497752586e-11, 3.7569780282384976e-8],
[-0.07411731833854303, 5.271379978248598e-10, 2.442886963433655e-10, 6.337932404153123e-10, -5.736086732119152e-11, 1.5509920065475417e-8],
[0.08588002251355902, 5.015371963609382e-10, 2.3009907681295157e-11, -2.9087146227111396e-10, 2.1318464591445362e-10, -3.29338952113673e-8],
[0.008844466684061981, 7.498680457582508e-10, 8.017503297866003e-11, 8.607662647189983e-10, 8.067740163796432e-11, 4.071496558767031e-8],
[0.015201235241261817, -1.680902408967688e-10, 9.665983547890558e-11, 1.0971107020574699e-9, 5.039444463466816e-11, 1.803843825118847e-8],
[0.018971961814644823, 5.809923750617929e-10, 7.398023030967056e-11, 4.2152889122624615e-10, 5.172934984295373e-11, 2.020067160203011e-8],
[-0.0074909012126365675, -3.017585090610922e-10, -2.2138044155577198e-11, -2.4739974246411057e-11, -4.3972588648062644e-11, 1.4271589241955448e-8],
[-0.041043655732285286, 3.635034430540076e-10, -3.301130252803775e-11, -5.232306336804949e-10, -2.7341705495582088e-11, 1.968964778140048e-8],
[-0.013304616695068208, -1.6323680603211043e-10, 5.707264173210976e-11, 8.2758263479189e-11, -1.377024494446168e-11, 5.418726687870174e-10],
[-0.019498158069304713, 5.797067178624486e-10, -1.1160551167937471e-11, -2.924617911351988e-10, 3.970246328876975e-11, 1.836060443532476e-8],
[0.018367836586508098, -5.305068411666227e-10, -8.484570476343561e-12, -3.0885791005884455e-10, 1.3071636714875083e-12, 1.593038622234472e-8],
[-0.09031490828142127, 7.81319283179718e-10, 4.0823349809873876e-11, 6.660714465847812e-10, 5.7249838085465766e-12, 1.5177686387705487e-8]]





plot(hcat(yy...)[6,:])
plot!(restBNN)





path="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/BHM/tests_mlp4bnn/"

#load all trained sub-MLP that compose the ensemble
MLP4bnn = Dict()
ensemble_size=["Xv","glc","gln","lac","amm","mAb"]
for i in 1:6
    pt=path*"$(ensemble_size[i])_sub_MLP.bson"
    m = BSON.load(pt, @__MODULE__)
    MLP4bnn[i]=m[:model]

    println("$(ensemble_size[i])_sub_MLP.bson")
    for j=1:4
        println("W$(j) maximum: ",maximum(Flux.params(MLP4bnn[i][j])[1]))
        println("W$(j) minimum: ",minimum(Flux.params(MLP4bnn[i][j])[1]))
        # println("b$(j) maximum: ",maximum(Flux.params(MLP4bnn[i][j])[2]))
        # println("b$(j) minimum: ",minimum(Flux.params(MLP4bnn[i][j])[2]))
    end

    for j=1:4
        # println("W$(j) maximum: ",maximum(Flux.params(MLP4bnn[i][j])[1]))
        # println("W$(j) minimum: ",minimum(Flux.params(MLP4bnn[i][j])[1]))
        println("b$(j) maximum: ",maximum(Flux.params(MLP4bnn[i][j])[2]))
        println("b$(j) minimum: ",minimum(Flux.params(MLP4bnn[i][j])[2]))
    end
end



pt="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/BHM/tests_mlp4bnn/7inp6out_sub_MLP.bson"
m = BSON.load(pt, @__MODULE__)
mlp4bnn=m[:model]
for j=1:4
    println("W$(j) maximum: ",maximum(Flux.params(mlp4bnn[j])[1]))
    println("W$(j) minimum: ",minimum(Flux.params(mlp4bnn[j])[1]))
end
for j=1:4
    println("b$(j) maximum: ",maximum(Flux.params(mlp4bnn[j])[2]))
    println("b$(j) minimum: ",minimum(Flux.params(mlp4bnn[j])[2]))
end


#
# W1 maximum: 3.6012979
# W1 minimum: -3.9478884
# W2 maximum: 2.4976637
# W2 minimum: -2.2417054
# W3 maximum: 1.5051757
# W3 minimum: -1.8970687
# W4 maximum: 1.563563
# W4 minimum: -1.8306369
# b1 maximum: 1.0692599
# b1 minimum: -1.9203515
# b2 maximum: 1.4769539
# b2 minimum: -0.8239974
# b3 maximum: 1.1428145
# b3 minimum: -0.88792354
# b4 maximum: 0.98097247
# b4 minimum: -0.6117222
#
#
#
# sd=Float32[1.875037 1.1254121 -1.6523789 1.5346975 1.0946065 2.0236628 -1.781368; 1.1849939 2.4872336 0.16947997 0.24015091 0.76725644 1.0294813 0.008388019; 0.19933994 0.041592497 0.9930616 3.3768384 -0.16517787 -0.8620704 1.1798269; 0.6621956 0.5697205 1.6394995 0.62589717 -1.2822636 -0.524545 -1.0280635; 0.75996464 1.2655748 2.1498165 -3.9478884 -1.0966669 2.515487 1.6903596; -1.2811621 1.2494835 -1.1632073 -1.6594534 -0.28918305 0.67527246 -1.0473766; 0.71244764 -0.027102798 1.3396991 1.3735096 1.1200949 -0.49458805 -0.62206423; -0.11508237 -0.01922267 1.5188955 1.3123283 -1.5188425 1.9113997 -1.1606687; 1.1261019 -1.1684437 0.41439736 -0.26806295 0.95478123 0.20197538 -2.5004952; 0.027464751 -0.6694523 0.30190885 2.1730852 -1.3137022 1.3764294 0.59978926; -0.79648256 1.7893125 0.790273 -0.38281518 -0.62346935 -2.2573824 -1.6544231; -2.0673068 0.08959968 -1.2679441 -0.18318911 0.47558436 2.3448966 -0.60707825; 0.27717242 1.3110815 0.73451287 0.12156324 -0.08354788 2.129595 0.49636868; -0.5185398 0.8297732 1.4335537 0.7593017 1.6420763 -1.1676195 1.8847827; -2.9929762 -1.693472 1.7082407 0.5164434 0.7216979 0.34592497 -0.29277512; -2.4589424 -1.5406679 1.5259192 -0.50386685 0.1312101 -0.06828885 0.7239177; 0.32808805 2.0651217 1.3987501 0.9505764 0.21946955 0.42859954 0.30438864; -0.4769967 -0.1268065 -0.1480291 -1.02924 0.35529235 0.38330525 -1.8763349; 0.048549738 0.030717384 -0.015512208 -1.2850326 1.6933136 0.027674092 -2.0974805; -1.9203032 1.3332775 -1.0716028 1.0925014 -1.6358042 -0.42013207 0.32777017; -1.7994617 0.36843354 0.9200355 1.6104907 -2.8435357 1.0258245 0.82370746; -0.35568294 -0.8280433 0.48461705 1.5055448 -2.0833967 0.3492692 -0.7755077; 0.2661939 -2.2464447 -1.1077467 0.8937257 1.2601322 2.5857713 1.4310482; -1.8176641 -0.13798366 0.34108278 0.045400903 1.8629364 0.90406436 -0.8243387; 2.3616302 -1.7160623 -0.51216495 -0.9347112 1.1020021 -1.1571113 -0.46800578; -1.4990104 0.58605605 -0.7178554 2.190985 1.9242573 -1.3922927 -0.13934413; 0.18476129 -0.58578783 -0.34979632 -0.5602423 2.241023 0.7317399 0.791364; 1.63639 -0.43810016 -0.12454961 2.9972796 1.1618222 3.285467 -0.97797906; 2.7793767 0.76737785 1.2944779 0.46054885 1.7080824 0.1778659 -1.1738527; -0.79900515 -0.71591353 1.5848976 0.55842113 0.80082273 0.8786477 -2.3495617; 3.6012979 0.063776866 0.5981749 -2.592507 -1.0484397 2.325337 -0.12874882; 0.74201524 -0.38235232 -2.0198178 -0.31031093 -1.541383 2.096844 -0.29958856]
#

#
#
#
#
#
#
# #predictions with Ensemble of MLP.
# for e in xdt
#     xtest =e[1]
#     sol_gt =e[2]
#
#     sol_gt= hcat(sol_gt[1,:], sol_gt'[:,3:end])
#     tgrid_opt=[0.0, 7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, 56.0, 63.0, 70.0, 77.0, 84.0, 91.0, 98.0]
#     datasetsize=14
#
#     pred_EnMLP=[]
#     for i=1:datasetsize
#         u02=xtest[i,1:6] #state (t)
#         p=ensemble_prediction(ensemble_MLP,xtest[i,:]) #parameters(t) with value ± std
#         tstart=tgrid_opt[i]
#         tend=tgrid_opt[i+1]
#         pp = Measurement{Float64}[]
#         for j=1:length(p[1])
#             push!(pp,measurement(p[1][j], p[2][j]))#parameters with value ± std
#         end
#         prob =  ODEProblem(ode_system, u02, (tstart,tend),pp)
#         sol = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid) #states (t+1) with uncertainty based on states (t) and parameters(t) with value ± std
#         push!(pred_EnMLP,sol[:,end])
#     end
#     pred_EnMLP=vcat(map(x->x', pred_EnMLP)...)
#
#     #plots
#     lws=2.5
#     gr( xtickfontsize=7, ytickfontsize=7, xguidefontsize=9, yguidefontsize=9, legendfontsize=6);
#     plots=plot(layout = (3, 2), size = (800, 700))
#     for i = 1:6
#         plot!(plots[i], tgrid_opt[2:end], Measurements.value.(pred_EnMLP[:, i]), ribbon = Measurements.uncertainty.(pred_EnMLP[:, i]), color = :blue, fillalpha=.075, lw = lws, label = "pred", ylabel = "[$(i)]")
#     end
#     # plot(tgrid_opt[2:end],pred_EnMLP,color=:black ,lw=lws, label = "pred", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
#     Plots.scatter!(tgrid_opt,xtest[:,1:6],color=:red, markerstrokewidth = 0, lw=lws, label = "noise dt", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
#     plot!(tgrid_opt, sol_gt,color=:green , lw=lws, label = "true", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
#     display(plots)
# end



/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/BHM/chain_BayesianMLP_23 Feb 2024 11:2:553.csv


pathBNN_frozen="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/BHM/chain_BayesianMLP_23 Feb 2024 11:2:553.csv"

chain_BNN_frozen = CSV.read(pathBNN_frozen, DataFrame)


# getting 200 samples around the parameters that maximise the log posterior of the model
N=2000
lp, maxInd = findmax(chain_BNN_frozen.lp[100:end])
BHM_params_xv=[]
for i in max(1, (maxInd[1]-100)):min(N, (maxInd[1]+100))
    push!(BHM_params_xv,Array(chain_BNN_pxv[100:end,:][i,1+2:291+2]))
end
