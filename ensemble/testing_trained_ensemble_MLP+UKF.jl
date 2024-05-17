# using  Plots,Distributions,DifferentialEquations
# using BSON#: @load
# using Statistics
# using DataFrames
# using Flux
# using Measurements, StaticArrays
# # Colors
# # https://github.com/JuliaGraphics/Colors.jl/blob/master/src/names_data.jl
#
#
#
# include("/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/mAb_synthetic_dataset.jl")
# # using .mAb_synthetic_dt
#
# path="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/ensemble/trained_Ensemble_MLP/"
#
# #load all trained sub-MLP that compose the ensemble
# ensemble_MLP = Dict()
# ensemble_size=100
# for i=1:ensemble_size
#     pt=path*"$(i)_sub_MLP.bson"
#     m = BSON.load(pt, @__MODULE__)
#     ensemble_MLP[i]=m[:model]
# end
#
#
# function normalize_data(x,min,max)
#     y=(x-min)/(max-min)
#     return y
# end
#
# function inverse_normalizetion(y,min,max)
#     x=(y*(max-min))+min
#     return x
# end
#
# function normalize_input(u0)
#     #the max and min values were obtained from the dataset used to train the Ensemble, see file dataset_to_train_EnMLP.jl
#     min_values_state_variables=[1.1247355623692597e8, 1.0876934495531811, 0.0016493210047543638, 0.0, 0.002675178467605843, 13.163672825054604, 0.0]
#     max_values_state_variables=[3.9843740973885384e9, 100.72750841246663, 25.078825296718655, 124.32540101966985, 16.566031188582603, 4632.744925522347, 91.0]
#     for i=1:7
#         u0[i]=normalize_data(u0[i],min_values_state_variables[i],max_values_state_variables[i])
#     end
#     return u0
# end
#
# function unnormalize_output(p)
#     #the max and min values were obtained from the dataset used to train the Ensemble, see file dataset_to_train_EnMLP.jl
#     min_values_estimated_params=[-0.13240732597403193, -5.0243094334252736e-9, -7.242819067849792e-10, -2.075801104192634e-9, -4.1946626959321127e-10, -7.189692896472483e-8]
#     max_values_estimated_params=[0.2247201064149725, 6.382248486023288e-9, 9.252830910761743e-10, 3.860459201515955e-9, 1.0219287122168513e-9, 1.2732423754124306e-7]
#     for i=1:6
#         p[i]=inverse_normalizetion(p[i],min_values_estimated_params[i],max_values_estimated_params[i])
#     end
#     return p
# end
#
#
# function ensemble_prediction(ensemble,inpt) # make predictions with MLP ensemble where input is states(t) and output params(t)
#     #the input is normalized and the output is unnromalized.
#     row_prediction=[]
#     u0=normalize_input(inpt)
#     for j=1:length(ensemble)
#         p=ensemble[j](u0)
#         # println(p)
#         p=unnormalize_output(p)
#         push!(row_prediction,p)
#     end
#     rp=vcat(map(x->x', row_prediction)...)
#     rp_mean= [mean(rp[:,1]),mean(rp[:,2]),mean(rp[:,3]),mean(rp[:,4]),mean(rp[:,5]),mean(rp[:,6])]
#     rp_std = [std(rp[:,1]),std(rp[:,2]),std(rp[:,3]),std(rp[:,4]),std(rp[:,5]),std(rp[:,6])]
#     return rp_mean,rp_std #return mean and std
# end
#
#
#
#
# ## making predictions with MLP ensemble where input is states(t) and the output is params(t) that enable to predict the states(t+1).
#
# tstart=0.0
# tend=103.0
# sampling= 7.0
# tgrid=tstart:sampling:tend
# sampling= 0.125
# tgrid_large=tstart:sampling:tend
# tgrid_opt=[0.0, 7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, 56.0, 63.0, 70.0, 77.0, 84.0, 91.0, 98.0]
#
# # # function ode_system(u, p, t)
# #     Xv, GLC, GLN, LAC, AMM, mAb = u
# #     μ_Xv, μ_GLC, μ_GLN, μ_LAC, μ_AMM, μ_mAb = p
# #     # du[1] = μ_Xv*Xv  #dXv
# #     # du[2] = -μ_GLC*Xv #dGLC
# #     # du[3] = -μ_GLN*Xv #dGLN
# #     # du[4] = μ_LAC*Xv #+ klac1*GLC + klac2*GLN   #dLAC
# #     # du[5] = μ_AMM*Xv  #- klac1*GLC  #μ_AMM*Xv - kdeg*du[3] #dAMM #(eq10: dAMM)
# #     # du[6] = μ_mAb*Xv
# #     du1 = μ_Xv*Xv;  #dXv
# #     du2 = -μ_GLC*Xv; #dGLC
# #     du3 = -μ_GLN*Xv; #dGLN
# #     du4 = μ_LAC*Xv; #+ klac1*GLC + klac2*GLN   #dLAC
# #     du5 = μ_AMM*Xv;  #- klac1*GLC  #μ_AMM*Xv - kdeg*du[3] #dAMM #(eq10: dAMM)
# #     du6 = μ_mAb*Xv;
# #     return SVector(du1,du2,du3,du4,du5,du6)
# # # end
#
#
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
#     return Array([du1,du2,du3,du4,du5,du6])
# end
#
#
#
#
#
#
#
# # ODE system for mAb production used in "Bioprocess optimization under uncertainty using ensemble modeling (2017)"
# function ode_system2!(du, u, p, t)
#     Xv, Xt, GLC, GLN, LAC, AMM, MAb = u
#     mu_max, Kglc, Kgln, KIlac, KIamm, mu_dmax, Kdamm, Klysis, Yxglc, mglc, Yxgln, alpha1, alpha2, Kdgln, Ylacglc, Yammgln, r1, r2 ,lambda = p
#
#     mu = mu_max*(GLC/(Kglc+GLC))*(GLN/(Kgln+GLN))*(KIlac/(KIlac+LAC))*(KIamm/(KIamm+AMM));
#     mu_d = mu_dmax/(1+(Kdamm/AMM)^2);
#
#     du[1] = mu*Xv-mu_d*Xv;  #viable cell density XV
#     du[2] = mu*Xv-Klysis*(Xt-Xv); #total cell density Xt
#     du[3] = -(mu/Yxglc+mglc)*Xv;
#     du[4] = -(mu/Yxgln+alpha1*GLN/(alpha2+GLN))*Xv - Kdgln*GLN;
#     du[5] = Ylacglc*(mu/Yxglc+mglc)*Xv;
#     du[6] = Yammgln*(mu/Yxgln+alpha1*GLN/(alpha2+GLN))*Xv+Kdgln*GLN;
#     du[7] = (r2-r1*mu)*lambda*Xv;
# end
# #parameters from the paper "Bioprocess optimization under uncertainty using ensemble modeling (2017)"
# p = [5.8e-2, 0.75, 0.075, 171.756, 28.484, 3e-2, 1.76, 0.05511, 1.061e8, 4.853e-14, 5.57e8, 3.4e-13, 4, 9.6e-3, 1.399, 4.27e-1, 0.1, 2, 7.21e-9 ]
#
# u0 = [2e8   2e8   29.1   4.9  0.0  0.310  80.6; #SPN initial condition from "Bioprocess optimization under uncertainty using ensemble modeling(2017)"
#       2e8   2e8   100    4.9  0.0  0.310  80.6; #SP1 In-silico Optimization of a Batch Bioreactor for mAbs Production in Relationship to the Net Evolution of the Hybridoma Cell Culture (2019)
#       2e8   2e8   29.1   9.0  0.0  0.310  80.6; #SP2 In-silico Optimization of a Batch Bioreactor for mAbs Production in Relationship to the Net Evolution of the Hybridoma Cell Culture (2019)
#       2e8   2e8   45.0   10   0.0  0.310  80.6; #SP3 In-silico Optimization of a Batch Bioreactor for mAbs Production in Relationship to the Net Evolution of the Hybridoma Cell Culture (2019)
#       2e9   2e8   100    25   0.0  0.310  80.6] #SP4  In-silico Optimization of a Batch Bioreactor for mAbs Production in Relationship to the Net Evolution of the Hybridoma Cell Culture (2019)
#
# prob =  ODEProblem(ode_system2!, u0[1,:], (tstart,tend), p)
# sol_SPN_gt = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid)
# sol_SPN_gt_7min = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid_large)
# prob =  ODEProblem(ode_system2!, u0[2,:], (tstart,tend), p)
# sol_SP1_gt = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid)
# sol_SP1_gt_7min = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid_large)
# prob =  ODEProblem(ode_system2!, u0[3,:], (tstart,tend), p)
# sol_SP2_gt = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid)
# sol_SP2_gt_7min = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid_large)
# prob =  ODEProblem(ode_system2!, u0[4,:], (tstart,tend), p)
# sol_SP3_gt = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid)
# sol_SP3_gt_7min = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid_large)
# prob =  ODEProblem(ode_system2!, u0[5,:], (tstart,tend), p)
# sol_SP4_gt = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid)
# sol_SP4_gt_7min = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid_large)
#
#
#
# xdt=[(mAb_synthetic_dt.sol_SPN_trainingset1[:,:],sol_SPN_gt,"SPN_trainingset1"),
# (mAb_synthetic_dt.sol_SPN_trainingset2[:,:],sol_SPN_gt,"SPN_trainingset2"),
# (mAb_synthetic_dt.sol_SPN_trainingset3[:,:],sol_SPN_gt,"SPN_trainingset3"),
# (mAb_synthetic_dt.sol_SPN_testingset1[:,:],sol_SPN_gt,"SPN_testingset1"),
# (mAb_synthetic_dt.sol_SPN_testingset2[:,:],sol_SPN_gt,"SPN_testingset2"),
# # (mAb_synthetic_dt.sol_SPN_testingset3[:,:],sol_SPN_gt_7min,"SPN_testingset3"),
# # (mAb_synthetic_dt.sol_SPN_testingset4[:,:],sol_SPN_gt_7min,"SPN_testingset4"),
#
# (mAb_synthetic_dt.sol_SP1_trainingset1[:,:],sol_SP1_gt,"SP1_trainingset1"),
# (mAb_synthetic_dt.sol_SP1_trainingset2[:,:],sol_SP1_gt,"SP1_trainingset2"),
# (mAb_synthetic_dt.sol_SP1_trainingset3[:,:],sol_SP1_gt,"SP1_trainingset3"),
# (mAb_synthetic_dt.sol_SP1_testingset1[:,:],sol_SP1_gt,"SP1_testingset1"),
# (mAb_synthetic_dt.sol_SP1_testingset2[:,:],sol_SP1_gt,"SP1_testingset2"),
# # (mAb_synthetic_dt.sol_SP1_testingset3[:,:],sol_SP1_gt_7min,"SP1_testingset3"),
# # (mAb_synthetic_dt.sol_SP1_testingset4[:,:],sol_SP1_gt_7min,"SP1_testingset4"),
#
# (mAb_synthetic_dt.sol_SP2_trainingset1[:,:],sol_SP2_gt,"SP2_trainingset1"),
# (mAb_synthetic_dt.sol_SP2_trainingset2[:,:],sol_SP2_gt,"SP2_trainingset2"),
# (mAb_synthetic_dt.sol_SP2_trainingset3[:,:],sol_SP2_gt,"SP2_trainingset3"),
# (mAb_synthetic_dt.sol_SP2_testingset1[:,:],sol_SP2_gt,"SP2_testingset1"),
# (mAb_synthetic_dt.sol_SP2_testingset2[:,:],sol_SP2_gt,"SP2_testingset2"),
# # (mAb_synthetic_dt.sol_SP2_testingset3[:,:],sol_SP2_gt_7min,"SP2_testingset3"),
# # (mAb_synthetic_dt.sol_SP2_testingset4[:,:],sol_SP2_gt_7min,"SP2_testingset4"),
#
# (mAb_synthetic_dt.sol_SP3_trainingset1[:,:],sol_SP3_gt,"SP3_trainingset1"),
# (mAb_synthetic_dt.sol_SP3_trainingset2[:,:],sol_SP3_gt,"SP3_trainingset2"),
# (mAb_synthetic_dt.sol_SP3_trainingset3[:,:],sol_SP3_gt,"SP3_trainingset3"),
# (mAb_synthetic_dt.sol_SP3_testingset1[:,:],sol_SP3_gt,"SP3_testingset1"),
# (mAb_synthetic_dt.sol_SP3_testingset2[:,:],sol_SP3_gt,"SP3_testingset2"),
# # (mAb_synthetic_dt.sol_SP3_testingset3[:,:],sol_SP3_gt_7min,"SP3_testingset3"),
# # (mAb_synthetic_dt.sol_SP3_testingset4[:,:],sol_SP3_gt_7min,"SP3_testingset4"),
#
# (mAb_synthetic_dt.sol_SP4_trainingset1[:,:],sol_SP4_gt,"SP4_trainingset1"),
# (mAb_synthetic_dt.sol_SP4_trainingset2[:,:],sol_SP4_gt,"SP4_trainingset2"),
# (mAb_synthetic_dt.sol_SP4_trainingset3[:,:],sol_SP4_gt,"SP4_trainingset3"),
# (mAb_synthetic_dt.sol_SP4_testingset1[:,:],sol_SP4_gt,"SP4_testingset1"),
# (mAb_synthetic_dt.sol_SP4_testingset2[:,:],sol_SP4_gt,"SP4_testingset2")
# # (mAb_synthetic_dt.sol_SP4_testingset3[:,:],sol_SP4_gt_7min,"SP4_testingset3"),
# # (mAb_synthetic_dt.sol_SP4_testingset4[:,:],sol_SP4_gt_7min,"SP4_testingset4"),
# ]
#
#
#
# #Offline measurements case
# #predictions with Ensemble of MLP.
# for e in xdt
#     xtest =e[1]
#     sol_gt =e[2]
#     str_title =e[3]
#     sol_gt= hcat(sol_gt[1,:], sol_gt'[:,3:end])
#
#     tgrid_opt=[0.0, 7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, 56.0, 63.0, 70.0, 77.0, 84.0, 91.0, 98.0]
#     datasetsize=14
#     pred_EnMLP=[]
#     for i=1:datasetsize
#         u02=xtest[i,1:6] #state (t)
#         p=ensemble_prediction(ensemble_MLP,xtest[i,:]) #parameters(t) with value ± std
#         tstart=tgrid_opt[i]
#         tend=tgrid_opt[i+1]
#         pp = Measurement{Float64}[]
#         for j=1:length(p[1])              #2*std
#             push!(pp,measurement(p[1][j], 1*p[2][j]))#parameters with value ± std
#         end
#         prob =  ODEProblem(ode_system, u02, (tstart,tend),pp)
#         sol = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid) #states (t+1) with uncertainty based on states (t) and parameters(t) with value ± std
#         push!(pred_EnMLP,sol[:,end])
#     end
#
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
#     Plots.scatter!(tgrid_opt,xtest[:,1:6],title=str_title,color=:red, markerstrokewidth = 0, lw=lws, label = "noise dt", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
#     plot!(tgrid_opt, sol_gt,color=:green , lw=lws, label = "true", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
#     display(plots)
#
# end
#
#
#




































#Offline measurements case
#predictions with Ensemble of MLP and correction with CKF.
for e in xdt
    # e = xdt[5:5][1]
    xtest =e[1]
    sol_gt =e[2]
    str_title =e[3]
    sol_gt= hcat(sol_gt[1,:], sol_gt'[:,3:end])

    tgrid_opt=[0.0, 7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, 56.0, 63.0, 70.0, 77.0, 84.0, 91.0, 98.0]
    steps=length(tgrid_opt)-1
    pred_EnMLP=[]
    pred_EnMLP2=[]

    # Filter parameters
    Q  = Diagonal([(1e6)^2, 100.01, 100.01, 100.01, 100.01, 100.01]);
    # Q  = Diagonal([(1e7)^2, (0.5)^2, (0.1)^2, (8)^2, (0.1)^2, (100)^2]);
    R  = (3e6)^2;
    DT = 0.125;

    P0 = Diagonal([0.01, 0.01, 0.01, 0.01, 0.01, 0.01 ]);
#     P0 = Diagonal((xtest[1,1:6]-sol_gt[1,:]).^2);
#     P0 = Diagonal(   [ 2.6634269348194384e17
#     0.36325764954020895
#     0.0007065618588301385
#     8.685451115398196
# 1000.010107046247258309
#  1.4700933187464]);
    m0 = u0_for_pred = xtest[1,1:6] #[2e8  ; 2e8 ;29.1 ; 4.9 ; 0.0 ; 0.310; 80.6; 7.21e-9]# %[0;0;0;0;0;0;0];
    u0_for_params=xtest[1,:]
    u0_for_pred2 = xtest[1,1:6] #[2e8  ; 2e8 ;29.1 ; 4.9 ; 0.0 ; 0.310; 80.6; 7.21e-9]# %[0;0;0;0;0;0;0];
    u0_for_params2=xtest[1,:]


    m = m0
    P = P0

    # Precompute the UT weights
    n = size(m, 1)
    XI = sqrt(n) * [Matrix{Float64}(I, n, n) -Matrix{Float64}(I, n, n)]
    W = ones(2 * n) / (2 * n)

    # Saving the filtering corrections
    MM = zeros(size(m, 1), length(tgrid_opt[2:end]));
    PP = zeros(size(P, 1), size(P, 2), length(tgrid_opt[2:end]));

    # run CKF
    for k=1:steps


        #estimate the best parameters(t) for the dynamic system based on Xv(t) observed and states(t) predited at t-1
        p=ensemble_prediction(ensemble_MLP,u0_for_params2) #parameters(t) with value ± std
        pp = Measurement{Float64}[]
        for j=1:length(p[1])
            push!(pp,measurement(p[1][j], 2*p[2][j]))#parameters with value ± std
        end
        # estimate the states (t+1) with uncertainty based on states (t) and parameters(t) with value ± std
        tstart=tgrid_opt[k]
        tend=tgrid_opt[k+1]
        prob =  ODEProblem(ode_system, u0_for_pred2, (tstart,tend),pp)
        sol = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid)
        push!(pred_EnMLP2,sol[:,end])
        # println(sol[:,end])
        u0_for_pred2 = [xtest[k+1,1:1];Measurements.value.(sol[2:6,end])] #[2e8  ; 2e8 ;29.1 ; 4.9 ; 0.0 ; 0.310; 80.6; 7.21e-9]# %[0;0;0;0;0;0;0];
        u0_for_params2=[xtest[k+1,1:1];Measurements.value.(sol[2:6,end]);xtest[k+1,7:7]]



        #estimate the best parameters(t) for the dynamic system based on states(t) updated by CKF using Xv(t) observed.
        p=ensemble_prediction(ensemble_MLP,u0_for_params) #parameters(t) with value ± std
        pp = Measurement{Float64}[]
        for j=1:length(p[1])
            push!(pp,measurement(p[1][j], 2*p[2][j]))#parameters with value ± std
        end
        # estimate the states (t+1) with uncertainty based on states (t) and parameters(t) with value ± std
        tstart=tgrid_opt[k]
        tend=tgrid_opt[k+1]
        prob =  ODEProblem(ode_system, u0_for_pred, (tstart,tend),pp)
        sol = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid)
        push!(pred_EnMLP,sol[:,end])
        # println(sol[:,end])

        # Form the sigma points for dynamic model
        SX = repeat(m, 1, 2 * n) + cholesky(Hermitian(P)).L * XI
        # Propagate through the dynamic model
        μ_Xv, μ_GLC, μ_GLN, μ_LAC, μ_AMM, μ_mAb = p[1]
        for i in tgrid_opt[k]:DT:tgrid_opt[k+1]
            HX =hcat(SX[1,:] + (μ_Xv.*SX[1,:]).*DT,
                    SX[2,:] + (-μ_GLC.*SX[1,:]).*DT,
                    SX[3,:] + (-μ_GLN.*SX[1,:]).*DT,
                    SX[4,:] + (μ_LAC.*SX[1,:]).*DT,
                    SX[5,:] + (μ_AMM.*SX[1,:]).*DT,
                    SX[6,:] + (μ_mAb.*SX[1,:]).*DT)'
            SX=HX[:,:]
        end

        # Compute the predicted mean and covariance
        m = zeros(size(m))
        P = zeros(size(P))
        for i in 1:size(HX, 2)
            m .+= W[i] .* HX[:, i]
        end
        for i in 1:size(HX, 2)
            P .+= W[i] .* (HX[:, i] .- m) * (HX[:, i] .- m)'
        end
        P .+= Q
        # Form sigma points for measurement step
        SX = repeat(m, 1, 2 * n) + cholesky(Hermitian(P)).L * XI
        HY = SX[1, :]
        # Compute the updated mean and covariance
        mu = zeros(1,1)
        S = zeros(1,1)
        C = zeros(size(SX, 1), 1)
        for i in 1:size(SX, 2)
            mu .+= W[i] .* HY[i]'
        end
        for i in 1:size(SX, 2)
            S = S + W[i] .* (HY[i]'.- mu) * (HY[i]'.- mu)'
            C .+= W[i] .* (SX[:, i] .- m) * (HY[i]' .- mu)'
        end
        S .+= R
        # Compute the gain and updated mean and covariance
        K = C / S
        Y=u0_obs=xtest[k+1,1]
        m .+= K * (Y .- mu)
        P .-= K * S * K'

        MM[:, k] = m
        PP[:, :, k] = P

        u0_for_params=[m;tgrid_opt[k+1]]
        u0_for_pred=m
    end

    pred_EnMLP=vcat(map(x->x', pred_EnMLP)...)
    pred_EnMLP2=vcat(map(x->x', pred_EnMLP2)...)

    #plots
    lws=2.5
    gr( xtickfontsize=7, ytickfontsize=7, xguidefontsize=9, yguidefontsize=9, legendfontsize=6);

    plots=Plots.scatter(tgrid_opt,xtest[:,1:6],title=str_title,color=:red, markerstrokewidth = 0, lw=lws, label = "noise dt", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
    plot!(tgrid_opt[2:end], MM', label = "CKF",color=:purple  , lw=lws,layout=(3,3))
    plot!(tgrid_opt, sol_gt,color=:green , lw=lws, label = "true", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))

    plot!(layout = (3, 2), size = (800, 700))
    for i = 1:6
        plot!(plots[i], tgrid_opt[2:end], Measurements.value.(pred_EnMLP[:, i]), ribbon = Measurements.uncertainty.(pred_EnMLP[:, i]), color = :blue, fillalpha=.075, lw = lws, label = "ENN+CKF", ylabel = "[$(i)]")
    end

    plot!(layout = (3, 2), size = (800, 700))
    for i = 1:6
        plot!(plots[i], tgrid_opt[2:end], Measurements.value.(pred_EnMLP2[:, i]), ribbon = Measurements.uncertainty.(pred_EnMLP2[:, i]), color = :orange, fillalpha=.075, lw = lws, label = "ENN", ylabel = "[$(i)]")
    end
    # plot(tgrid_opt[2:end],pred_EnMLP,color=:black ,lw=lws, label = "pred", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
    display(plots)

end






#
# xdt=[(mAb_synthetic_dt.sol_SPN_trainingset1[:,:],sol_SPN_gt,"SPN_trainingset1"),
# (mAb_synthetic_dt.sol_SPN_trainingset2[:,:],sol_SPN_gt,"SPN_trainingset2"),
# (mAb_synthetic_dt.sol_SPN_trainingset3[:,:],sol_SPN_gt,"SPN_trainingset3"),
# (mAb_synthetic_dt.sol_SPN_testingset1[:,:],sol_SPN_gt,"SPN_testingset1"),
# (mAb_synthetic_dt.sol_SPN_testingset2[:,:],sol_SPN_gt,"SPN_testingset2"),
# # (mAb_synthetic_dt.sol_SPN_testingset3[:,:],sol_SPN_gt_7min,"SPN_testingset3"),
# # (mAb_synthetic_dt.sol_SPN_testingset4[:,:],sol_SPN_gt_7min,"SPN_testingset4"),
#
# (mAb_synthetic_dt.sol_SP1_trainingset1[:,:],sol_SP1_gt,"SP1_trainingset1"),
# (mAb_synthetic_dt.sol_SP1_trainingset2[:,:],sol_SP1_gt,"SP1_trainingset2"),
# (mAb_synthetic_dt.sol_SP1_trainingset3[:,:],sol_SP1_gt,"SP1_trainingset3"),
# (mAb_synthetic_dt.sol_SP1_testingset1[:,:],sol_SP1_gt,"SP1_testingset1"),
# (mAb_synthetic_dt.sol_SP1_testingset2[:,:],sol_SP1_gt,"SP1_testingset2"),
# # (mAb_synthetic_dt.sol_SP1_testingset3[:,:],sol_SP1_gt_7min,"SP1_testingset3"),
# # (mAb_synthetic_dt.sol_SP1_testingset4[:,:],sol_SP1_gt_7min,"SP1_testingset4"),
#
# (mAb_synthetic_dt.sol_SP2_trainingset1[:,:],sol_SP2_gt,"SP2_trainingset1"),
# (mAb_synthetic_dt.sol_SP2_trainingset2[:,:],sol_SP2_gt,"SP2_trainingset2"),
# (mAb_synthetic_dt.sol_SP2_trainingset3[:,:],sol_SP2_gt,"SP2_trainingset3"),
# (mAb_synthetic_dt.sol_SP2_testingset1[:,:],sol_SP2_gt,"SP2_testingset1"),
# (mAb_synthetic_dt.sol_SP2_testingset2[:,:],sol_SP2_gt,"SP2_testingset2"),
# # (mAb_synthetic_dt.sol_SP2_testingset3[:,:],sol_SP2_gt_7min,"SP2_testingset3"),
# # (mAb_synthetic_dt.sol_SP2_testingset4[:,:],sol_SP2_gt_7min,"SP2_testingset4"),
#
# (mAb_synthetic_dt.sol_SP3_trainingset1[:,:],sol_SP3_gt,"SP3_trainingset1"),
# (mAb_synthetic_dt.sol_SP3_trainingset2[:,:],sol_SP3_gt,"SP3_trainingset2"),
# (mAb_synthetic_dt.sol_SP3_trainingset3[:,:],sol_SP3_gt,"SP3_trainingset3"),
# (mAb_synthetic_dt.sol_SP3_testingset1[:,:],sol_SP3_gt,"SP3_testingset1"),
# (mAb_synthetic_dt.sol_SP3_testingset2[:,:],sol_SP3_gt,"SP3_testingset2"),
# # (mAb_synthetic_dt.sol_SP3_testingset3[:,:],sol_SP3_gt_7min,"SP3_testingset3"),
# # (mAb_synthetic_dt.sol_SP3_testingset4[:,:],sol_SP3_gt_7min,"SP3_testingset4"),
#
# (mAb_synthetic_dt.sol_SP4_trainingset1[:,:],sol_SP4_gt,"SP4_trainingset1"),
# (mAb_synthetic_dt.sol_SP4_trainingset2[:,:],sol_SP4_gt,"SP4_trainingset2"),
# (mAb_synthetic_dt.sol_SP4_trainingset3[:,:],sol_SP4_gt,"SP4_trainingset3"),
# (mAb_synthetic_dt.sol_SP4_testingset1[:,:],sol_SP4_gt,"SP4_testingset1"),
# (mAb_synthetic_dt.sol_SP4_testingset2[:,:],sol_SP4_gt,"SP4_testingset2")
# # (mAb_synthetic_dt.sol_SP4_testingset3[:,:],sol_SP4_gt_7min,"SP4_testingset3"),
# # (mAb_synthetic_dt.sol_SP4_testingset4[:,:],sol_SP4_gt_7min,"SP4_testingset4"),
# ]
#
#
#
#
#
# #OK
#
# #Online measurements case
# #predictions with Ensemble of MLP and correction with CKF.
# for e in xdt[6:6]
#     e = xdt[6:6][1]
#     xtest = e[1]
#     xtest =hcat(xtest[:,1:1],xtest[:,3:end])
#     sol_gt = e[2]
#     sol_gt = hcat(sol_gt[1,:], sol_gt'[:,3:end])
#     str_title = e[3]
#
#     tgrid_opt=Array(0:0.125:103)
#     datasetsize=825-1
#     pred_EnMLP=[]
#
#     Q  = Diagonal([(10e2)^2, 100.01, 100.01, 100.01, 100.01, 100.01]);
#     Q  = Diagonal([(10e2)^2, (10e2)^2, (10e2)^2, (10e2)^2, (10e2)^2, (10e2)^2]);
#     #Q  = Diagonal([(7.894794917982332e7)^2, (5)^2, (1)^2, (10)^2, (10)^2, (50)^2]);
#
#     DT = 0.125;
#     R  = (10e5)^2; #7.894794917982332e7
#
#     # CKF Filter
#     P0 = Diagonal([0.01, 0.01, 0.01, 0.01, 0.01, 0.01 ]);
#     m0 = u0_for_pred = xtest[1,:]#[xtest[1,1:1];xtest[1,3:end]] #[2e8  ; 2e8 ;29.1 ; 4.9 ; 0.0 ; 0.310; 80.6; 7.21e-9]# %[0;0;0;0;0;0;0];
#     u0_for_params=[xtest[1,:];0]
#
#     m = m0
#     P = P0
#
#     # Precompute the UT weights
#     n = size(m, 1)
#     XI = sqrt(n) * [Matrix{Float64}(I, n, n) -Matrix{Float64}(I, n, n)]
#     W = ones(2 * n) / (2 * n)
#
#     # Do the filtering
#     Sk =zeros(1,length(tgrid_opt[2:end]));
#     NIS=zeros(1,length(tgrid_opt[2:end]));
#     KG = zeros(size(m,1),length(tgrid_opt[2:end]));
#     MM = zeros(size(m, 1), length(tgrid_opt[2:end]));
#     PP = zeros(size(P, 1), size(P, 2), length(tgrid_opt[2:end]));
#
#
#     tstart=0.0
#     tend=103.0
#     sampling=0.125
#     tgrid=tstart:sampling:tend
#
#     for k=1:datasetsize
#         p=ensemble_prediction(ensemble_MLP,u0_for_params) #parameters(t) with value ± std
#         pp = Measurement{Float64}[]
#         for j=1:length(p[1])
#             push!(pp,measurement(p[1][j], 1*p[2][j]))#parameters with value ± std
#         end
#
#         tstart=tgrid_opt[k]
#         tend=tgrid_opt[k+1]
#         prob =  ODEProblem(ode_system, u0_for_pred, (tstart,tend),pp)
#         sol = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid) #states (t+1) with uncertainty based on states (t) and parameters(t) with value ± std
#         push!(pred_EnMLP,sol[:,end])
#
#
#         # Form the sigma points for dynamic model
#         SX = repeat(m, 1, 2 * n) + cholesky(Hermitian(P)).L * XI
#         # Propagate through the dynamic model
#         μ_Xv, μ_GLC, μ_GLN, μ_LAC, μ_AMM, μ_mAb = p[1]
#
#         HX =hcat(SX[1,:] + (μ_Xv.*SX[1,:]).*DT,
#                 SX[2,:] + (-μ_GLC.*SX[1,:]).*DT,
#                 SX[3,:] + (-μ_GLN.*SX[1,:]).*DT,
#                 SX[4,:] + (μ_LAC.*SX[1,:]).*DT,
#                 SX[5,:] + (μ_AMM.*SX[1,:]).*DT,
#                 SX[6,:] + (μ_mAb.*SX[1,:]).*DT)'
#
#         # Compute the predicted mean and covariance
#         m = zeros(size(m))
#         P = zeros(size(P))
#         for i in 1:size(HX, 2)
#             m .+= W[i] .* HX[:, i]
#         end
#         for i in 1:size(HX, 2)
#             P .+= W[i] .* (HX[:, i] .- m) * (HX[:, i] .- m)'
#         end
#         P .+= Q
#
#         # Form sigma points for measurement step
#         SX = repeat(m, 1, 2 * n) + cholesky(Hermitian(P)).L * XI
#         HY = SX[1, :]
#
#         # Compute the updated mean and covariance
#         mu = zeros(1,1)
#         S = zeros(1,1)
#         C = zeros(size(SX, 1), 1)
#
#         for i in 1:size(SX, 2)
#             mu .+= W[i] .* HY[i]'
#         end
#
#         for i in 1:size(SX, 2)
#             S = S + W[i] .* (HY[i]'.- mu) * (HY[i]'.- mu)'
#             C .+= W[i] .* (SX[:, i] .- m) * (HY[i]' .- mu)'
#         end
#         S .+= R
#
#         # Compute the gain and updated mean and covariance
#         K = C / S
#         Y=u0_obs=xtest[k+1,1]
#         m .+= K * (Y .- mu)
#         P .-= K * S * K'
#
#         MM[:, k] = m
#         PP[:, :, k] = P
#
#         # if k==1
#         #     u0_for_params=xtest[k,:]
#         #     u0_for_pred=xtest[k,1:6] #state (t)
#         # else
#         u0_for_params=[m;tgrid_opt[k+1]]
#         u0_for_pred=m[:]
#         # end
#     end
#
#     pred_EnMLP=vcat(map(x->x', pred_EnMLP)...)
#
#     #plots
#     lws=2.5
#     gr( xtickfontsize=7, ytickfontsize=7, xguidefontsize=9, yguidefontsize=9, legendfontsize=6);
#     # plot(tgrid_opt[2:end],pred_EnMLP,color=:black ,lw=lws, label = "pred", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
#     plots=Plots.scatter(tgrid_opt,xtest[:,1],title=str_title,color=:indianred1, markerstrokewidth = 0, lw=lws, label = "noise dt", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
#     plot!(Array(0:0.125:103),sol_gt,color=:green , lw=lws, label = "true", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
#     plot!(layout = (3, 2), size = (800, 700))
#     for i = 1:6
#         plot!(plots[i], tgrid_opt[2:end], Measurements.value.(pred_EnMLP[:, i]), ribbon = Measurements.uncertainty.(pred_EnMLP[:, i]), color = :blue, fillalpha=.075, lw = lws, label = "pred", ylabel = "[$(i)]")
#     end
#     plot!(tgrid_opt[2:end], MM', label = "PF",color=:orange  , lw=lws,layout=(3,3))
#     display(plots)
#
#         # plots=Plots.scatter(Array(0:0.125:103),xtest,title=str_title,color=:indianred1, markerstrokewidth = 0, lw=lws, label = "noise dt", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
#         # plot!(Array(0:0.125:103),sol_gt,color=:green , lw=lws, label = "true", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
#         #
# end
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# for e in xdt[6:6]
#     e = xdt[6:6][1]
#     xtest = e[1]
#     xtest =hcat(xtest[:,1:1],xtest[:,3:end])
#     sol_gt = e[2]
#     sol_gt = hcat(sol_gt[1,:], sol_gt'[:,3:end])
#     str_title = e[3]
#
#     tgrid_opt=Array(0:0.125:103)
#     datasetsize=825-1
#     pred_EnMLP=[]
#
#     Q  = Diagonal([(10e2)^2, 100.01, 100.01, 100.01, 100.01, 100.01]);
#     Q  = Diagonal([(10e2)^2, (10e2)^2, (10e2)^2, (10e2)^2, (10e2)^2, (10e2)^2]);
#     Q  = Diagonal([(7.894794917982332e6)^2, (5)^2, (1)^2, (10)^2, (10)^2, (50)^2]);
#
#     DT = 0.125;
#     R  = (7.894794917982332e7)^2; #10e5   7.894794917982332e7
#
#     # CKF Filter
#     P0 = Diagonal([0.01, 0.01, 0.01, 0.01, 0.01, 0.01 ]);
#     m0 = u0_for_pred = xtest[1,:]#[xtest[1,1:1];xtest[1,3:end]] #[2e8  ; 2e8 ;29.1 ; 4.9 ; 0.0 ; 0.310; 80.6; 7.21e-9]# %[0;0;0;0;0;0;0];
#     u0_for_params=[xtest[1,:];0]
#
#     m = m0
#     P = P0
#
#     # Precompute the UT weights
#     n = size(m, 1)
#     XI = sqrt(n) * [Matrix{Float64}(I, n, n) -Matrix{Float64}(I, n, n)]
#     W = ones(2 * n) / (2 * n)
#
#     # Do the filtering
#     Sk =zeros(1,length(tgrid_opt[2:end]));
#     NIS=zeros(1,length(tgrid_opt[2:end]));
#     KG = zeros(size(m,1),length(tgrid_opt[2:end]));
#     MM = zeros(size(m, 1), length(tgrid_opt[2:end]));
#     PP = zeros(size(P, 1), size(P, 2), length(tgrid_opt[2:end]));
#
#
#     tstart=0.0
#     tend=103.0
#     sampling=0.125
#     tgrid=tstart:sampling:tend
#
#     for k=1:datasetsize
#         p=ensemble_prediction(ensemble_MLP,u0_for_params) #parameters(t) with value ± std
#         pp = Measurement{Float64}[]
#         for j=1:length(p[1])
#             push!(pp,measurement(p[1][j], 2*p[2][j]))#parameters with value ± std
#         end
#
#         tstart=tgrid_opt[k]
#         tend=tgrid_opt[k+1]
#         prob =  ODEProblem(ode_system, u0_for_pred, (tstart,tend),pp)
#         sol = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid) #states (t+1) with uncertainty based on states (t) and parameters(t) with value ± std
#         push!(pred_EnMLP,sol[:,end])
#         println(sol[:,end])
#
#         # Form the sigma points for dynamic model
#         SX = repeat(m, 1, 2 * n) + cholesky(Hermitian(P)).L * XI
#         # Propagate through the dynamic model
#         μ_Xv, μ_GLC, μ_GLN, μ_LAC, μ_AMM, μ_mAb = p[1]
#
#         HX =hcat(SX[1,:] + (μ_Xv.*SX[1,:]).*DT,
#                 SX[2,:] + (-μ_GLC.*SX[1,:]).*DT,
#                 SX[3,:] + (-μ_GLN.*SX[1,:]).*DT,
#                 SX[4,:] + (μ_LAC.*SX[1,:]).*DT,
#                 SX[5,:] + (μ_AMM.*SX[1,:]).*DT,
#                 SX[6,:] + (μ_mAb.*SX[1,:]).*DT)'
#
#         # Compute the predicted mean and covariance
#         m = zeros(size(m))
#         P = zeros(size(P))
#         for i in 1:size(HX, 2)
#             m .+= W[i] .* HX[:, i]
#         end
#         for i in 1:size(HX, 2)
#             P .+= W[i] .* (HX[:, i] .- m) * (HX[:, i] .- m)'
#         end
#         P .+= Q
#
#         # Form sigma points for measurement step
#         SX = repeat(m, 1, 2 * n) + cholesky(Hermitian(P)).L * XI
#         HY = SX[1, :]
#
#         # Compute the updated mean and covariance
#         mu = zeros(1,1)
#         S = zeros(1,1)
#         C = zeros(size(SX, 1), 1)
#
#         for i in 1:size(SX, 2)
#             mu .+= W[i] .* HY[i]'
#         end
#
#         for i in 1:size(SX, 2)
#             S = S + W[i] .* (HY[i]'.- mu) * (HY[i]'.- mu)'
#             C .+= W[i] .* (SX[:, i] .- m) * (HY[i]' .- mu)'
#         end
#         S .+= R
#
#         # Compute the gain and updated mean and covariance
#         K = C / S
#         Y=u0_obs=xtest[k+1,1]
#         m .+= K * (Y .- mu)
#         P .-= K * S * K'
#
#         MM[:, k] = m
#         PP[:, :, k] = P
#
#         # if k==1
#         #     u0_for_params=xtest[k,:]
#         #     u0_for_pred=xtest[k,1:6] #state (t)
#         # else
#         u0_for_params=[xtest[k+1,1:6];tgrid_opt[k+1]]
#         u0_for_pred=xtest[k+1,1:6]
#         # end
#     end
#
#     pred_EnMLP=vcat(map(x->x', pred_EnMLP)...)
#
#     #plots
#     lws=2.5
#     gr( xtickfontsize=7, ytickfontsize=7, xguidefontsize=9, yguidefontsize=9, legendfontsize=6);
#     # plot(tgrid_opt[2:end],pred_EnMLP,color=:black ,lw=lws, label = "pred", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
#     plots=Plots.scatter(tgrid_opt,xtest[:,1],title=str_title,color=:indianred1, markerstrokewidth = 0, lw=lws, label = "noise dt", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
#     plot!(Array(0:0.125:103),sol_gt,color=:green , lw=lws, label = "true", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
#     # plot!(layout = (3, 2), size = (800, 700))
#     for i = 1:6
#         plot!(plots[i], tgrid_opt[2:end], Measurements.value.(pred_EnMLP[:, i]), ribbon = Measurements.uncertainty.(pred_EnMLP[:, i]), color = :blue, fillalpha=.075, lw = lws, label = "pred", ylabel = "[$(i)]")
#     end
#     ## plot!(tgrid_opt[2:end], pred_EnMLP, color = :blue, lw = lws, label = "pred", )
#
#     # plot!(tgrid_opt[2:end], MM', label = "PF",color=:orange  , lw=lws,layout=(3,2))
#     display(plots)
#
#         # plots=Plots.scatter(Array(0:0.125:103),xtest,title=str_title,color=:indianred1, markerstrokewidth = 0, lw=lws, label = "noise dt", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
#         # plot!(Array(0:0.125:103),sol_gt,color=:green , lw=lws, label = "true", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
#         #
# end
#
#
#
#
#
#
#
#
#
#
# #




#OK

#Online measurements case
#predictions with Ensemble of MLP and correction with CKF.
for e in xdt[6:6]
    e = xdt[1:1][1]
    xtest = e[1]
    xtest =hcat(xtest[:,1:1],xtest[:,3:end])
    sol_gt = e[2]
    sol_gt = hcat(sol_gt[1,:], sol_gt'[:,3:end])
    str_title = e[3]

    tgrid_opt=Array(0:0.125:103)
    steps=length(tgrid_opt)-1

    pred_EnMLP=[]
    pred_EnMLP2=[]

    # Filter parameters
    Q  = Diagonal([(1e3)^2, 100.01, 100.01, 100.01, 100.01, 100.01]);
    # Q  = Diagonal([(10e2)^2, (10e2)^2, (10e2)^2, (10e2)^2, (10e2)^2, (10e2)^2]);
    #Q  = Diagonal([(7.894794917982332e7)^2, (5)^2, (1)^2, (10)^2, (10)^2, (50)^2]);

    DT = 0.125;
    R  = (1e7)^2; #7.894794917982332e7

    # CKF Filter
    P0 = Diagonal([0.01, 0.01, 0.01, 0.01, 0.01, 0.01 ]);
#     P0 = Diagonal((xtest[1,1:6]-sol_gt[1,:]).^2);
#     P0 = Diagonal(   [ 2.6634269348194384e17
#     0.36325764954020895
#     0.0007065618588301385
#     8.685451115398196
# 1000.010107046247258309
#  1.4700933187464]);
    m0 = u0_for_pred = xtest[1,:]#[xtest[1,1:1];xtest[1,3:end]] #[2e8  ; 2e8 ;29.1 ; 4.9 ; 0.0 ; 0.310; 80.6; 7.21e-9]# %[0;0;0;0;0;0;0];
    u0_for_params=[xtest[1,:];0]
    u0_for_pred2 = xtest[1,:] #[2e8  ; 2e8 ;29.1 ; 4.9 ; 0.0 ; 0.310; 80.6; 7.21e-9]# %[0;0;0;0;0;0;0];
    u0_for_params2=[xtest[1,:];0]

    m = m0
    P = P0

    # Precompute the UT weights
    n = size(m, 1)
    XI = sqrt(n) * [Matrix{Float64}(I, n, n) -Matrix{Float64}(I, n, n)]
    W = ones(2 * n) / (2 * n)

    # Saving the filtering corrections
    MM = zeros(size(m, 1), length(tgrid_opt[2:end]));
    PP = zeros(size(P, 1), size(P, 2), length(tgrid_opt[2:end]));

    tstart=0.0
    tend=103.0
    sampling=0.125
    tgrid=tstart:sampling:tend

    pp=[]
    for k=1:steps

        #estimate the best parameters(t) for the dynamic system based on states(t) updated by CKF using Xv(t) observed.
        p=ensemble_prediction(ensemble_MLP,u0_for_params2) #parameters(t) with value ± std
        pp = Measurement{Float64}[]
        for j=1:length(p[1])
            push!(pp,measurement(p[1][j], 2*p[2][j]))#parameters with value ± std
        end
        # estimate the states (t+1) with uncertainty based on states (t) and parameters(t) with value ± std
        tstart=tgrid_opt[k]
        tend=tgrid_opt[k+1]
        prob =  ODEProblem(ode_system, u0_for_pred2, (tstart,tend),pp)
        sol = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid) #states (t+1) with uncertainty based on states (t) and parameters(t) with value ± std
        push!(pred_EnMLP2,sol[:,end])
        u0_for_pred2 = [xtest[k+1,1:1];Measurements.value.(sol[2:6,end])] #[2e8  ; 2e8 ;29.1 ; 4.9 ; 0.0 ; 0.310; 80.6; 7.21e-9]# %[0;0;0;0;0;0;0];
        u0_for_params2=[xtest[k+1,1:1];Measurements.value.(sol[2:6,end]);tgrid_opt[k+1]]


        #estimate the best parameters(t) for the dynamic system based on states(t) updated by CKF using Xv(t) observed.
        # if tgrid_opt[k] in [0.0, 7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, 56.0, 63.0, 70.0, 77.0, 84.0, 91.0, 98.0]
            p=ensemble_prediction(ensemble_MLP,u0_for_params) #parameters(t) with value ± std
            pp = Measurement{Float64}[]
            for j=1:length(p[1])
                push!(pp,measurement(p[1][j], 2*p[2][j]))#parameters with value ± std
            end
            println(tgrid_opt[k])
        # end

        # estimate the states (t+1) with uncertainty based on states (t) and parameters(t) with value ± std
        tstart=tgrid_opt[k]
        tend=tgrid_opt[k+1]
        prob =  ODEProblem(ode_system, u0_for_pred, (tstart,tend),pp)
        sol = solve(prob, AutoTsit5(Rosenbrock23()),saveat=tgrid) #states (t+1) with uncertainty based on states (t) and parameters(t) with value ± std
        push!(pred_EnMLP,sol[:,end])


        # Form the sigma points for dynamic model
        SX = repeat(m, 1, 2 * n) + cholesky(Hermitian(P)).L * XI
        # Propagate through the dynamic model
        μ_Xv, μ_GLC, μ_GLN, μ_LAC, μ_AMM, μ_mAb = p[1]

        HX =hcat(SX[1,:] + (μ_Xv.*SX[1,:]).*DT,
                SX[2,:] + (-μ_GLC.*SX[1,:]).*DT,
                SX[3,:] + (-μ_GLN.*SX[1,:]).*DT,
                SX[4,:] + (μ_LAC.*SX[1,:]).*DT,
                SX[5,:] + (μ_AMM.*SX[1,:]).*DT,
                SX[6,:] + (μ_mAb.*SX[1,:]).*DT)'

        # Compute the predicted mean and covariance
        m = zeros(size(m))
        P = zeros(size(P))
        for i in 1:size(HX, 2)
            m .+= W[i] .* HX[:, i]
        end
        for i in 1:size(HX, 2)
            P .+= W[i] .* (HX[:, i] .- m) * (HX[:, i] .- m)'
        end
        P .+= Q

        # Form sigma points for measurement step
        SX = repeat(m, 1, 2 * n) + cholesky(Hermitian(P)).L * XI
        HY = SX[1, :]

        # Compute the updated mean and covariance
        mu = zeros(1,1)
        S = zeros(1,1)
        C = zeros(size(SX, 1), 1)

        for i in 1:size(SX, 2)
            mu .+= W[i] .* HY[i]'
        end

        for i in 1:size(SX, 2)
            S = S + W[i] .* (HY[i]'.- mu) * (HY[i]'.- mu)'
            C .+= W[i] .* (SX[:, i] .- m) * (HY[i]' .- mu)'
        end
        S .+= R

        # Compute the gain and updated mean and covariance
        K = C / S
        Y=u0_obs=xtest[k+1,1]
        m .+= K * (Y .- mu)
        P .-= K * S * K'

        MM[:, k] = m
        PP[:, :, k] = P

        # if k==1
        #     u0_for_params=xtest[k,:]
        #     u0_for_pred=xtest[k,1:6] #state (t)
        # else
        u0_for_params=[m;tgrid_opt[k+1]]
        u0_for_pred=m
        # end
    end

    pred_EnMLP=vcat(map(x->x', pred_EnMLP)...)
    pred_EnMLP2=vcat(map(x->x', pred_EnMLP2)...)

    #plots
    lws=2.5
    gr( xtickfontsize=7, ytickfontsize=7, xguidefontsize=9, yguidefontsize=9, legendfontsize=6);
    # plot(tgrid_opt[2:end],pred_EnMLP,color=:black ,lw=lws, label = "pred", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
    plots=Plots.scatter(tgrid_opt,xtest[:,1],title=str_title,color=:indianred1, markerstrokewidth = 0, lw=lws, label = "noise dt", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
    plot!(Array(0:0.125:103),sol_gt,color=:green , lw=lws, label = "true", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
    plot!(layout = (3, 2), size = (800, 700))
    # for i = 1:6
    #     plot!(plots[i], tgrid_opt[2:end], Measurements.value.(pred_EnMLP[:, i]), ribbon = Measurements.uncertainty.(pred_EnMLP[:, i]), color = :blue, fillalpha=.5, lw = lws, label = "ENN", ylabel = "[$(i)]")
    # end
    plot!(layout = (3, 2), size = (800, 700))
    for i = 1:6
        plot!(plots[i], tgrid_opt[2:end], Measurements.value.(pred_EnMLP2[:, i]), ribbon = Measurements.uncertainty.(pred_EnMLP2[:, i]), color = :purple, fillalpha=.075, lw = lws, label = "ENN")
    end
    # plot!(tgrid_opt[1:400],pred_EnMLP[1:400,:],color=:red ,lw=lws, label = "pred", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
    # plot!(tgrid_opt[2:end],pred_EnMLP[:,:],color=:red ,lw=lws, label = "ENN-pred", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))

    plot!(tgrid_opt[2:end], MM', label = "ENN+CKF",color=:orange  , lw=lws,layout=(3,3))
    display(plots)

        # plots=Plots.scatter(Array(0:0.125:103),xtest,title=str_title,color=:indianred1, markerstrokewidth = 0, lw=lws, label = "noise dt", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
        # plot!(Array(0:0.125:103),sol_gt,color=:green , lw=lws, label = "true", ylabel=["[Xv]" "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800,700))
        #
end
