using  Plots,Distributions,DifferentialEquations
using BSON#: @load
using Statistics
using DataFrames
using Flux
#using Measurements,
using StaticArrays
using LinearAlgebra,Measures

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







xdt=[
(mAb_synthetic_dt.sol_SPN_testingset3[:,:],sol_SPN_gt_7min,"SP0 B.1","SPN_testingset3HNKE-C.csv","SPN_testingset3CKF.csv","SPN_testingset3HNKE-U.csv","SPN_testingset3UKF.csv","SPN_testingset3JUKF-SANTO.csv","SPN_testingset3JCKF-SANTO.csv"),
(mAb_synthetic_dt.sol_SPN_testingset4[:,:],sol_SPN_gt_7min,"SP0 B.2","SPN_testingset4HNKE-C.csv","SPN_testingset4CKF.csv","SPN_testingset4HNKE-U.csv","SPN_testingset4UKF.csv","SPN_testingset4JUKF-SANTO.csv","SPN_testingset4JCKF-SANTO.csv"),

(mAb_synthetic_dt.sol_SP1_testingset3[:,:],sol_SP1_gt_7min,"SP1 B.1","SP1_testingset3HNKE-C.csv","SP1_testingset3CKF.csv","SP1_testingset3HNKE-U.csv","SP1_testingset3UKF.csv","SP1_testingset3JUKF-SANTO.csv","SP1_testingset3JCKF-SANTO.csv"),
(mAb_synthetic_dt.sol_SP1_testingset4[:,:],sol_SP1_gt_7min,"SP1 B.2","SP1_testingset4HNKE-C.csv","SP1_testingset4CKF.csv","SP1_testingset4HNKE-U.csv","SP1_testingset4UKF.csv","SP1_testingset4JUKF-SANTO.csv","SP1_testingset4JCKF-SANTO.csv"),

(mAb_synthetic_dt.sol_SP2_testingset3[:,:],sol_SP2_gt_7min,"SP2 B.1","SP2_testingset3HNKE-C.csv","SP2_testingset3CKF.csv","SP2_testingset3HNKE-U.csv","SP2_testingset3UKF.csv","SP2_testingset3JUKF-SANTO.csv","SP2_testingset3JCKF-SANTO.csv"),
(mAb_synthetic_dt.sol_SP2_testingset4[:,:],sol_SP2_gt_7min,"SP2 B.2","SP2_testingset4HNKE-C.csv","SP2_testingset4CKF.csv","SP2_testingset4HNKE-U.csv","SP2_testingset4UKF.csv","SP2_testingset4JUKF-SANTO.csv","SP2_testingset4JCKF-SANTO.csv"),

(mAb_synthetic_dt.sol_SP3_testingset3[:,:],sol_SP3_gt_7min,"SP3 B.1","SP3_testingset3HNKE-C.csv","SP3_testingset3CKF.csv","SP3_testingset3HNKE-U.csv","SP3_testingset3UKF.csv","SP3_testingset3JUKF-SANTO.csv","SP3_testingset3JCKF-SANTO.csv"),
(mAb_synthetic_dt.sol_SP3_testingset4[:,:],sol_SP3_gt_7min,"SP3 B.2","SP3_testingset4HNKE-C.csv","SP3_testingset4CKF.csv","SP3_testingset4HNKE-U.csv","SP3_testingset4UKF.csv","SP3_testingset4JUKF-SANTO.csv","SP3_testingset4JCKF-SANTO.csv"),

(mAb_synthetic_dt.sol_SP4_testingset3[:,:],sol_SP4_gt_7min,"SP4 B.1","SP4_testingset3HNKE-C.csv","SP4_testingset3CKF.csv","SP4_testingset3HNKE-U.csv","SP4_testingset3UKF.csv","SP4_testingset3JUKF-SANTO.csv","SP4_testingset3JCKF-SANTO.csv"),
(mAb_synthetic_dt.sol_SP4_testingset4[:,:],sol_SP4_gt_7min,"SP4 B.2","SP4_testingset4HNKE-C.csv","SP4_testingset4CKF.csv","SP4_testingset4HNKE-U.csv","SP4_testingset4UKF.csv","SP4_testingset4JUKF-SANTO.csv","SP4_testingset4JCKF-SANTO.csv")

]







using CSV, DataFrames
using LaTeXStrings



for e in xdt
    # e = xdt[10:10][1]
    xtest = e[1]
    xtest =hcat(xtest[:,1:1],xtest[:,3:end])
    sol_gt = e[2]
    sol_gt = hcat(sol_gt[1,:], sol_gt'[:,3:end])
    str_title = e[3]

    path="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/HNKE-C/"
    HNKEc = Array(CSV.read(path*"$(e[4])", DataFrame))
    CKF = Array(CSV.read(path*"$(e[5])", DataFrame))
    path="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/HNKE-U/"
    HNKEu = Array(CSV.read(path*"$(e[6])", DataFrame))
    UKF = Array(CSV.read(path*"$(e[7])", DataFrame))
    path="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/"
    UKFsanto = Array(CSV.read(path*"$(e[8])", DataFrame))
    CKFsanto = Array(CSV.read(path*"$(e[9])", DataFrame))

    tgrid_opt=Array(0:0.125:103)
    steps=length(tgrid_opt)-1

    lws=6
    gr( xtickfontsize=10, ytickfontsize=10, xguidefontsize=10, yguidefontsize=10, legendfontsize=6);

    plots=Plots.scatter(tgrid_opt,xtest[:,1],title=str_title,color=:grey, markerstrokewidth = 0, lw=lws, label = "Noisy Xv", ylabel=["Xv(Cell/L)" "GLC(mM)" "GLN(mM)" "LAC(mM)" "AMM(mM)" "mAb(mg/L)"], layout=(3,2),size = (800,700))
    plot!(Array(0:0.125:103), sol_gt, color=:red, lw=lws, label = "True", layout=(3,2),size = (800,700))
    plot!(tgrid_opt[2:end], CKFsanto[:,1:6], label = "JCKF", grid=false,linestyle=:solid,color=:purple1, lw=lws,layout=(3,2))
    plot!(tgrid_opt[2:end], UKFsanto[:,1:6], label = "JUKF", grid=false,linestyle=:solid,color=:purple4, lw=lws,layout=(3,2))
    # plot!(tgrid_opt[2:end], CKF, label = "CKF", grid=false,linestyle=:dashdot,color=:green3, lw=lws,layout=(3,2))
    # plot!(tgrid_opt[2:end], UKF, label = "UKF", grid=false,linestyle=:dash,color=:green, lw=lws,layout=(3,2))
    plot!(tgrid_opt[2:end], HNKEc, label = "HNKE-C", grid=false,linestyle=:dashdotdot,color=:blue, lw=lws,layout=(3,2))
    plot!(tgrid_opt[2:end], HNKEu, label = "HNKE-U", grid=false,linestyle=:dot,color=:lightblue, lw=lws,layout=(3,2),xlabel="Hours")

    # plot!(Array(0:0.125:103), sol_gt, color=:red, lw=lws, label = "true", layout=(3,2),size = (800,700))
    # plot!(tgrid_opt[2:end], CKFsanto[:,1:6], label = "JCKF", grid=false,linestyle=:solid,color=:purple1, lw=lws,layout=(3,2))
    # plot!(tgrid_opt[2:end], UKFsanto[:,1:6], label = "JUKF", grid=false,linestyle=:solid,color=:purple4, lw=lws,layout=(3,2))
    # plot!(tgrid_opt[2:end], CKF, label = "CKF", grid=false,linestyle=:dashdot,color=:blue, lw=lws,layout=(3,2))
    # plot!(tgrid_opt[2:end], HNKEc, label = "HNKE-C", grid=false,linestyle=:dashdotdot,color=:orange, lw=lws,layout=(3,2))
    # plot!(tgrid_opt[2:end], UKF, label = "UKF", grid=false,linestyle=:dash,color=:green, lw=lws,layout=(3,2))
    # plot!(tgrid_opt[2:end], HNKEu, label = "HNKE-U", grid=false,linestyle=:dot,color=:lightblue, lw=lws,layout=(3,2),xlabel="Hours")


    display(plots)
    sss=replace(str_title, " " => "_")
    savefig(sss*".png")

    # # println("\multirow{6}{*}{",str_title,"} & & & & & NNline")
    # println(str_title)
    # println("&Xv & ",round(RMSPE(sol_gt[2:end,1],CKF[:,1]), digits=3),  " & ",round(RMSPE(sol_gt[2:end,1],HNKEc[:,1]), digits=3), " & ",round(RMSPE(sol_gt[2:end,1],UKF[:,1]), digits=3), " & ",round(RMSPE(sol_gt[2:end,1],HNKEu[:,1]), digits=3)," & ",round(RMSPE(sol_gt[2:end,1],CKFsanto[:,1]), digits=3), " & ",round(RMSPE(sol_gt[2:end,1],UKFsanto[:,1]), digits=3), "NNline")
    # println("&Glc & ",round(RMSPE(sol_gt[2:end,2],CKF[:,2]), digits=3), " & ",round(RMSPE(sol_gt[2:end,2],HNKEc[:,2]), digits=3), " & ",round(RMSPE(sol_gt[2:end,2],UKF[:,2]), digits=3), " & ",round(RMSPE(sol_gt[2:end,2],HNKEu[:,2]), digits=3)," & ",round(RMSPE(sol_gt[2:end,2],CKFsanto[:,2]), digits=3), " & ",round(RMSPE(sol_gt[2:end,2],UKFsanto[:,2]), digits=3), "NNline")
    # println("&Gln & ",round(RMSPE(sol_gt[2:end,3].+10.0,CKF[:,3].+10.0), digits=3), " & ",round(RMSPE(sol_gt[2:end,3].+10.0,HNKEc[:,3].+10.0), digits=3), " & ",round(RMSPE(sol_gt[2:end,3].+10.0,UKF[:,3].+10.0), digits=3), " & ",round(RMSPE(sol_gt[2:end,3].+10.0,HNKEu[:,3].+10.0), digits=3), " & ",round(RMSPE(sol_gt[2:end,3].+10.0,CKFsanto[:,3].+10.0), digits=3), " & ",round(RMSPE(sol_gt[2:end,3].+10.0,UKFsanto[:,3].+10.0), digits=3),"NNline")
    # println("&Lac & ",round(RMSPE(sol_gt[2:end,4],CKF[:,4]), digits=3), " & ",round(RMSPE(sol_gt[2:end,4],HNKEc[:,4]), digits=3), " & ",round(RMSPE(sol_gt[2:end,4],UKF[:,4]), digits=3), " & ",round(RMSPE(sol_gt[2:end,4],HNKEu[:,4]), digits=3)," & ",round(RMSPE(sol_gt[2:end,4],CKFsanto[:,4]), digits=3), " & ",round(RMSPE(sol_gt[2:end,4],UKFsanto[:,4]), digits=3), "NNline")
    # println("&Amm & ",round(RMSPE(sol_gt[2:end,5],CKF[:,5]), digits=3), " & ",round(RMSPE(sol_gt[2:end,5],HNKEc[:,5]), digits=3), " & ",round(RMSPE(sol_gt[2:end,5],UKF[:,5]), digits=3), " & ",round(RMSPE(sol_gt[2:end,5],HNKEu[:,5]), digits=3), " & ",round(RMSPE(sol_gt[2:end,5],CKFsanto[:,5]), digits=3), " & ",round(RMSPE(sol_gt[2:end,5],UKFsanto[:,5]), digits=3),"NNline")
    # println("&mAb & ",round(RMSPE(sol_gt[2:end,6],CKF[:,6]), digits=3), " & ",round(RMSPE(sol_gt[2:end,6],HNKEc[:,6]), digits=3), " & ",round(RMSPE(sol_gt[2:end,6],UKF[:,6]), digits=3), " & ",round(RMSPE(sol_gt[2:end,6],HNKEu[:,6]), digits=3), " & ",round(RMSPE(sol_gt[2:end,6],CKFsanto[:,6]), digits=3), " & ",round(RMSPE(sol_gt[2:end,6],UKFsanto[:,6]), digits=3),"NNline")


    println(str_title)
    println("&Xv & ",round(RMSPE(sol_gt[2:end,1],HNKEc[:,1]), digits=3)             , " & ",round(RMSPE(sol_gt[2:end,1],HNKEu[:,1]), digits=3)," & ",round(RMSPE(sol_gt[2:end,1],CKFsanto[:,1]), digits=3), " & ",round(RMSPE(sol_gt[2:end,1],UKFsanto[:,1]), digits=3), "NNline")
    println("&Glc & ",round(RMSPE(sol_gt[2:end,2].+10.0,HNKEc[:,2].+10.0), digits=3)            , " & ",round(RMSPE(sol_gt[2:end,2].+10.0,HNKEu[:,2].+10.0), digits=3)," & ",round(RMSPE(sol_gt[2:end,2].+10.0,CKFsanto[:,2].+10.0), digits=3), " & ",round(RMSPE(sol_gt[2:end,2].+10.0,UKFsanto[:,2].+10.0), digits=3), "NNline")
    println("&Gln & ",round(RMSPE(sol_gt[2:end,3].+10.0,HNKEc[:,3].+10.0), digits=3)  , " & ",round(RMSPE(sol_gt[2:end,3].+10.0,HNKEu[:,3].+10.0), digits=3), " & ",round(RMSPE(sol_gt[2:end,3].+10.0,CKFsanto[:,3].+10.0), digits=3), " & ",round(RMSPE(sol_gt[2:end,3].+10.0,UKFsanto[:,3].+10.0), digits=3),"NNline")
    println("&Lac & ",round(RMSPE(sol_gt[2:end,4].+10.0,HNKEc[:,4].+10.0), digits=3)            , " & ",round(RMSPE(sol_gt[2:end,4].+10.0,HNKEu[:,4].+10.0), digits=3)," & ",round(RMSPE(sol_gt[2:end,4].+10.0,CKFsanto[:,4].+10.0), digits=3), " & ",round(RMSPE(sol_gt[2:end,4].+10.0,UKFsanto[:,4].+10.0), digits=3), "NNline")
    println("&Amm & ",round(RMSPE(sol_gt[2:end,5],HNKEc[:,5]), digits=3)            , " & ",round(RMSPE(sol_gt[2:end,5],HNKEu[:,5]), digits=3), " & ",round(RMSPE(sol_gt[2:end,5],CKFsanto[:,5]), digits=3), " & ",round(RMSPE(sol_gt[2:end,5],UKFsanto[:,5]), digits=3),"NNline")
    println("&mAb & ",round(RMSPE(sol_gt[2:end,6],HNKEc[:,6]), digits=3)            , " & ",round(RMSPE(sol_gt[2:end,6],HNKEu[:,6]), digits=3), " & ",round(RMSPE(sol_gt[2:end,6],CKFsanto[:,6]), digits=3), " & ",round(RMSPE(sol_gt[2:end,6],UKFsanto[:,6]), digits=3),"NNline")
    println(str_title,  " heatmap")
    println("Xv  ",round(RMSPE(sol_gt[2:end,1],HNKEc[:,1]), digits=3)             , "  ",round(RMSPE(sol_gt[2:end,1],HNKEu[:,1]), digits=3),"  ",round(RMSPE(sol_gt[2:end,1],CKFsanto[:,1]), digits=3), "  ",round(RMSPE(sol_gt[2:end,1],UKFsanto[:,1]), digits=3), "")
    println("Glc  ",round(RMSPE(sol_gt[2:end,2].+10.0,HNKEc[:,2].+10.0), digits=3)            , "  ",round(RMSPE(sol_gt[2:end,2].+10.0,HNKEu[:,2].+10.0), digits=3),"  ",round(RMSPE(sol_gt[2:end,2].+10.0,CKFsanto[:,2].+10.0), digits=3), "  ",round(RMSPE(sol_gt[2:end,2].+10.0,UKFsanto[:,2].+10.0), digits=3), "")
    println("Gln  ",round(RMSPE(sol_gt[2:end,3].+10.0,HNKEc[:,3].+10.0), digits=3)  , "  ",round(RMSPE(sol_gt[2:end,3].+10.0,HNKEu[:,3].+10.0), digits=3), "  ",round(RMSPE(sol_gt[2:end,3].+10.0,CKFsanto[:,3].+10.0), digits=3), "  ",round(RMSPE(sol_gt[2:end,3].+10.0,UKFsanto[:,3].+10.0), digits=3),"")
    println("Lac  ",round(RMSPE(sol_gt[2:end,4].+10.0,HNKEc[:,4].+10.0), digits=3)            , "  ",round(RMSPE(sol_gt[2:end,4].+10.0,HNKEu[:,4].+10.0), digits=3),"  ",round(RMSPE(sol_gt[2:end,4].+10.0,CKFsanto[:,4].+10.0), digits=3), "  ",round(RMSPE(sol_gt[2:end,4].+10.0,UKFsanto[:,4].+10.0), digits=3), "")
    println("Amm  ",round(RMSPE(sol_gt[2:end,5],HNKEc[:,5]), digits=3)            , "  ",round(RMSPE(sol_gt[2:end,5],HNKEu[:,5]), digits=3), "  ",round(RMSPE(sol_gt[2:end,5],CKFsanto[:,5]), digits=3), "  ",round(RMSPE(sol_gt[2:end,5],UKFsanto[:,5]), digits=3),"")
    println("mAb  ",round(RMSPE(sol_gt[2:end,6],HNKEc[:,6]), digits=3)            , "  ",round(RMSPE(sol_gt[2:end,6],HNKEu[:,6]), digits=3), "  ",round(RMSPE(sol_gt[2:end,6],CKFsanto[:,6]), digits=3), " ",round(RMSPE(sol_gt[2:end,6],UKFsanto[:,6]), digits=3),"")


end




##### Answering the Research Question 2


pltres=Dict()
i1=1
for e in xdt #[10:10]
    # e = xdt[10:10][1]
    xtest = e[1]
    xtest =hcat(xtest[:,1:1],xtest[:,3:end])
    sol_gt = e[2]
    sol_gt = hcat(sol_gt[1,:], sol_gt'[:,3:end])
    str_title = e[3]

    path="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/HNKE-C/"
    HNKEc = Array(CSV.read(path*"$(e[4])", DataFrame))
    CKF = Array(CSV.read(path*"$(e[5])", DataFrame))
    path="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/HNKE-U/"
    HNKEu = Array(CSV.read(path*"$(e[6])", DataFrame))
    UKF = Array(CSV.read(path*"$(e[7])", DataFrame))
    path="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/"
    UKFsanto = Array(CSV.read(path*"$(e[8])", DataFrame))
    CKFsanto = Array(CSV.read(path*"$(e[9])", DataFrame))

    tgrid_opt=Array(0:0.125:103)
    steps=length(tgrid_opt)-1

    lws=4
    gr( xtickfontsize=10, ytickfontsize=10, xguidefontsize=10, yguidefontsize=10, legendfontsize=6);

    plots=Plots.scatter(tgrid_opt,xtest[:,1],title=str_title,color=:grey, markerstrokewidth = 0, lw=lws, label = "Noisy Xv", ylabel=["Xv(Cell/L)" "GLC(mM)" "GLN(mM)" "LAC(mM)" "AMM(mM)" "mAb(mg/L)"], layout=(6,1),size = (200,800))
    plot!(Array(0:0.125:103), sol_gt, color=:red, lw=lws, label = "True", layout=(3,2))
    plot!(tgrid_opt[2:end], CKFsanto[:,1:6], label = "JCKF", grid=false,linestyle=:solid,color=:mediumorchid1, lw=lws,layout=(6,1))
    plot!(tgrid_opt[2:end], UKFsanto[:,1:6], label = "JUKF", grid=false,linestyle=:solid,color=:purple4, lw=lws,layout=(6,1))
    # plot!(tgrid_opt[2:end], CKF, label = "CKF", grid=false,linestyle=:dashdot,color=:green3, lw=lws,layout=(3,2))
    # plot!(tgrid_opt[2:end], UKF, label = "UKF", grid=false,linestyle=:dash,color=:green, lw=lws,layout=(3,2))
    plot!(tgrid_opt[2:end], HNKEc, label = "HNKE-C", grid=false,linestyle=:dashdotdot,color=:blue, lw=lws,layout=(6,1))
    plot!(tgrid_opt[2:end], HNKEu, label = "HNKE-U", grid=false,linestyle=:dot,color=:lightblue, lw=lws,layout=(6,1),xlabel="Hours")

    # plot!(Array(0:0.125:103), sol_gt, color=:red, lw=lws, label = "true", layout=(3,2),size = (800,700))
    # plot!(tgrid_opt[2:end], CKFsanto[:,1:6], label = "JCKF", grid=false,linestyle=:solid,color=:purple1, lw=lws,layout=(3,2))
    # plot!(tgrid_opt[2:end], UKFsanto[:,1:6], label = "JUKF", grid=false,linestyle=:solid,color=:purple4, lw=lws,layout=(3,2))
    # plot!(tgrid_opt[2:end], CKF, label = "CKF", grid=false,linestyle=:dashdot,color=:blue, lw=lws,layout=(3,2))
    # plot!(tgrid_opt[2:end], HNKEc, label = "HNKE-C", grid=false,linestyle=:dashdotdot,color=:orange, lw=lws,layout=(3,2))
    # plot!(tgrid_opt[2:end], UKF, label = "UKF", grid=false,linestyle=:dash,color=:green, lw=lws,layout=(3,2))
    # plot!(tgrid_opt[2:end], HNKEu, label = "HNKE-U", grid=false,linestyle=:dot,color=:lightblue, lw=lws,layout=(3,2),xlabel="Hours")

    pltres[i1]=plots
    i1=i1+1

    display(plots)
    sss=replace(str_title, " " => "_")
    savefig(sss*".png")

    # println("\multirow{6}{*}{",str_title,"} & & & & & NNline")
    # println(str_title)
    # println("&Xv & ",round(RMSPE(sol_gt[2:end,1],CKF[:,1]), digits=3),  " & ",round(RMSPE(sol_gt[2:end,1],HNKEc[:,1]), digits=3), " & ",round(RMSPE(sol_gt[2:end,1],UKF[:,1]), digits=3), " & ",round(RMSPE(sol_gt[2:end,1],HNKEu[:,1]), digits=3)," & ",round(RMSPE(sol_gt[2:end,1],CKFsanto[:,1]), digits=3), " & ",round(RMSPE(sol_gt[2:end,1],UKFsanto[:,1]), digits=3), "NNline")
    # println("&Glc & ",round(RMSPE(sol_gt[2:end,2],CKF[:,2]), digits=3), " & ",round(RMSPE(sol_gt[2:end,2],HNKEc[:,2]), digits=3), " & ",round(RMSPE(sol_gt[2:end,2],UKF[:,2]), digits=3), " & ",round(RMSPE(sol_gt[2:end,2],HNKEu[:,2]), digits=3)," & ",round(RMSPE(sol_gt[2:end,2],CKFsanto[:,2]), digits=3), " & ",round(RMSPE(sol_gt[2:end,2],UKFsanto[:,2]), digits=3), "NNline")
    # println("&Gln & ",round(RMSPE(sol_gt[2:end,3].+10.0,CKF[:,3].+10.0), digits=3), " & ",round(RMSPE(sol_gt[2:end,3].+10.0,HNKEc[:,3].+10.0), digits=3), " & ",round(RMSPE(sol_gt[2:end,3].+10.0,UKF[:,3].+10.0), digits=3), " & ",round(RMSPE(sol_gt[2:end,3].+10.0,HNKEu[:,3].+10.0), digits=3), " & ",round(RMSPE(sol_gt[2:end,3].+10.0,CKFsanto[:,3].+10.0), digits=3), " & ",round(RMSPE(sol_gt[2:end,3].+10.0,UKFsanto[:,3].+10.0), digits=3),"NNline")
    # println("&Lac & ",round(RMSPE(sol_gt[2:end,4],CKF[:,4]), digits=3), " & ",round(RMSPE(sol_gt[2:end,4],HNKEc[:,4]), digits=3), " & ",round(RMSPE(sol_gt[2:end,4],UKF[:,4]), digits=3), " & ",round(RMSPE(sol_gt[2:end,4],HNKEu[:,4]), digits=3)," & ",round(RMSPE(sol_gt[2:end,4],CKFsanto[:,4]), digits=3), " & ",round(RMSPE(sol_gt[2:end,4],UKFsanto[:,4]), digits=3), "NNline")
    # println("&Amm & ",round(RMSPE(sol_gt[2:end,5],CKF[:,5]), digits=3), " & ",round(RMSPE(sol_gt[2:end,5],HNKEc[:,5]), digits=3), " & ",round(RMSPE(sol_gt[2:end,5],UKF[:,5]), digits=3), " & ",round(RMSPE(sol_gt[2:end,5],HNKEu[:,5]), digits=3), " & ",round(RMSPE(sol_gt[2:end,5],CKFsanto[:,5]), digits=3), " & ",round(RMSPE(sol_gt[2:end,5],UKFsanto[:,5]), digits=3),"NNline")
    # println("&mAb & ",round(RMSPE(sol_gt[2:end,6],CKF[:,6]), digits=3), " & ",round(RMSPE(sol_gt[2:end,6],HNKEc[:,6]), digits=3), " & ",round(RMSPE(sol_gt[2:end,6],UKF[:,6]), digits=3), " & ",round(RMSPE(sol_gt[2:end,6],HNKEu[:,6]), digits=3), " & ",round(RMSPE(sol_gt[2:end,6],CKFsanto[:,6]), digits=3), " & ",round(RMSPE(sol_gt[2:end,6],UKFsanto[:,6]), digits=3),"NNline")
    println(str_title)
    println("&Xv & ",round(RMSPE(sol_gt[2:end,1],HNKEc[:,1]), digits=3)             , " & ",round(RMSPE(sol_gt[2:end,1],HNKEu[:,1]), digits=3)," & ",round(RMSPE(sol_gt[2:end,1],CKFsanto[:,1]), digits=3), " & ",round(RMSPE(sol_gt[2:end,1],UKFsanto[:,1]), digits=3), "NNline")
    println("&Glc & ",round(RMSPE(sol_gt[2:end,2].+10.0,HNKEc[:,2].+10.0), digits=3)            , " & ",round(RMSPE(sol_gt[2:end,2].+10.0,HNKEu[:,2].+10.0), digits=3)," & ",round(RMSPE(sol_gt[2:end,2].+10.0,CKFsanto[:,2].+10.0), digits=3), " & ",round(RMSPE(sol_gt[2:end,2].+10.0,UKFsanto[:,2].+10.0), digits=3), "NNline")
    println("&Gln & ",round(RMSPE(sol_gt[2:end,3].+10.0,HNKEc[:,3].+10.0), digits=3)  , " & ",round(RMSPE(sol_gt[2:end,3].+10.0,HNKEu[:,3].+10.0), digits=3), " & ",round(RMSPE(sol_gt[2:end,3].+10.0,CKFsanto[:,3].+10.0), digits=3), " & ",round(RMSPE(sol_gt[2:end,3].+10.0,UKFsanto[:,3].+10.0), digits=3),"NNline")
    println("&Lac & ",round(RMSPE(sol_gt[2:end,4].+10.0,HNKEc[:,4].+10.0), digits=3)            , " & ",round(RMSPE(sol_gt[2:end,4].+10.0,HNKEu[:,4].+10.0), digits=3)," & ",round(RMSPE(sol_gt[2:end,4].+10.0,CKFsanto[:,4].+10.0), digits=3), " & ",round(RMSPE(sol_gt[2:end,4].+10.0,UKFsanto[:,4].+10.0), digits=3), "NNline")
    println("&Amm & ",round(RMSPE(sol_gt[2:end,5],HNKEc[:,5]), digits=3)            , " & ",round(RMSPE(sol_gt[2:end,5],HNKEu[:,5]), digits=3), " & ",round(RMSPE(sol_gt[2:end,5],CKFsanto[:,5]), digits=3), " & ",round(RMSPE(sol_gt[2:end,5],UKFsanto[:,5]), digits=3),"NNline")
    println("&mAb & ",round(RMSPE(sol_gt[2:end,6],HNKEc[:,6]), digits=3)            , " & ",round(RMSPE(sol_gt[2:end,6],HNKEu[:,6]), digits=3), " & ",round(RMSPE(sol_gt[2:end,6],CKFsanto[:,6]), digits=3), " & ",round(RMSPE(sol_gt[2:end,6],UKFsanto[:,6]), digits=3),"NNline")
    println(str_title,  "RMSPE values used in the heatmap")
    println("  ",round(RMSPE(sol_gt[2:end,1],HNKEc[:,1]), digits=5)             , "  ",round(RMSPE(sol_gt[2:end,1],HNKEu[:,1]), digits=5),"  ",round(RMSPE(sol_gt[2:end,1],CKFsanto[:,1]), digits=5), "  ",round(RMSPE(sol_gt[2:end,1],UKFsanto[:,1]), digits=5), "")
    println("  ",round(RMSPE(sol_gt[2:end,2].+10.0,HNKEc[:,2].+10.0), digits=5)            , "  ",round(RMSPE(sol_gt[2:end,2].+10.0,HNKEu[:,2].+10.0), digits=5),"  ",round(RMSPE(sol_gt[2:end,2].+10.0,CKFsanto[:,2].+10.0), digits=5), "  ",round(RMSPE(sol_gt[2:end,2].+10.0,UKFsanto[:,2].+10.0), digits=5), "")
    println("  ",round(RMSPE(sol_gt[2:end,3].+10.0,HNKEc[:,3].+10.0), digits=5)  , "  ",round(RMSPE(sol_gt[2:end,3].+10.0,HNKEu[:,3].+10.0), digits=5), "  ",round(RMSPE(sol_gt[2:end,3].+10.0,CKFsanto[:,3].+10.0), digits=5), "  ",round(RMSPE(sol_gt[2:end,3].+10.0,UKFsanto[:,3].+10.0), digits=5),"")
    println("  ",round(RMSPE(sol_gt[2:end,4].+10.0,HNKEc[:,4].+10.0), digits=5)            , "  ",round(RMSPE(sol_gt[2:end,4].+10.0,HNKEu[:,4].+10.0), digits=5),"  ",round(RMSPE(sol_gt[2:end,4].+10.0,CKFsanto[:,4].+10.0), digits=5), "  ",round(RMSPE(sol_gt[2:end,4].+10.0,UKFsanto[:,4].+10.0), digits=5), "")
    println("  ",round(RMSPE(sol_gt[2:end,5].+1.0,HNKEc[:,5].+1.0), digits=5)            , "  ",round(RMSPE(sol_gt[2:end,5].+1.0,HNKEu[:,5].+1.0), digits=5), "  ",round(RMSPE(sol_gt[2:end,5].+1.0,CKFsanto[:,5].+1.0), digits=5), "  ",round(RMSPE(sol_gt[2:end,5].+1.0,UKFsanto[:,5].+1.0), digits=5),"")
    println("  ",round(RMSPE(sol_gt[2:end,6],HNKEc[:,6]), digits=5)            , "  ",round(RMSPE(sol_gt[2:end,6],HNKEu[:,6]), digits=5), "  ",round(RMSPE(sol_gt[2:end,6],CKFsanto[:,6]), digits=5), " ",round(RMSPE(sol_gt[2:end,6],UKFsanto[:,6]), digits=5),"")


end

gr()
lws = 4
default(xtickfontsize=8, ytickfontsize=6, xguidefontsize=8, yguidefontsize=8, legendfontsize=6)
pltres[1] = plot(pltres[1], legend=false, lw=lws)
pltres[3] = plot(pltres[3], ylabel="", legend=false, lw=lws)
pltres[5] = plot(pltres[5], ylabel="", legend=false, lw=lws)
pltres[7] = plot(pltres[7], ylabel="", legend=false, lw=lws)
pltres[9] = plot(pltres[9], ylabel="", legend=false, lw=lws)

pltres[2] = plot(pltres[2], legend=false, lw=lws)
pltres[4] = plot(pltres[4], ylabel="", legend=false, lw=lws)
pltres[6] = plot(pltres[6], ylabel="", legend=false, lw=lws)
pltres[8] = plot(pltres[8], ylabel="", legend=false, lw=lws)
pltres[10] = plot(pltres[10], ylabel="", legend=false, lw=lws)

pltL = scatter([NaN], label="Noisy Xv", color=:grey, lw=lws)
plot!([NaN], label="Ground truth", color=:red, lw=lws)
plot!([NaN], label="JCKF", color=:purple1, lw=lws)
plot!([NaN], label="JUKF", color=:purple4, lw=lws)
plot!([NaN], label="HNKE-C", color=:blue, linestyle=:dashdotdot, lw=lws)
plot!([NaN], label="HNKE-U", color=:lightblue, linestyle=:dot, lw=lws)
plot!(xaxis=false, yaxis=false, grid=false, legend_columns=-1)

allplots = plot(pltres[1], pltres[3], pltres[5], pltres[7], pltres[9], layout=(1, 5))
plots = plot(allplots, pltL, layout=(grid(2,1, heights=[0.98, 0.02])), size=(900, 800))
plot!(plots, left_margin=2mm, bottom_margin=-4mm)
display(plots)
savefig("plots_B1")

allplots = plot(pltres[2], pltres[4], pltres[6], pltres[8], pltres[10], layout=(1, 5))
plots = plot(allplots, pltL, layout=(grid(2,1, heights=[0.98, 0.02])), size=(900, 800))
plot!(plots, left_margin=2mm, bottom_margin=-4mm)
display(plots)
savefig("plots_B2")






##### Answering the Research Question 1


path="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/HNKEu_test_ensemble_size/"
RMSPE_HNKEu_EnSizes_SPN_testingset3 = Array(CSV.read(path*"$("RMSPE_HNKEu_EnSizes_SPN_testingset3.csv")", DataFrame))
RMSPE_HNKEu_EnSizes_SP1_testingset3 = Array(CSV.read(path*"$("RMSPE_HNKEu_EnSizes_SP1_testingset3.csv")", DataFrame))
RMSPE_HNKEu_EnSizes_SP2_testingset3 = Array(CSV.read(path*"$("RMSPE_HNKEu_EnSizes_SP2_testingset3.csv")", DataFrame))
RMSPE_HNKEu_EnSizes_SP3_testingset3 = Array(CSV.read(path*"$("RMSPE_HNKEu_EnSizes_SP3_testingset3.csv")", DataFrame))
RMSPE_HNKEu_EnSizes_SP4_testingset3 = Array(CSV.read(path*"$("RMSPE_HNKEu_EnSizes_SP4_testingset3.csv")", DataFrame))

path="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/HNKEc_test_ensemble_size/"
RMSPE_HNKEc_EnSizes_SPN_testingset3 = Array(CSV.read(path*"$("RMSPE_HNKEc_EnSizes_SPN_testingset3.csv")", DataFrame))
RMSPE_HNKEc_EnSizes_SP1_testingset3 = Array(CSV.read(path*"$("RMSPE_HNKEc_EnSizes_SP1_testingset3.csv")", DataFrame))
RMSPE_HNKEc_EnSizes_SP2_testingset3 = Array(CSV.read(path*"$("RMSPE_HNKEc_EnSizes_SP2_testingset3.csv")", DataFrame))
RMSPE_HNKEc_EnSizes_SP3_testingset3 = Array(CSV.read(path*"$("RMSPE_HNKEc_EnSizes_SP3_testingset3.csv")", DataFrame))
RMSPE_HNKEc_EnSizes_SP4_testingset3 = Array(CSV.read(path*"$("RMSPE_HNKEc_EnSizes_SP4_testingset3.csv")", DataFrame))


# ylabel=["Xv(Cell/L)" "GLC(mM)" "GLN(mM)" "LAC(mM)" "AMM(mM)" "mAb(mg/L)"]
sss=(600,900)
lws=4.5
gr( xtickfontsize=7, ytickfontsize=7, xguidefontsize=9, yguidefontsize=9, legendfontsize=6);
plots=plot(Array(2:1:100),RMSPE_HNKEu_EnSizes_SPN_testingset3,color=:green , lw=lws, label = "HNKE-U SP0", xlabel="Ensemble size", ylabel=["RMSPE (Xv)" "RMSPE (GLC)" "RMSPE (GLN)" "RMSPE (LAC)" "RMSPE (AMM)" "RMSPE (mAb)"], layout=(3,2),size =sss)
plot!(Array(2:1:100),RMSPE_HNKEu_EnSizes_SP1_testingset3,color=:purple1 , lw=lws, label = "HNKE-U SP1", layout=(3,2),size =sss)
plot!(Array(2:1:100),RMSPE_HNKEu_EnSizes_SP2_testingset3,color=:yellow2 , lw=lws, label = "HNKE-U SP2", layout=(3,2),size =sss)
plot!(Array(2:1:100),RMSPE_HNKEu_EnSizes_SP3_testingset3,color=:lightblue , lw=lws, label = "HNKE-U SP3", layout=(3,2),size =sss)
plot!(Array(2:1:100),RMSPE_HNKEu_EnSizes_SP4_testingset3,color=:red , lw=lws, label = "HNKE-U SP4", layout=(3,2),size =sss)

plot!(Array(2:1:100),RMSPE_HNKEc_EnSizes_SPN_testingset3,color=:green3 ,linestyle=:dot, lw=lws, label = "HNKE-C SP0", xlabel="Ensemble size", ylabel=["RMSPE (Xv)" "RMSPE (GLC)" "RMSPE (GLN)" "RMSPE (LAC)" "RMSPE (AMM)" "RMSPE (mAb)"], layout=(3,2),size =sss)
plot!(Array(2:1:100),RMSPE_HNKEc_EnSizes_SP1_testingset3,color=:purple4,linestyle=:dot , lw=lws, label = "HNKE-C SP1", layout=(3,2),size =sss)
plot!(Array(2:1:100),RMSPE_HNKEc_EnSizes_SP2_testingset3,color=:yellow4 ,linestyle=:dot, lw=lws, label = "HNKE-C SP2", layout=(3,2),size =sss)
plot!(Array(2:1:100),RMSPE_HNKEc_EnSizes_SP3_testingset3,color=:blue2 ,linestyle=:dot, lw=lws, label = "HNKE-C SP3", layout=(3,2),size =sss)
plot!(Array(2:1:100),RMSPE_HNKEc_EnSizes_SP4_testingset3,color=:red4 ,linestyle=:dot, lw=lws, label = "HNKE-C SP4", layout=(3,2),size =sss)
plot!(left_margin=3mm,bottom_margin=0mm,xticks=(2:4:100),xrotation=90, grid=false)
display(plots)
savefig("RMSPE_HNKEu_HNKEc_EnSizes")





# lws=2.5
# gr( xtickfontsize=7, ytickfontsize=7, xguidefontsize=9, yguidefontsize=9, legendfontsize=6);
# plots=
# plot!(Array(2:1:100),RMSPE_HNKEc_EnSizes_SPN_testingset3,color=:green , lw=lws, label = "SP0", xlabel="Ensemble size", ylabel=["Xv(Cell/L)" "GLC(mM)" "GLN(mM)" "LAC(mM)" "AMM(mM)" "mAb(mg/L)"], layout=(3,2),size =sss)
# plot!(Array(2:1:100),RMSPE_HNKEc_EnSizes_SP1_testingset3,color=:purple , lw=lws, label = "SP1", layout=(3,2),size =sss)
# plot!(Array(2:1:100),RMSPE_HNKEc_EnSizes_SP2_testingset3,color=:yellow3 , lw=lws, label = "SP2", layout=(3,2),size =sss)
# plot!(Array(2:1:100),RMSPE_HNKEc_EnSizes_SP3_testingset3,color=:blue , lw=lws, label = "SP3", layout=(3,2),size =sss)
# plot!(Array(2:1:100),RMSPE_HNKEc_EnSizes_SP4_testingset3,color=:red , lw=lws, label = "SP4", layout=(3,2),size =sss)
#
# display(plots)
# # savefig("RMSPE_HNKEu_EnSizes_$(xdt[dt_idx:dt_idx][1][3])")







# Define the data in a matrix format
data = [
# SP0 B.1
# &Xv & 15.235 & 15.233 & 3.487 & 3.486NNline
# &Glc & 4.302 & 4.302 & 29.976 & 30.557NNline
# &Gln & 2.065 & 2.066 & 22.05 & 25.236NNline
# &Lac & 6.857 & 6.857 & 27.515 & 32.176NNline
# &Amm & 8.845 & 8.845 & 104.753 & 103.284NNline
# &mAb & 19.172 & 19.171 & 21.945 & 20.314NNline
# SP0 B.1RMSPE values used in the heatmap
  15.23496  15.23255  3.4867  3.4864
  4.30178  4.302  29.97609  30.55657
  2.06544  2.06553  22.05047  25.23616
  6.85676  6.85674  27.51533  32.17561
  5.49851  5.49883  79.64699  78.51499
  19.17187  19.17129  21.94486 20.31445
# SP0 B.2
# &Xv & 11.602 & 11.6 & 3.47 & 3.464NNline
# &Glc & 4.266 & 4.266 & 29.768 & 27.114NNline
# &Gln & 2.116 & 2.117 & 22.025 & 21.009NNline
# &Lac & 6.437 & 6.437 & 28.455 & 27.726NNline
# &Amm & 8.824 & 8.825 & 107.327 & 105.667NNline
# &mAb & 19.065 & 19.064 & 22.413 & 19.692NNline
# SP0 B.2RMSPE values used in the heatmap
  # 11.60168  11.59986  3.4697  3.46389
  # 4.26554  4.26573  29.76834  27.11379
  # 2.11638  2.1165  22.02541  21.00891
  # 6.43676  6.43672  28.45495  27.72581
  # 5.74584  5.74617  81.64821  80.38034
  # 19.06469  19.06423  22.41281 19.6919
# SP1 B.1
# &Xv & 11.13 & 11.121 & 2.964 & 2.967NNline
# &Glc & 1.102 & 1.102 & 9.125 & 8.852NNline
# &Gln & 2.312 & 2.313 & 25.111 & 25.282NNline
# &Lac & 4.124 & 4.124 & 42.53 & 43.691NNline
# &Amm & 7.996 & 7.996 & 35.826 & 37.844NNline
# &mAb & 15.646 & 15.645 & 24.412 & 26.592NNline
# SP1 B.1RMSPE values used in the heatmap
  11.12985  11.12103  2.9643  2.9672
  1.10173  1.1018  9.12501  8.85189
  2.31244  2.31273  25.1113  25.28235
  4.12439  4.12433  42.53036  43.6909
  4.0863  4.0869  22.86363  24.72062
  15.6458  15.64498  24.41163 26.59175
# SP1 B.2
# &Xv & 11.36 & 11.352 & 3.656 & 3.657NNline
# &Glc & 1.475 & 1.475 & 9.052 & 7.614NNline
# &Gln & 2.528 & 2.528 & 24.213 & 22.393NNline
# &Lac & 4.284 & 4.284 & 42.089 & 42.565NNline
# &Amm & 7.875 & 7.876 & 36.339 & 36.594NNline
# &mAb & 15.787 & 15.787 & 23.161 & 18.677NNline
# SP1 B.2RMSPE values used in the heatmap
  # 11.35951  11.3522  3.65556  3.65704
  # 1.47541  1.47522  9.0516  7.6137
  # 2.52798  2.52801  24.2128  22.39311
  # 4.28411  4.28393  42.08886  42.56528
  # 4.0379  4.03821  23.19268  23.33509
  # 15.78722  15.78688  23.16119 18.6771
# SP2 B.1
# &Xv & 9.178 & 9.189 & 3.009 & 3.007NNline
# &Glc & 28.424 & 28.426 & 12.507 & 37.765NNline
# &Gln & 2.482 & 2.482 & 10.953 & 12.206NNline
# &Lac & 6.174 & 6.176 & 14.953 & 17.732NNline
# &Amm & 9.515 & 9.513 & 39.982 & 39.561NNline
# &mAb & 16.042 & 16.042 & 24.34 & 22.657NNline
# SP2 B.1RMSPE values used in the heatmap
  9.17792  9.18922  3.00895  3.00655
  28.42351  28.42551  12.50711  37.76521
  2.48162  2.48181  10.95307  12.20618
  6.17425  6.17633  14.95293  17.73203
  6.80953  6.80793  28.45515  28.09123
  16.0417  16.0419  24.34025 22.65703
# SP2 B.2
# &Xv & 9.909 & 9.919 & 3.124 & 3.112NNline
# &Glc & 28.578 & 28.579 & 14.648 & 16.832NNline
# &Gln & 2.606 & 2.606 & 11.035 & 10.501NNline
# &Lac & 5.848 & 5.85 & 18.364 & 16.883NNline
# &Amm & 8.691 & 8.689 & 40.462 & 40.127NNline
# &mAb & 15.606 & 15.606 & 22.983 & 25.088NNline
# SP2 B.2RMSPE values used in the heatmap
  # 9.90904  9.91925  3.1235  3.11156
  # 28.5777  28.57858  14.64822  16.83198
  # 2.60611  2.6063  11.03487  10.50105
  # 5.84819  5.85019  18.36389  16.88331
  # 6.6641  6.66228  28.83844  28.6066
  # 15.60552  15.60607  22.9833 25.08824
# SP3 B.1
# &Xv & 6.607 & 6.594 & 2.868 & 2.879NNline
# &Glc & 18.617 & 18.619 & 12.413 & 14.487NNline
# &Gln & 2.134 & 2.133 & 12.666 & 11.893NNline
# &Lac & 3.611 & 3.61 & 8.531 & 8.273NNline
# &Amm & 6.868 & 6.867 & 40.092 & 39.624NNline
# &mAb & 8.217 & 8.219 & 17.851 & 18.703NNline
# SP3 B.1RMSPE values used in the heatmap
  6.60675  6.59377  2.86829  2.87869
  18.61668  18.61948  12.41323  14.48664
  2.1343  2.13313  12.6661  11.89279
  3.61071  3.60981  8.53144  8.2729
  5.20129  5.20056  29.03892  28.62039
  8.21712  8.21906  17.85053 18.70342
# SP3 B.2
# &Xv & 6.795 & 6.78 & 2.707 & 2.711NNline
# &Glc & 18.488 & 18.491 & 15.067 & 15.838NNline
# &Gln & 2.138 & 2.137 & 12.38 & 12.166NNline
# &Lac & 3.583 & 3.582 & 7.813 & 7.517NNline
# &Amm & 6.847 & 6.846 & 39.692 & 38.847NNline
# &mAb & 8.174 & 8.176 & 18.741 & 24.051NNline
# SP3 B.2RMSPE values used in the heatmap
  # 6.79496  6.78028  2.70705  2.71103
  # 18.48752  18.49087  15.06741  15.83839
  # 2.1383  2.1371  12.38043  12.1663
  # 3.58302  3.58174  7.81251  7.51718
  # 5.19044  5.18949  28.72503  27.976
  # 8.17356  8.17623  18.74127 24.05063
# SP4 B.1
# &Xv & 1.306 & 1.264 & 1.474 & 1.494NNline
# &Glc & 7.056 & 7.047 & 26.906 & 23.931NNline
# &Gln & 2.825 & 2.831 & 31.519 & 28.671NNline
# &Lac & 4.895 & 4.87 & 52.471 & 53.95NNline
# &Amm & 6.194 & 6.211 & 105.138 & 106.669NNline
# &mAb & 7.088 & 7.095 & 9.687 & 10.631NNline
# SP4 B.1RMSPE values used in the heatmap
  1.30589  1.26377  1.474  1.49362
  7.05558  7.04698  26.90646  23.93125
  2.82521  2.83098  31.51859  28.67107
  4.89548  4.87046  52.47141  53.95022
  4.70497  4.72276  98.27092  99.71822
  7.08829  7.0947  9.68697 10.63149
# SP4 B.2
# &Xv & 1.21 & 1.172 & 1.565 & 1.587NNline
# &Glc & 7.123 & 7.116 & 27.619 & 19.936NNline
# &Gln & 2.854 & 2.862 & 26.138 & 26.696NNline
# &Lac & 4.852 & 4.83 & 19.613 & 20.034NNline
# &Amm & 6.207 & 6.22 & 51.979 & 43.673NNline
# &mAb & 7.156 & 7.169 & 9.964 & 7.373NNline
# SP4 B.2RMSPE values used in the heatmap
  # 1.20984  1.17181  1.56536  1.58695
  # 7.12273  7.11589  27.61937  19.93589
  # 2.85418  2.86175  26.13768  26.69622
  # 4.85229  4.83027  19.61294  20.03447
  # 4.73582  4.75134  48.16096  40.2093
  # 7.15618  7.16931  9.96355 7.37268
]

# Define row labels (entities) and column labels (conditions)
row_labels = [  "SP0 Xv", "SP0 Glc", "SP0 Gln", "SP0 Lac", "SP0 Amm", "SP0 mAb",
                "SP1 Xv", "SP1 Glc", "SP1 Gln", "SP1 Lac", "SP1 Amm", "SP1 mAb",
                "SP2 Xv", "SP2 Glc", "SP2 Gln", "SP2 Lac", "SP2 Amm", "SP2 mAb",
                "SP3 Xv", "SP3 Glc", "SP3 Gln", "SP3 Lac", "SP3 Amm", "SP3 mAb",
                "SP4 Xv", "SP4 Glc", "SP4 Gln", "SP4 Lac", "SP4 Amm", "SP4 mAb"]
                # Reverse the order of data rows and row labels
data = reverse(data, dims=1)

# Define row labels (entities) and reverse their order
row_labels = reverse([
    "SP0 Xv", "SP0 Glc", "SP0 Gln", "SP0 Lac", "SP0 Amm", "SP0 mAb",
    "SP1 Xv", "SP1 Glc", "SP1 Gln", "SP1 Lac", "SP1 Amm", "SP1 mAb",
    "SP2 Xv", "SP2 Glc", "SP2 Gln", "SP2 Lac", "SP2 Amm", "SP2 mAb",
    "SP3 Xv", "SP3 Glc", "SP3 Gln", "SP3 Lac", "SP3 Amm", "SP3 mAb",
    "SP4 Xv", "SP4 Glc", "SP4 Gln", "SP4 Lac", "SP4 Amm", "SP4 mAb"
])

col_labels = ["HNKE-C", "HNKE-U", "JCKF", "JUKF"]

# Create the heatmap
plots=heatmap(col_labels, row_labels, data, color=:Blues, clims=(0, 60), ylabel="", colorbar_title="RMSPE",
        title="Heatmap of RMPSE with B.1", size=(400, 900), xrotation=45, yticks=:all, ytickfontsize=6)
plot!(left_margin=12mm,bottom_margin=0mm)
# Annotate each cell with its value
for j in 1:size(data, 2)
    for i in 1:size(data, 1)
        if data[i, j] > 30
            annotate!(j-0.5, i-0.5, text(string(round(data[i, j], digits=5)), 6, :center, :white))
        else
            annotate!(j-0.5, i-0.5, text(string(round(data[i, j], digits=5)), 6, :center, :black))
        end
    end
end
display(plots)
savefig("rmspe_plots_B1")




data = [
# SP0 B.1
# &Xv & 15.235 & 15.233 & 3.487 & 3.486NNline
# &Glc & 4.302 & 4.302 & 29.976 & 30.557NNline
# &Gln & 2.065 & 2.066 & 22.05 & 25.236NNline
# &Lac & 6.857 & 6.857 & 27.515 & 32.176NNline
# &Amm & 8.845 & 8.845 & 104.753 & 103.284NNline
# &mAb & 19.172 & 19.171 & 21.945 & 20.314NNline
# SP0 B.1RMSPE values used in the heatmap
#   15.23496  15.23255  3.4867  3.4864
#   4.30178  4.302  29.97609  30.55657
#   2.06544  2.06553  22.05047  25.23616
#   6.85676  6.85674  27.51533  32.17561
#   5.49851  5.49883  79.64699  78.51499
#   19.17187  19.17129  21.94486 20.31445
# SP0 B.2
# &Xv & 11.602 & 11.6 & 3.47 & 3.464NNline
# &Glc & 4.266 & 4.266 & 29.768 & 27.114NNline
# &Gln & 2.116 & 2.117 & 22.025 & 21.009NNline
# &Lac & 6.437 & 6.437 & 28.455 & 27.726NNline
# &Amm & 8.824 & 8.825 & 107.327 & 105.667NNline
# &mAb & 19.065 & 19.064 & 22.413 & 19.692NNline
# SP0 B.2RMSPE values used in the heatmap
  11.60168  11.59986  3.4697  3.46389
  4.26554  4.26573  29.76834  27.11379
  2.11638  2.1165  22.02541  21.00891
  6.43676  6.43672  28.45495  27.72581
  5.74584  5.74617  81.64821  80.38034
  19.06469  19.06423  22.41281 19.6919
# SP1 B.1
# &Xv & 11.13 & 11.121 & 2.964 & 2.967NNline
# &Glc & 1.102 & 1.102 & 9.125 & 8.852NNline
# &Gln & 2.312 & 2.313 & 25.111 & 25.282NNline
# &Lac & 4.124 & 4.124 & 42.53 & 43.691NNline
# &Amm & 7.996 & 7.996 & 35.826 & 37.844NNline
# &mAb & 15.646 & 15.645 & 24.412 & 26.592NNline
# SP1 B.1RMSPE values used in the heatmap
#   11.12985  11.12103  2.9643  2.9672
#   1.10173  1.1018  9.12501  8.85189
#   2.31244  2.31273  25.1113  25.28235
#   4.12439  4.12433  42.53036  43.6909
#   4.0863  4.0869  22.86363  24.72062
#   15.6458  15.64498  24.41163 26.59175
# SP1 B.2
# &Xv & 11.36 & 11.352 & 3.656 & 3.657NNline
# &Glc & 1.475 & 1.475 & 9.052 & 7.614NNline
# &Gln & 2.528 & 2.528 & 24.213 & 22.393NNline
# &Lac & 4.284 & 4.284 & 42.089 & 42.565NNline
# &Amm & 7.875 & 7.876 & 36.339 & 36.594NNline
# &mAb & 15.787 & 15.787 & 23.161 & 18.677NNline
# SP1 B.2RMSPE values used in the heatmap
  11.35951  11.3522  3.65556  3.65704
  1.47541  1.47522  9.0516  7.6137
  2.52798  2.52801  24.2128  22.39311
  4.28411  4.28393  42.08886  42.56528
  4.0379  4.03821  23.19268  23.33509
  15.78722  15.78688  23.16119 18.6771
# SP2 B.1
# &Xv & 9.178 & 9.189 & 3.009 & 3.007NNline
# &Glc & 28.424 & 28.426 & 12.507 & 37.765NNline
# &Gln & 2.482 & 2.482 & 10.953 & 12.206NNline
# &Lac & 6.174 & 6.176 & 14.953 & 17.732NNline
# &Amm & 9.515 & 9.513 & 39.982 & 39.561NNline
# &mAb & 16.042 & 16.042 & 24.34 & 22.657NNline
# SP2 B.1RMSPE values used in the heatmap
#   9.17792  9.18922  3.00895  3.00655
#   28.42351  28.42551  12.50711  37.76521
#   2.48162  2.48181  10.95307  12.20618
#   6.17425  6.17633  14.95293  17.73203
#   6.80953  6.80793  28.45515  28.09123
#   16.0417  16.0419  24.34025 22.65703
# SP2 B.2
# &Xv & 9.909 & 9.919 & 3.124 & 3.112NNline
# &Glc & 28.578 & 28.579 & 14.648 & 16.832NNline
# &Gln & 2.606 & 2.606 & 11.035 & 10.501NNline
# &Lac & 5.848 & 5.85 & 18.364 & 16.883NNline
# &Amm & 8.691 & 8.689 & 40.462 & 40.127NNline
# &mAb & 15.606 & 15.606 & 22.983 & 25.088NNline
# SP2 B.2RMSPE values used in the heatmap
  9.90904  9.91925  3.1235  3.11156
  28.5777  28.57858  14.64822  16.83198
  2.60611  2.6063  11.03487  10.50105
  5.84819  5.85019  18.36389  16.88331
  6.6641  6.66228  28.83844  28.6066
  15.60552  15.60607  22.9833 25.08824
# SP3 B.1
# &Xv & 6.607 & 6.594 & 2.868 & 2.879NNline
# &Glc & 18.617 & 18.619 & 12.413 & 14.487NNline
# &Gln & 2.134 & 2.133 & 12.666 & 11.893NNline
# &Lac & 3.611 & 3.61 & 8.531 & 8.273NNline
# &Amm & 6.868 & 6.867 & 40.092 & 39.624NNline
# &mAb & 8.217 & 8.219 & 17.851 & 18.703NNline
# SP3 B.1RMSPE values used in the heatmap
#   6.60675  6.59377  2.86829  2.87869
#   18.61668  18.61948  12.41323  14.48664
#   2.1343  2.13313  12.6661  11.89279
#   3.61071  3.60981  8.53144  8.2729
#   5.20129  5.20056  29.03892  28.62039
#   8.21712  8.21906  17.85053 18.70342
# SP3 B.2
# &Xv & 6.795 & 6.78 & 2.707 & 2.711NNline
# &Glc & 18.488 & 18.491 & 15.067 & 15.838NNline
# &Gln & 2.138 & 2.137 & 12.38 & 12.166NNline
# &Lac & 3.583 & 3.582 & 7.813 & 7.517NNline
# &Amm & 6.847 & 6.846 & 39.692 & 38.847NNline
# &mAb & 8.174 & 8.176 & 18.741 & 24.051NNline
# SP3 B.2RMSPE values used in the heatmap
  6.79496  6.78028  2.70705  2.71103
  18.48752  18.49087  15.06741  15.83839
  2.1383  2.1371  12.38043  12.1663
  3.58302  3.58174  7.81251  7.51718
  5.19044  5.18949  28.72503  27.976
  8.17356  8.17623  18.74127 24.05063
# SP4 B.1
# &Xv & 1.306 & 1.264 & 1.474 & 1.494NNline
# &Glc & 7.056 & 7.047 & 26.906 & 23.931NNline
# &Gln & 2.825 & 2.831 & 31.519 & 28.671NNline
# &Lac & 4.895 & 4.87 & 52.471 & 53.95NNline
# &Amm & 6.194 & 6.211 & 105.138 & 106.669NNline
# &mAb & 7.088 & 7.095 & 9.687 & 10.631NNline
# SP4 B.1RMSPE values used in the heatmap
#   1.30589  1.26377  1.474  1.49362
#   7.05558  7.04698  26.90646  23.93125
#   2.82521  2.83098  31.51859  28.67107
#   4.89548  4.87046  52.47141  53.95022
#   4.70497  4.72276  98.27092  99.71822
#   7.08829  7.0947  9.68697 10.63149
# SP4 B.2
# &Xv & 1.21 & 1.172 & 1.565 & 1.587NNline
# &Glc & 7.123 & 7.116 & 27.619 & 19.936NNline
# &Gln & 2.854 & 2.862 & 26.138 & 26.696NNline
# &Lac & 4.852 & 4.83 & 19.613 & 20.034NNline
# &Amm & 6.207 & 6.22 & 51.979 & 43.673NNline
# &mAb & 7.156 & 7.169 & 9.964 & 7.373NNline
# SP4 B.2RMSPE values used in the heatmap
  1.20984  1.17181  1.56536  1.58695
  7.12273  7.11589  27.61937  19.93589
  2.85418  2.86175  26.13768  26.69622
  4.85229  4.83027  19.61294  20.03447
  4.73582  4.75134  48.16096  40.2093
  7.15618  7.16931  9.96355 7.37268
]

# Define row labels (entities) and column labels (conditions)
row_labels = [  "SP0 Xv", "SP0 Glc", "SP0 Gln", "SP0 Lac", "SP0 Amm", "SP0 mAb",
                "SP1 Xv", "SP1 Glc", "SP1 Gln", "SP1 Lac", "SP1 Amm", "SP1 mAb",
                "SP2 Xv", "SP2 Glc", "SP2 Gln", "SP2 Lac", "SP2 Amm", "SP2 mAb",
                "SP3 Xv", "SP3 Glc", "SP3 Gln", "SP3 Lac", "SP3 Amm", "SP3 mAb",
                "SP4 Xv", "SP4 Glc", "SP4 Gln", "SP4 Lac", "SP4 Amm", "SP4 mAb"]
                # Reverse the order of data rows and row labels
data = reverse(data, dims=1)

# Define row labels (entities) and reverse their order
row_labels = reverse([
    "SP0 Xv", "SP0 Glc", "SP0 Gln", "SP0 Lac", "SP0 Amm", "SP0 mAb",
    "SP1 Xv", "SP1 Glc", "SP1 Gln", "SP1 Lac", "SP1 Amm", "SP1 mAb",
    "SP2 Xv", "SP2 Glc", "SP2 Gln", "SP2 Lac", "SP2 Amm", "SP2 mAb",
    "SP3 Xv", "SP3 Glc", "SP3 Gln", "SP3 Lac", "SP3 Amm", "SP3 mAb",
    "SP4 Xv", "SP4 Glc", "SP4 Gln", "SP4 Lac", "SP4 Amm", "SP4 mAb"
])

col_labels = ["HNKE-C", "HNKE-U", "JCKF", "JUKF"]

# Create the heatmap
plots=heatmap(col_labels, row_labels, data, color=:Blues, clims=(0, 60), ylabel="", colorbar_title="RMSPE",
        title="Heatmap of RMPSE with B.2", size=(400, 900), xrotation=45, yticks=:all, ytickfontsize=6)
plot!(left_margin=12mm,bottom_margin=0mm)
# Annotate each cell with its value
for j in 1:size(data, 2)
    for i in 1:size(data, 1)
        if data[i, j] > 30
            annotate!(j-0.5, i-0.5, text(string(round(data[i, j], digits=5)), 6, :center, :white))
        else
            annotate!(j-0.5, i-0.5, text(string(round(data[i, j], digits=5)), 6, :center, :black))
        end
    end
end
display(plots)
savefig("rmspe_plots_B2")
