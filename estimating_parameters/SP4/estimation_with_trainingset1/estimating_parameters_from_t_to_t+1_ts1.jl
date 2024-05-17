# using DifferentialEquations, DiffEqFlux, Plots, Statistics
# using XLSX, DataFrames, Optim, BlackBoxOptim,DiffEqParamEstim, CSV, Tables
# using Flux
# using BSON: @save
# using BSON#: @load
# using OrdinaryDiffEq, Optimization, OptimizationOptimJL, OptimizationOptimisers
# using Random, ComponentArrays, Lux

# using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL
# using OptimizationOptimisers, Random, Plots, DifferentialEquations

using Flux, DiffEqFlux, DifferentialEquations, Plots

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
tstart=0.0
tend=103.0
sampling= 7.0
tgrid=tstart:sampling:tend
tgrid_opt=[0.0, 7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, 56.0, 63.0, 70.0, 77.0, 84.0, 91.0, 98.0]


sol_SP4_trainingset1=[2.137568131576367e9 7.348539083599892e7 95.93663335085945 24.39401771397516 0.563092128050637 0.822761729820102 105.21152017451232; 2.6462110193448324e9 1.7660798294112093e9 97.77727630937464 23.506321047772158 9.669834033250785 1.6118350289996246 391.8538128275463; 3.944620204589727e9 3.5202752464417696e9 78.23898016623474 20.33150863949406 26.986469357655366 3.9970909955390335 487.4334786702161; 3.5189471762705126e9 3.51207869433211e9 74.60797585227333 15.536811235742862 44.47392611261586 5.830344675267039 684.0653880611037; 3.5451866705072575e9 4.752865984461549e9 58.21580167937214 13.33127952775958 52.53681178400917 9.442323174652715 1233.598802545304; 3.5573699126380763e9 4.15112801042587e9 61.72039008222022 11.2538785199427 72.7674622317538 9.52333781342074 2298.44267104627; 3.532526586033545e9 5.11185345052132e9 58.91357477233689 7.22402214572795 67.23364954104889 10.097079610389734 2179.608897409749; 3.5441260283663855e9 5.8147165986542015e9 42.269531135664415 6.382525354549756 85.14743675792622 11.940818276219144 2540.7555500298113; 3.3883835491811643e9 5.74425931077491e9 33.43217517491312 5.297236918918164 80.58107478380649 13.220443359810718 2576.509185243928; 3.707268684354623e9 5.638724346588853e9 30.15402468708428 4.693999878682582 88.30076074091377 14.223703509254333 3295.3288560085375; 3.4480124163402095e9 5.574105912563596e9 20.40527091336932 2.395817602102666 102.49072975568492 13.261170771912038 3521.4849200997915; 3.1128341506485972e9 5.456023297865833e9 5.202733294442428 2.591990954876354 115.01270260011015 14.451283907642143 3575.540375821441; 2.861336965640974e9 4.593848925903505e9 16.324678746115858 1.7699067254448595 112.6806852889971 14.64980221326592 4304.758757860345; 2.692816880186907e9 4.440145734899661e9 10.185868953848399 0.5044222100558318 122.53220374215712 15.491530166856093 4603.578262850062; 2.5537938797534943e9 3.4936683723833013e9 7.1760430856408455 1.094680766243127 125.1686723155188 14.296705938506763 4733.721285437861]'

trainingset=sol_SP4_trainingset1

full_path="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/estimating_parameters/SP4/estimation_with_trainingset1"


all_estimated_parameters=Array[]
all_initial_condition=Array[]

for t = 1:14
    p=zeros(6)

    tstart=tgrid_opt[t]
    tend=tgrid_opt[t+1]
    u0=[trainingset[:,t][1];trainingset[:,t][3:end]]
    push!(all_initial_condition, [u0;tstart])

    prob = ODEProblem(ode_system!, u0, (tstart,tend), p)
    function loss_func()
      sol = solve(prob, AutoTsit5(Rosenbrock23()), p=p, save_everystep=false, save_start=false)#, maxiters=1e7)
      l=Flux.Losses.mse(sol[1,1],trainingset[:,t+1][1])+
      Flux.Losses.mse(sol[2,1],trainingset[:,t+1][3])+
      Flux.Losses.mse(sol[3,1],trainingset[:,t+1][4])+
      Flux.Losses.mse(sol[4,1],trainingset[:,t+1][5])+
      Flux.Losses.mse(sol[5,1],trainingset[:,t+1][6])+
      Flux.Losses.mse(sol[6,1],trainingset[:,t+1][7])
      return l
    end

    epochs = 600
    learning_rate = 0.05
    data = Iterators.repeated((), epochs)
    opt = Adam(learning_rate)
    counter=0

    callback_func = function ()
      global counter=counter+1
      # println("loss: ", loss_func(), "    epoch: ",counter)
    end

    fparams = Flux.params(p)
    Flux.train!(loss_func, fparams, data, opt, cb=callback_func)

    # p = round.(p;digits=4)
    push!(all_estimated_parameters, p)
    println("\n\nParameters estimated: ", p)
    println("loss: ", loss_func(), "    epoch: ",counter, "    t=", t)

    prob = ODEProblem(ode_system!, u0, (tstart,tend), p)
    sol = solve(prob, AutoTsit5(Rosenbrock23()), p=p, saveat = tgrid)

    plots=plot(sol.t,sol', title="from t"*string(t)*" to t"*string(t+1), idxs = (1,2,3,4,5,6), color=[:blue :yellow :orange :green :lightgreen :purple ], label = ["Prediction" "Prediction" "Prediction" "Prediction" "Prediction" "Prediction"], ylabel=["[Xv]"  "[GLC]" "[GLN]" "[LAC]" "[AMM]" "[mAb]"], layout=(3,2),size = (800, 600))
    Plots.scatter!([tgrid_opt[t]],[trainingset[:,t][1];trainingset[:,t][3:end]]' , color=:red,   labels = false, layout=(3,2))#,size = (600, 1000)),title = ["p 1" "p 2" "p 3" "p 4" "p 5" "p 6"],
    Plots.scatter!([tgrid_opt[t+1]],[trainingset[:,t+1][1];trainingset[:,t+1][3:end]]' , color=:red,   labels = "Observed values", layout=(3,2))#,size = (600, 1000)),title = ["p 1" "p 2" "p 3" "p 4" "p 5" "p 6"],
    display(plots)
    savefig(full_path*"/from t"*string(t)*" to t"*string(t+1))

end





println("\n\n all_estimated_parameters ")
# all_estimated_parameters = hcat(all_estimated_parameters...)'
display(all_estimated_parameters)
println("\n\n all_initial_condition")
# all_initial_condition = hcat(all_initial_condition...)'
display(all_initial_condition)

# # #
# all_estimated_parameters
# 14-element Vector{Array}:
# [0.03049428814633751, -1.1035013680725441e-10, 5.3218857624271616e-11, 5.459691542096036e-10, 4.730616436879576e-11, 1.7184851283133763e-8]
# [0.057031980627998524, 8.582113402003793e-10, 1.3945341860038303e-10, 7.606259886671735e-10, 1.0477254280124364e-10, 4.19829085666726e-9]
# [-0.016312975694541522, 1.3915060122241422e-10, 1.8374656062416068e-10, 6.701684408339317e-10, 7.025579124015388e-11, 7.535482677806045e-9]
# [0.0010612812548054075, 6.62995797841103e-10, 8.920357529679012e-11, 3.26109817010075e-10, 1.460886509815234e-10, 2.222639939842818e-8]
# [0.0004900953193934319, -1.4097829338105078e-10, 8.356682156661864e-11, 8.138176437005922e-10, 3.2581808398271707e-12, 4.2835476841394916e-8]
# [-0.0010011602239965306, 1.1311070916189512e-10, 1.6239799203672179e-10, -2.230058674090443e-10, 2.312016908836598e-11, -4.788876456468831e-9]
# [0.00046831878423254355, 6.719900029301305e-10, 3.397402646318241e-11, 7.23254947632259e-10, 7.443879043848967e-11, 1.4581024452565034e-8]
# [-0.006419802091280098, 3.6428240753606655e-10, 4.473735103995637e-11, -1.882293726095346e-10, 5.274802883016013e-11, 1.4737882196064164e-9]
# [0.012848917612282573, 1.3208576243845334e-10, 2.430481338467736e-11, 3.110496495001299e-10, 4.042304345028068e-11, 2.8963577193386825e-8]
# [-0.01035677828422244, 3.8944247405291316e-10, 9.180678841081518e-11, 5.668602927885893e-10, -3.845021481892278e-11, 9.034489127063496e-9]
# [-0.014609191452050679, 6.626222387965895e-10, -8.549851054994824e-12, 5.457862503147683e-10, 5.187204126571672e-11, 2.3560784593346475e-9]
# [-0.012034946795065789, -5.322193929244513e-10, 3.9338004202469924e-11, -1.1159314416196342e-10, 9.498375118695607e-12, 3.489543665036827e-8]
# [-0.008671596147636188, 3.158866191742585e-10, 6.511823993867489e-11, 5.069327489869429e-10, 4.331285241469465e-11, 1.5376458076702994e-8]
# [-0.00757253765602599, 1.6394469776169246e-10, -3.2151652547752433e-11, 1.4360804723904656e-10, -6.508214824845657e-11, 7.0888486093902324e-9]
#
#
# all_initial_condition
# 14-element Vector{Array}:
# [2.137568131576367e9, 95.93663335085945, 24.39401771397516, 0.563092128050637, 0.822761729820102, 105.21152017451232, 0.0]
# [2.6462110193448324e9, 97.77727630937464, 23.506321047772158, 9.669834033250785, 1.6118350289996246, 391.8538128275463, 7.0]
# [3.944620204589727e9, 78.23898016623474, 20.33150863949406, 26.986469357655366, 3.9970909955390335, 487.4334786702161, 14.0]
# [3.5189471762705126e9, 74.60797585227333, 15.536811235742862, 44.47392611261586, 5.830344675267039, 684.0653880611037, 21.0]
# [3.5451866705072575e9, 58.21580167937214, 13.33127952775958, 52.53681178400917, 9.442323174652715, 1233.598802545304, 28.0]
# [3.5573699126380763e9, 61.72039008222022, 11.2538785199427, 72.7674622317538, 9.52333781342074, 2298.44267104627, 35.0]
# [3.532526586033545e9, 58.91357477233689, 7.22402214572795, 67.23364954104889, 10.097079610389734, 2179.608897409749, 42.0]
# [3.5441260283663855e9, 42.269531135664415, 6.382525354549756, 85.14743675792622, 11.940818276219144, 2540.7555500298113, 49.0]
# [3.3883835491811643e9, 33.43217517491312, 5.297236918918164, 80.58107478380649, 13.220443359810718, 2576.509185243928, 56.0]
# [3.707268684354623e9, 30.15402468708428, 4.693999878682582, 88.30076074091377, 14.223703509254333, 3295.3288560085375, 63.0]
# [3.4480124163402095e9, 20.40527091336932, 2.395817602102666, 102.49072975568492, 13.261170771912038, 3521.4849200997915, 70.0]
# [3.1128341506485972e9, 5.202733294442428, 2.591990954876354, 115.01270260011015, 14.451283907642143, 3575.540375821441, 77.0]
# [2.861336965640974e9, 16.324678746115858, 1.7699067254448595, 112.6806852889971, 14.64980221326592, 4304.758757860345, 84.0]
# [2.692816880186907e9, 10.185868953848399, 0.5044222100558318, 122.53220374215712, 15.491530166856093, 4603.578262850062, 91.0]
