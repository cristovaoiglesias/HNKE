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


sol_SPN_trainingset2=[2.945647256448397e8 7.919760033462238e7 31.686856232377806 5.01236496192192 0.7599895909790566 0.8604668241991238 66.60268337932095; 3.834078232859567e8 5.586328280103637e8 28.17125607702335 4.436563628209611 1.2638540086428145 0.9363290016412311 218.3220651590721; 3.6249229169838053e8 1.871920861451829e8 26.763437130877453 4.351969582356733 5.845913721064459 0.9841690077308585 92.22921964998648; 7.022585918507075e8 6.609203068651861e8 27.337214238485295 3.691513350556881 4.844159699687917 1.3108005502971778 322.3795101231353; 8.124867789061472e8 7.438458743333207e8 20.288461522435256 2.9166106313971234 8.601849124166018 1.4068754349353652 80.12473988769162; 7.418523195043583e8 9.080042421973882e8 20.099979019721584 2.24033452905882 10.17153166596628 1.9695694987354304 290.0626172014355; 1.0837093774939625e9 1.1581372497030532e9 22.053088491081986 1.222580766047908 11.383345131405285 2.9300747680051615 435.96857939007276; 1.139196272396505e9 1.3443434973231044e9 11.86443853203598 0.7016730360247637 18.191114199489995 2.697816419197709 671.6867366442526; 1.3042532554191854e9 1.4704178567860303e9 9.473901498103075 0.14145402221510886 25.61475385830961 3.453514246173081 620.9038507847961; 1.1927706369942985e9 1.3578190540148034e9 11.732534628187999 0.0793682735388306 26.117076028049457 3.157592571704592 681.2617059022672; 9.966063708426251e8 1.535075823252565e9 12.997463809910457 0.07971021890530561 25.404623755023724 3.683717165598622 949.3455311363595; 8.155738284431202e8 1.396745988296804e9 11.795988927769324 0.11411833976720082 26.09715073951782 3.551786214302725 1118.077951312577; 6.129262612459478e8 1.2950100227458928e9 10.792051049859554 0.010880547978754485 27.03646426067376 3.127103065424377 996.9061871362524; 6.17248240369503e8 1.280327972254227e9 11.48217895630597 0.42344377522728166 30.377181685022897 3.0622738279205715 1122.7651212319045; 6.220110221011975e8 9.51137119578363e8 10.472990787137228 0.15310194948552822 21.477372269875627 3.11198046636705 1196.1630256425005]'

trainingset=sol_SPN_trainingset2

full_path="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/estimating_parameters/SPN/estimation_with_trainingset2"


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

# #
#
#  all_estimated_parameters
# 14-element Vector{Array}:
#  [0.037657210687658337, 1.490128799683545e-9, 2.4405986980264796e-10, 2.1356854660745098e-10, 3.215467245612029e-11, 6.430807614135169e-8]
#  [-0.008013717948401049, 5.394014351401667e-10, 3.241211531084257e-11, 1.7556014678876617e-9, 1.8329924432764015e-11, -4.831206411406057e-8]
#  [0.09447121366153427, -1.5953811857209485e-10, 1.8363904340761382e-10, -2.785360840710137e-10, 9.081988056471128e-11, 6.399274298999088e-8]
#  [0.020828277324581014, 1.3319040564048222e-9, 1.4642245575547543e-10, 7.100379007484355e-10, 1.815386338670019e-11, -4.5775492326678515e-8]
#  [-0.01299277841754444, 3.466896375333009e-11, 1.2439562418300431e-10, 2.887322950228331e-10, 1.0350290027134776e-10, 3.861679209538456e-8]
#  [0.054142121820437025, -3.0932590236889e-10, 1.611875459338504e-10, 1.9192204096216316e-10, 1.5212073028767748e-10, 2.3108073817536372e-8]
#  [0.007133317773915472, 1.3098402071332031e-9, 6.696875798988017e-11, 8.751988779803107e-10, -2.986039636550022e-11, 3.0303598927240623e-8]
#  [0.019329667051616494, 2.799547045519709e-10, 6.560789365371505e-11, 8.693765890042996e-10, 8.85002324383301e-11, -5.947136871030633e-9]
#  [-0.012764541662319365, -2.5860778329794877e-10, 7.1074062950999095e-12, 5.751361505935983e-11, -3.388116223194499e-11, 6.910855013234406e-9]
#  [-0.02566832401886941, -1.6551654996428118e-10, -4.382871183480593e-14, -9.322429927335886e-11, 6.884310418916708e-11, 3.50790816580427e-8]
#  [-0.028637704213382673, 1.9006339058327014e-10, -5.44402732410293e-12, 1.0955245503687852e-10, -2.0871253473721045e-11, 2.6691937552860497e-8]
#  [-0.04080675998715948, 2.0216196103042028e-10, 2.0789676386638003e-11, 1.8914867568862347e-10, -8.551852420388893e-11, -2.4400131443055087e-8]
#  [0.0010038057668216483, -1.6028516651728092e-10, -9.581909260117643e-11, 7.759005981210444e-10, -1.5055772315590455e-11, 2.9231496730979443e-8]
#  [0.0010980765055851487, 2.3267069985925703e-10, 6.23270106764934e-11, -2.051881886246757e-9, 1.1458772507942112e-11, 1.692215030474486e-8]
#
#
#  all_initial_condition
# 14-element Vector{Array}:
#  [2.945647256448397e8, 31.686856232377806, 5.01236496192192, 0.7599895909790566, 0.8604668241991238, 66.60268337932095, 0.0]
#  [3.834078232859567e8, 28.17125607702335, 4.436563628209611, 1.2638540086428145, 0.9363290016412311, 218.3220651590721, 7.0]
#  [3.6249229169838053e8, 26.763437130877453, 4.351969582356733, 5.845913721064459, 0.9841690077308585, 92.22921964998648, 14.0]
#  [7.022585918507075e8, 27.337214238485295, 3.691513350556881, 4.844159699687917, 1.3108005502971778, 322.3795101231353, 21.0]
#  [8.124867789061472e8, 20.288461522435256, 2.9166106313971234, 8.601849124166018, 1.4068754349353652, 80.12473988769162, 28.0]
#  [7.418523195043583e8, 20.099979019721584, 2.24033452905882, 10.17153166596628, 1.9695694987354304, 290.0626172014355, 35.0]
#  [1.0837093774939625e9, 22.053088491081986, 1.222580766047908, 11.383345131405285, 2.9300747680051615, 435.96857939007276, 42.0]
#  [1.139196272396505e9, 11.86443853203598, 0.7016730360247637, 18.191114199489995, 2.697816419197709, 671.6867366442526, 49.0]
#  [1.3042532554191854e9, 9.473901498103075, 0.14145402221510886, 25.61475385830961, 3.453514246173081, 620.9038507847961, 56.0]
#  [1.1927706369942985e9, 11.732534628187999, 0.0793682735388306, 26.117076028049457, 3.157592571704592, 681.2617059022672, 63.0]
#  [9.966063708426251e8, 12.997463809910457, 0.07971021890530561, 25.404623755023724, 3.683717165598622, 949.3455311363595, 70.0]
#  [8.155738284431202e8, 11.795988927769324, 0.11411833976720082, 26.09715073951782, 3.551786214302725, 1118.077951312577, 77.0]
#  [6.129262612459478e8, 10.792051049859554, 0.010880547978754485, 27.03646426067376, 3.127103065424377, 996.9061871362524, 84.0]
#  [6.17248240369503e8, 11.48217895630597, 0.42344377522728166, 30.377181685022897, 3.0622738279205715, 1122.7651212319045, 91.0]