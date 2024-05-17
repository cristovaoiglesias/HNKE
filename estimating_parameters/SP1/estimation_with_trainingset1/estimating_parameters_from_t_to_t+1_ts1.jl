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


sol_SP1_trainingset1=[2.205841592557993e8 2.058688151252709e8 100.72750841246663 4.9713482419119766 0.2671396783756082 0.021514015238271567 100.60922257316341; 2.6396667473759595e8 3.782952161789805e8 99.2175437067149 4.780761332807146 0.07398585319684914 0.8778777280055013 63.38978155056393; 4.071624303515834e8 2.993186653598452e8 94.95059330585138 3.9782879177341597 0.6789622247738918 0.3482863814535705 51.08746103977444; 5.941089307159216e8 6.118013463392375e8 95.86733473102815 3.226261724289963 1.4383314825968974 1.5467264079442642 153.15485133076268; 4.9565956379776293e8 9.617201698834172e8 94.3907892073811 3.1278287827083027 5.332004822610516 1.3300595520500016 389.81120910973414; 8.064880514152669e8 1.2184402130938983e9 90.39997005622335 1.6861135759609207 9.675773874131863 1.2536160130020813 306.13942700113176; 1.2019138965688155e9 1.2568439707814662e9 85.92549703277703 1.4387607963519708 16.230652862046448 2.6400955825521257 280.8139113096503; 1.1693209645071886e9 1.6367636342945857e9 88.59437571167494 0.7004621419202275 20.07945413128332 2.502337878637426 497.42777518075104; 1.6859149982726007e9 1.855410343213831e9 80.9734980580663 0.25770496146233984 24.147474751798047 3.546501406585847 805.6893005126789; 1.2271781924950643e9 1.8168840451471846e9 82.30994087035812 0.0016493210047543638 28.336015737318668 3.398499966126689 838.0844013430267; 1.0267680649142337e9 1.3010300457043517e9 83.37596438669982 0.21306120509988738 30.264217956183998 3.622455398001426 946.6211845416598; 7.020293783221964e8 1.2660314138835185e9 82.54087940226695 0.05710218066544668 24.99773891371841 3.496190993016747 930.9953242364337; 7.380139602131093e8 1.0410262008250856e9 79.74926894746896 0.5888932251177958 25.648455018996394 3.0986317357136435 926.1471740201077; 5.654258889051758e8 9.00574638720496e8 80.26935822089374 0.022257438551802372 27.74488473799487 2.791549058359265 1119.16734269183; 6.02046762545369e8 9.781467358178523e8 80.338370418608 0.2905518630820171 23.73224938537546 3.392564444746878 1231.724815005269]'

trainingset=sol_SP1_trainingset1

full_path="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/estimating_parameters/SP1/estimation_with_trainingset1"


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
# # all_estimated_parameters
# 14-element Vector{Array}:
#  [0.025649080913742482, 8.927376770802995e-10, 1.1268083882662348e-10, -1.14198478764008e-10, 5.063086153189999e-10, -2.2005281227689823e-8]
#  [0.06191276202255822, 1.8448785388628232e-9, 3.469616611409715e-10, 2.6157094965231086e-10, -2.2897714060819297e-10, -5.319087360502091e-9]
#  [0.05397864140192608, -2.646980512005367e-10, 2.1713843168519e-10, 2.1925865665928306e-10, 3.4603526668192053e-10, 2.947077889258591e-8]
#  [-0.025881908643024268, 3.881765821542538e-10, 2.587680242993214e-11, 1.0236289296849868e-9, -5.695998218524636e-11, 6.221592221379952e-8]
#  [0.06954282271147261, 8.928821036409446e-10, 3.225615225429165e-10, 9.71848907923389e-10, -1.71041744475493e-11, -1.8720202639544795e-8]
#  [0.05699734257591693, 6.449593798680048e-10, 3.565519906994555e-11, 9.448325745697004e-10, 1.9985083949017202e-10, -3.650463538438916e-9]
#  [-0.003927427423509276, -3.21599149153099e-10, 8.896549358229496e-11, 4.637790650346114e-10, -1.6600693953034985e-11, 2.610183273986691e-8]
#  [0.05226931927512667, 7.710865784409374e-10, 4.479969495716445e-11, 4.116062003000352e-10, 1.0565037912442706e-10, 3.119010223407902e-8]
#  [-0.04537015280425088, -1.3217848352678373e-10, 2.532564228373549e-11, 4.142577244309637e-10, -1.4638826872483768e-11, 3.2039530666988203e-9]
#  [-0.025471616205087276, -1.3548784823909062e-10, -2.6868888890324357e-11, 2.450685629253694e-10, 2.846314182926537e-11, 1.3794747372270318e-8]
#  [-0.0543137306543838, 1.3967188574825268e-10, 2.608559004354041e-11, -8.808387170985888e-10, -2.111905655148083e-11, -2.613483046336825e-9]
#  [0.007141069726763664, 5.539912094143459e-10, -1.0553447791452251e-10, 1.2913492307714137e-10, -7.889644292254775e-11, -9.621074931456805e-10]
#  [-0.03805478759886589, -1.1467785613196542e-10, 1.2494112437315724e-10, 4.6225286746047156e-10, -6.771099040819185e-11, 4.255996146569576e-8]
#  [0.008965126867513497, -1.689568919675393e-11, -6.568181219828812e-11, -9.82330928359539e-10, 1.4713496549609643e-10, 2.7555105890595132e-8]
# 14-element Vector{Array}:
#  [2.205841592557993e8, 100.72750841246663, 4.9713482419119766, 0.2671396783756082, 0.021514015238271567, 100.60922257316341, 0.0]
#  [2.6396667473759595e8, 99.2175437067149, 4.780761332807146, 0.07398585319684914, 0.8778777280055013, 63.38978155056393, 7.0]
#  [4.071624303515834e8, 94.95059330585138, 3.9782879177341597, 0.6789622247738918, 0.3482863814535705, 51.08746103977444, 14.0]
#  [5.941089307159216e8, 95.86733473102815, 3.226261724289963, 1.4383314825968974, 1.5467264079442642, 153.15485133076268, 21.0]
#  [4.9565956379776293e8, 94.3907892073811, 3.1278287827083027, 5.332004822610516, 1.3300595520500016, 389.81120910973414, 28.0]
#  [8.064880514152669e8, 90.39997005622335, 1.6861135759609207, 9.675773874131863, 1.2536160130020813, 306.13942700113176, 35.0]
#  [1.2019138965688155e9, 85.92549703277703, 1.4387607963519708, 16.230652862046448, 2.6400955825521257, 280.8139113096503, 42.0]
#  [1.1693209645071886e9, 88.59437571167494, 0.7004621419202275, 20.07945413128332, 2.502337878637426, 497.42777518075104, 49.0]
#  [1.6859149982726007e9, 80.9734980580663, 0.25770496146233984, 24.147474751798047, 3.546501406585847, 805.6893005126789, 56.0]
#  [1.2271781924950643e9, 82.30994087035812, 0.0016493210047543638, 28.336015737318668, 3.398499966126689, 838.0844013430267, 63.0]
#  [1.0267680649142337e9, 83.37596438669982, 0.21306120509988738, 30.264217956183998, 3.622455398001426, 946.6211845416598, 70.0]
#  [7.020293783221964e8, 82.54087940226695, 0.05710218066544668, 24.99773891371841, 3.496190993016747, 930.9953242364337, 77.0]
#  [7.380139602131093e8, 79.74926894746896, 0.5888932251177958, 25.648455018996394, 3.0986317357136435, 926.1471740201077, 84.0]
#  [5.654258889051758e8, 80.26935822089374, 0.022257438551802372, 27.74488473799487, 2.791549058359265, 1119.16734269183, 91.0]
