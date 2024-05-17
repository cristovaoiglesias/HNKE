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


sol_SPN_trainingset1=[2.4445259281844485e8 1.6564223476446635e8 33.74952720254497 5.139062262525559 1.5783812811037112 0.09531671978388118 28.86266232140354; 3.396325913932981e8 2.1975461024354476e8 30.375377537229326 4.355571739860554 1.7698160683231383 0.9559814163465556 150.9683637002025; 1.925462156304023e8 5.691110037334551e8 23.743347891022513 3.8492033693535306 5.3632894521199646 1.2186240634420904 84.9430721943757; 9.283195093991194e8 5.834638178723078e8 26.613358421344145 3.4266248520010416 7.8521202736084525 1.3409830680862749 207.9531264009106; 5.525560045300707e8 5.980079429584424e8 23.940834493503903 2.1881037636138467 11.065370022287514 1.0501575765649303 286.5860713564228; 1.0079914278203977e9 1.1654186730096722e9 21.28118960369631 2.066165222156412 9.52291903850498 2.1806230127826 111.93247173574943; 1.072369763975047e9 1.1351778039138393e9 15.822812349802081 1.4824427596876795 15.788517696985483 2.768002186229957 408.29436822135057; 1.192771485837667e9 1.3921199362927597e9 17.154021575289445 0.7169996491801401 24.478054149833238 3.166999239447984 551.1680663627948; 1.3621758509196446e9 1.703806257699498e9 11.966234284296142 0.056423160999699976 28.24196074260893 3.6288932552977715 731.5438144618658; 1.2925887664939604e9 1.5395943981777983e9 14.769421501658897 0.26206272131741215 28.012150428503496 3.220420651596536 864.1204111534144; 9.698066291612679e8 1.5081064206609604e9 11.9108580462937 0.5215165798889059 23.89743463812208 3.0055545524602794 1018.966756039457; 8.835645788096658e8 1.2060134172083497e9 12.96907822023761 0.1514646872129411 24.433983674985388 2.9161937589760423 1022.4793372740894; 7.708372616848432e8 1.1589839739480634e9 9.617649483532638 0.21588119419337715 22.743242569978264 3.1456234516865806 1128.6298554718699; 8.76601179094303e8 8.078472804617661e8 12.672320414070091 0.26469348331981074 20.964849285696864 3.153107484379107 1220.3586369023133; 4.6584261475269645e8 1.1251701167518518e9 9.11889078753297 0.07909151331644906 23.994123634738003 3.1790796139044954 1289.3877580260246]'

trainingset=sol_SPN_trainingset1

full_path="/Users/cristovao/PhD_courses/Thesis/BHM_NSE_vs_EnMLP_NSE/estimating_parameters/SPN/estimation_with_trainingset1"


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

    epochs = 550
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
#
# #
# all_estimated_parameters
# 14-element Vector{Array}:
# [0.04697757483881504, 1.6653490479117395e-9, 3.86688766486935e-10, 9.447041138158076e-11, 4.2477928533989273e-10, 6.026716077550061e-8]
# [-0.08107546324058655, 3.6556449960585604e-9, 2.791199910543287e-10, 1.9807627690834076e-9, 1.447759647650815e-10, -3.639379762511664e-8]
# [0.2247201064149725, -8.76571055667918e-10, 1.2907561474663883e-10, 7.601510313262345e-10, 3.7382478497752586e-11, 3.7569780282384976e-8]
# [-0.07411731833854303, 5.271379978248598e-10, 2.442886963433655e-10, 6.337932404153123e-10, -5.736086732119152e-11, 1.5509920065475417e-8]
# [0.08588002251355902, 5.015371963609382e-10, 2.3009907681295157e-11, -2.9087146227111396e-10, 2.1318464591445362e-10, -3.29338952113673e-8]
# [0.008844466684061981, 7.498680457582508e-10, 8.017503297866003e-11, 8.607662647189983e-10, 8.067740163796432e-11, 4.071496558767031e-8]
# [0.015201235241261817, -1.680902408967688e-10, 9.665983547890558e-11, 1.0971107020574699e-9, 5.039444463466816e-11, 1.803843825118847e-8]
# [0.018971961814644823, 5.809923750617929e-10, 7.398023030967056e-11, 4.2152889122624615e-10, 5.172934984295373e-11, 2.020067160203011e-8]
# [-0.0074909012126365675, -3.017585090610922e-10, -2.2138044155577198e-11, -2.4739974246411057e-11, -4.3972588648062644e-11, 1.4271589241955448e-8]
# [-0.041043655732285286, 3.635034430540076e-10, -3.301130252803775e-11, -5.232306336804949e-10, -2.7341705495582088e-11, 1.968964778140048e-8]
# [-0.013304616695068208, -1.6323680603211043e-10, 5.707264173210976e-11, 8.2758263479189e-11, -1.377024494446168e-11, 5.418726687870174e-10]
# [-0.019498158069304713, 5.797067178624486e-10, -1.1160551167937471e-11, -2.924617911351988e-10, 3.970246328876975e-11, 1.836060443532476e-8]
# [0.018367836586508098, -5.305068411666227e-10, -8.484570476343561e-12, -3.0885791005884455e-10, 1.3071636714875083e-12, 1.593038622234472e-8]
# [-0.09031490828142127, 7.81319283179718e-10, 4.0823349809873876e-11, 6.660714465847812e-10, 5.7249838085465766e-12, 1.5177686387705487e-8]
#
#
# all_initial_condition
# 14-element Vector{Array}:
# [2.4445259281844485e8, 33.74952720254497, 5.139062262525559, 1.5783812811037112, 0.09531671978388118, 28.86266232140354, 0.0]
# [3.396325913932981e8, 30.375377537229326, 4.355571739860554, 1.7698160683231383, 0.9559814163465556, 150.9683637002025, 7.0]
# [1.925462156304023e8, 23.743347891022513, 3.8492033693535306, 5.3632894521199646, 1.2186240634420904, 84.9430721943757, 14.0]
# [9.283195093991194e8, 26.613358421344145, 3.4266248520010416, 7.8521202736084525, 1.3409830680862749, 207.9531264009106, 21.0]
# [5.525560045300707e8, 23.940834493503903, 2.1881037636138467, 11.065370022287514, 1.0501575765649303, 286.5860713564228, 28.0]
# [1.0079914278203977e9, 21.28118960369631, 2.066165222156412, 9.52291903850498, 2.1806230127826, 111.93247173574943, 35.0]
# [1.072369763975047e9, 15.822812349802081, 1.4824427596876795, 15.788517696985483, 2.768002186229957, 408.29436822135057, 42.0]
# [1.192771485837667e9, 17.154021575289445, 0.7169996491801401, 24.478054149833238, 3.166999239447984, 551.1680663627948, 49.0]
# [1.3621758509196446e9, 11.966234284296142, 0.056423160999699976, 28.24196074260893, 3.6288932552977715, 731.5438144618658, 56.0]
# [1.2925887664939604e9, 14.769421501658897, 0.26206272131741215, 28.012150428503496, 3.220420651596536, 864.1204111534144, 63.0]
# [9.698066291612679e8, 11.9108580462937, 0.5215165798889059, 23.89743463812208, 3.0055545524602794, 1018.966756039457, 70.0]
# [8.835645788096658e8, 12.96907822023761, 0.1514646872129411, 24.433983674985388, 2.9161937589760423, 1022.4793372740894, 77.0]
# [7.708372616848432e8, 9.617649483532638, 0.21588119419337715, 22.743242569978264, 3.1456234516865806, 1128.6298554718699, 84.0]
# [8.76601179094303e8, 12.672320414070091, 0.26469348331981074, 20.964849285696864, 3.153107484379107, 1220.3586369023133, 91.0]
