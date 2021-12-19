
"""
Joao Fonseca Rodrigues May 27th, 2020
This code implements the fake news algorithm developed in Auclert et al. 2020.
It applies the algorithm to a Krusell-Smith type economy with idiosyncratic 
income risk and aggregate productivity shocks in the production technology.

Individual Problem: Use endogenous gridpoint method from Carroll (2006). 
Stationary distribution: Use an iterative power method described in the documentation 
    for IterativeSolvers.
Steady state: Use a bisection method on the asset market. Ensure solution by checking goods market
Transition Dynamics: Use the method described in Mittman and Krusell (2019) by taking a 
    numerical derivative of the asset market clearing condition. I then compare this with a solution
    based on Auclert et al. 2020
"""
# this is where SequenceSpace.jl file is
push!(LOAD_PATH, "/home/joao/Dropbox/alisdair_work/MicroMacro/code")
# this is where KS file is
push!(LOAD_PATH, pwd())

using LinearAlgebra
using Parameters
using QuantEcon
using Plots

using SequenceSpace: SimpleBlock, FakeObj, HetBlock, SteadyState

using SequenceSpace: Perturb, AssembleImpulses, FakeFunc

using KS: grid_fun, grid_fun2, Solve!, EulerForward!, MakeTransMat, Steady!

function model(K0,
               Z = 1.0,
               β = 0.98,
               α = 0.11,
               δ = 0.025, #increase this in order to better compare with Auclert's code
               γ = 1.0,
               ρ = 0.966, #alisdair value was 0.95
               σz = 1.0,
               σ = 0.5, #alisdair value was 0.2
               lamw = 0.6,
               Lbar = 1.0,
               amin = 1e-16,
               amax = 200.0,
               aSize = 201,
               sSize = 2,
               endow = [1.0;2.5])

    ### Create grids to solve individual problem
    aGrid = grid_fun(amin,amax,aSize,4.0)
    aGrid = grid_fun2(amin,amax,aSize)
    aGridtiled = repeat(aGrid,1,sSize)

    ### Exogenous process
    mc = rouwenhorst(sSize, ρ, σ)
    transmat = mc.p
    endow = σ * range(-1.0,stop=1.0,length=sSize)
    dis = LinearAlgebra.eigen(transmat)
    mini = argmin(abs.(dis.values .- 1.0))
    stdist = abs.(dis.vectors[:,mini]) / sum(abs.(dis.vectors[:,mini]))
    sGrid = exp.(endow)/dot(stdist,exp.(endow))
    sGrid = reshape(sGrid,1,sSize)
    sGridtiled = repeat(sGrid,aSize,1)
    @assert sum(transmat[:,1]) == 1.0 ###sum to 1 across rows
    sGrid = reshape(sGrid,sSize)

    grids = Dict("agrid" => aGrid,"sgrid" => sGrid)

    ####### Calibration #####################
    #Going to be solving for beta
    r = 0.01

    # solve analytically what we can
    rk = r + δ
    Z = (rk / α)^α  # normalize so that Y=1
    K = (α * Z / rk) ^ (1 / (1 - α))
    Y = Z*K^α
    w = (1 - α) * Z * (α * Z / rk) ^ (α / (1 - α))
    Pr0 = [1+r,w]

    ######################################################
    # Create steady state object
    agg_pol = Dict("Y" => Y, "K" => K, "Z" => Z,"r" => 1+r,"w" => w,"β" => β,"γ" => γ,"σ" => σ,"lamw" => lamw,"α" => α,"δ" => δ,"ρ" => ρ,"Lbar" => Lbar)

    ####### Heterogenous policies
    vpol0 = 1.0 .- 0.1*aGridtiled
    ind_pol = Dict("C" => vpol0,"A" => vpol0,"V" => vpol0,"dist" => vpol0)

    ss = SteadyState(grids,agg_pol,ind_pol,transmat)
    ss = Steady!(ss)
    return ss
end

function RBC(r = 0.01,
             β = 0.98,
             σ = 1.0,
             α = 0.11,
             γ = 1.0,
             δ = 0.025,
             ρ = 0.8,
             φ = 1.0,
             periods=Periods)

    #### Steady state
    rk = r + δ
    Z = (rk / α) ^ α  # normalize so that Y=1
    K = (α * Z / rk) ^ (1 / (1 - α))
    Y = Z * K ^ α
    w = (1 - α) * Z * K ^ α
    X = δ * K
    C = Y - X
    β = 1 / (1 + r)
    φ = w*C^(-σ)
    @assert abs(C - r*K - w) < 0.000000000001 
    #params = Params(β,σ,α,γ,δ,ρ,φ)
    ss = Dict("Y" => Y, "K" => K, "Lbar" => 1.0, "C" => C, "X" => X, "r" => r, "w" => w, "Z" => Z,"β" =>β,"σ" => σ,"α" => α,"γ" => γ,"δ" => δ,"ρ" => ρ,"φ" => φ)

    ss
end

############################################################
######################### Define blocks ####################
############################################################
function HouseholdHet(inputs,outputs,steady,periods)

    dF = Dict{String,Dict{String,Array{Float64,2}}}()
    curlY = Dict{String,Dict{String,Array{Float64,1}}}()
    curlD = Dict{String,Dict{String,Array{Float64,2}}}()
    curlP = Dict{String,Array{Float64,2}}()
    curlF = Dict{String,Dict{String,Array{Float64,2}}}()
    curlJ = Dict{String,Dict{String,Array{Float64,2}}}()
    fnobj = FakeObj(curlY,curlD,curlP,curlF,curlJ)
    hetblock = HetBlock(dF,inputs,outputs,"household")

    Pi,agrid = steady.Pi,steady.grids["agrid"]
    pol = steady.ind_pol["A"]
    tran =  MakeTransMat(pol,Pi,agrid)
    #Get the derivative of outputs wrt to inputs
    FakeFunc(steady,fnobj,hetblock,UpdatePolicy,tran,periods)
    hetblock.dF = fnobj.curlJ
    return hetblock
end

function firm(inputs,steady,periods,eps,lags)
    α,δ,Lbar = steady["α"],steady["δ"],steady["Lbar"]
    #@unpack inputs, outputs,dF = firm

    #########################################################
    ################ Define block output functions##########
    # This is the only portion that needs to be changed in each block
    if issubset(["K","Z"],inputs) == false
        throw("inputs used not a subset of inputs defined in constructor")
    end

    function f1(var)
        K,Z = var["K"],var["Z"]
        out1 = Z["i"]*K["m"]^α*Lbar^(1-α)
        return out1
    end
    function f2(var)
        K,Z = var["K"],var["Z"]
        out2 = (1-α)*Z["i"]*K["m"]^α*Lbar^(-α)
        return out2
    end
    function f3(var)
        K,Z = var["K"],var["Z"]
        out3 = α*Z["i"]*K["m"]^(α-1)*Lbar^(1-α) - δ
        return out3
    end
    funcs = Dict{String,Function}()
    funcs["Y"],funcs["w"],funcs["r"] = f1, f2, f3
    ############################################################

    dF = Perturb(funcs,inputs,steady,periods,eps)
    SimpleBlock(dF,inputs,collect(keys(funcs)),"firm")
end

function targetHet(inputs,steady,periods,eps,lags)
    
    #########################################################
    ################ Define block output functions##########
    # This is the only portion that needs to be changed in each block
    if issubset(["A","K"],inputs) == false
        throw("inputs used not a subset of inputs defined in constructor")
    end
    function f1(var)
        K,A = var["K"],var["A"]
        out1 = A["i"] - K["i"] 
        return out1
    end
    funcs = Dict{String,Function}()
    funcs["mc"] = f1
    ############################################################

    dF = Perturb(funcs,inputs,steady,periods,eps)
    SimpleBlock(dF,inputs,collect(keys(funcs)),"target")
end

function UpdatePolicy(output,ind_pol,agg_pol,Va_P,Pi,grids)
    γ,β,r,w = agg_pol["γ"],agg_pol["β"],agg_pol["r"],agg_pol["w"]
    na,ns = size(Va_P)
    EulerForward!(ind_pol,Va_P,Pi,γ,β,r,w,grids)
    tran = MakeTransMat(ind_pol["A"],Pi,grids["agrid"])
    y = reshape(ind_pol[output],na*ns)
    return tran,y
end

######################## Representative specific functions ####
function targetRep(inputs,steady,periods,eps,lags)
    
    β,σ = steady["β"],steady["σ"]

    #########################################################
    ################ Define block output functions##########
    #if issubset(["C","r","Y","X"],inputs) == false
    #    throw("inputs used not a subset of inputs defined in constructor")
    #end
    # This is the only portion that needs to be changed in each block
    function f1(var)
        Y,X,C,r = var["Y"],var["X"],var["C"],var["r"]
        out1 = C["i"]^(-σ) - β*(1+r["p"]) * C["p"]^(-σ) 
        return out1
    end
    function f2(var)
        Y,X,C,r = var["C"],var["Y"],var["X"],var["r"]
        out2 = Y["i"] - C["i"] - X["i"]
        return out2
    end
    funcs = Dict{String,Function}()
    funcs["euler"],funcs["mc"] = f1, f2
    #funcs["euler"] = f1
    #funcs["mc"] = f2
    ############################################################

    dF = Perturb(funcs,inputs,steady,periods,eps)
    SimpleBlock(dF,inputs,collect(keys(funcs)),"target")
end

function householdRep(inputs,steady,
                      periods,eps,lags)
    γ,σ,δ,φ,Lbar = steady["γ"],steady["σ"],steady["δ"],steady["φ"],steady["Lbar"]

    #########################################################
    ################ Define block output functions##########
    if issubset(["K","w","r"],inputs) == false
        throw("inputs used not a subset of inputs defined in constructor")
    end
    # This is the only portion that needs to be changed in each block
    function f1(var)
        K,w,r = var["K"],var["w"],var["r"]
        out = K["i"] - (1-δ)*K["m"]
        return out
    end
    function f2(var)
        K,w,r = var["K"],var["w"],var["r"]
        out = (1+r["i"])*K["m"] + w["i"] * Lbar - K["i"]
        return out
    end
    funcs = Dict{String,Function}()
    funcs["X"],funcs["C"] = f1,f2
    ############################################################

    dF = Perturb(funcs,inputs,steady,periods,eps)
    SimpleBlock(dF,inputs,collect(keys(funcs)),"household")
end

Periods = 200
eps0,lags0 = 0.0001,["m","i","p"]
unknowns = ["K"]
exogenous = ["Z"]
target = ["mc"]

### Get Steady State parameters individual and aggregate policies
stead = model(46.5)

J_H = HouseholdHet(["r","w"],["A","C"],stead,Periods)
J_F = firm(["K","Z"],stead.agg_pol,Periods,eps0,lags0)
J_T = targetHet(["A","K"],stead.agg_pol,Periods,eps0,lags0)
J,G = AssembleImpulses(Periods,unknowns,exogenous,target,J_F,J_H,J_T)
ss = stead.agg_pol
impact, rho, news = 0.01, 0.8, 10
dZ = zeros(Periods,2)
dZ[1,1] = impact
#dZ[:,1] = IRF[8,:]
dZ[:,2] = vcat(zeros(news),dZ[1:end-news,1])
dr = G["r"]["Z"] * dZ/ (ss["r"] - 1)
dw = G["w"]["Z"] * dZ/ ss["w"]
dK = G["K"]["Z"] * dZ/ ss["K"]
dC = G["C"]["Z"] * dZ/ ss["C"]
steadRBC=RBC()
unk,exo = ["K"], ["Z"]
J_Hrep = householdRep(["K","w","r"],steadRBC,Periods,eps0,lags0)
J_Frep = firm(["K","Z"],steadRBC,Periods,eps0,lags0)
J_Trep = targetRep(["Y","X","C","r"],steadRBC,Periods,eps0,lags0)
Jrep,Grep = AssembleImpulses(Periods,unk,exo,["euler"],J_Frep,J_Hrep,J_Trep)
drR = Grep["r"]["Z"] * dZ/ steadRBC["r"]
dwR = Grep["w"]["Z"] * dZ/ steadRBC["w"]
dKR = Grep["K"]["Z"] * dZ/ steadRBC["K"]
dCR = Grep["C"]["Z"] * dZ/ steadRBC["C"]

time = 50
p1 = plot(title="C")
p1 = plot!(dCR[1:time,1],label="Representative Agents",line=:solid,color=:black)
p1 = plot!(dC[1:time,1],label="Heterogeneous Agents",line=:dash,color=:black)

p2 = plot(title="K")
p2 = plot!(dKR[1:time,1],label="Representative Agents",line=:solid,color=:black)
p2 = plot!(dK[1:time,1],label="Heterogeneous Agents",line=:dash,color=:black)

p3 = plot(title="r")
p3 = plot!(drR[1:time,1],label="Representative Agents",line=:solid,color=:black)
p3 = plot!(dr[1:time,1],label="Heterogeneous Agents",line=:dash,color=:black)

p4 = plot(title="w")
p4 = plot!(dwR[1:time,1],label="Representative Agents",line=:solid,color=:black)
p4 = plot!(dw[1:time,1],label="Heterogeneous Agents",line=:dash,color=:black)

p = plot(p1,p2,p3,p4,layout=(2,2))
savefig(p,"rbcirf.pdf")

