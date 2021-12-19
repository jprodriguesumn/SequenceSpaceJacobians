"""
Joao Rodrigues, March 2021
University of Minnesota, Minneapolis
Federal Reserve Bank of Minneapolis

These are the primary functions that implements the algorithm delineated in Auclert et al (2021) to
solve for dynamics of incomplete markets heterogeneous agent models
"""

module SequenceSpace
push!(LOAD_PATH, pwd())

### types
export SimpleBlock, HetBlock, FuncObj, SteadyState

### Functions
export Perturb, AssembleImpulses, ForwIte, BackIter, FakeFunc

### Get model specific functions
#using KS: ModelGrids, SteadyState
using LinearAlgebra
using Parameters
using SparseArrays
using Pardiso
using IterativeSolvers

mutable struct SteadyState{S <: String,T <: Real}
    grids::Dict{S,Array{T,1}}
    agg_pol::Dict{S,T}
    ind_pol::Dict{S,Array{T,2}}
    Pi::Array{T,2}
end

mutable struct SimpleBlock{T <: Real,S <: String,I <: Integer}
    dF::Dict{S,Dict{S,SparseMatrixCSC{T,I}}}
    inputs::Array{S,1}
    outputs::Array{S,1}
    name::S
end

mutable struct FakeObj{T <: Real,S <: String}
    curlY::Dict{S,Dict{S,Array{T,1}}}
    curlD::Dict{S,Dict{S,Array{T,2}}}
    curlP::Dict{S,Array{T,2}}
    curlF::Dict{S,Dict{S,Array{T,2}}}
    curlJ::Dict{S,Dict{S,Array{T,2}}}
end

mutable struct HetBlock{T <: Real,S <: String}
    dF::Dict{S,Dict{S,Array{T,2}}} #Derivative of outputs wrt inputs
    inputs::Array{S,1}
    outputs::Array{S,1}
    name::S
end

function AssembleImpulses(periods,unknowns,exogenous,targets,blks...)
    """
    This function takes derivatives of outputs of partial equilibrium blocks wrt to their inputs and computes response of each output of all blocks in the economy wrt to exogenous variables
     inputs: 
        blks : tuple with partial equilibrium blocks in order
        periods: Horizon for returning to steady state
        unknowns: list of strings with variable names of unknowns
        exogenous: list of strings with variable names of exogenous variables
        targets: list of strings with target equation names (one equation in target block redundant for checkin)
    outputs:
        dH: Ouput reponses wrt unknowns (Input to get G)
        G: General equilibrium responses of all variables wrt exogenous variables
    """
    nu = length(unknowns)
    ne = length(exogenous)
    if length(targets) != nu
        throw("Number of targets must be same as unknowns")
    end
    targetblk = blks[end]

    ### this holds all exogenous variables and unknowns
    vars = union(unknowns,exogenous)

    ### Assemble outputs
    outputs = String[]
    for block in blks
        if issubset(block.inputs,union(vars,outputs)) ==  false
            name = block.name
            throw("$name block has inputs not in previous block or in set of unknowns and exogenous variables")
        end
        outputs = union(outputs,block.outputs)
    end
    #@show outputs
    ### last block in iteratble list is the target block

    # Object to store total derivatives of each output
    J = Dict{String,Dict{String,SparseMatrixCSC{Float64,Int64}}}()
    for var_o in union(vars,outputs)
        J_i = Dict{String,SparseMatrixCSC{Float64,Int64}}()
        for var_i in union(vars,outputs)
            if var_i == var_o
                J_i[var_i] = sparse(I, periods, periods)
            else
                J_i[var_i] = spzeros(periods,periods)
            end
        end
        J[var_o] = J_i
    end

    # Object to store general equilibrium derivatives
    G = Dict{String,Dict{String,Array{Float64,2}}}()
    for var_o in union(vars,outputs)
        G_i = Dict{String,Array{Float64,2}}()
        for var_i in exogenous
            G_i[var_i] = zeros(periods,periods)
        end
        G[var_o] = G_i
    end

    # Forward accumulation along ordered blocks
    for block in blks
        for var in vars
            for output in block.outputs
                mat = spzeros(periods,periods)
                for input in block.inputs
                    mat += block.dF[output][input] * J[input][var]
                end
                J[output][var] = mat
            end
        end
    end

    H_U = spzeros(periods,nu*periods)
    #for eqn in targetblk.outputs
    for eqn in targets
        H_U_i = spzeros(periods,periods)
        # concatenate across columns (unknowns)
        for un in unknowns
            if un == unknowns[1]
                H_U_i = J[eqn][unknowns[1]]
            else
                H_U_i = hcat(H_U_i,J[eqn][un])
            end
        end
        # concatenate across rows (target equations)
        #if eqn == targetblk.outputs[1]
        if eqn == targets[1]
            H_U = H_U_i
        else
            H_U = vcat(H_U,H_U_i)
        end
    end

    #H_Z = spzeros(length(targetblk.outputs)*periods,ne*periods)
    H_Z = spzeros(length(targets)*periods,ne*periods)
    #for eqn in targetblk.outputs
    for eqn in targets
        H_Z_i = spzeros(periods,periods)
        # concatenate across columns (exogenous variables)
        for ex in exogenous
            if ex == exogenous[1]
                H_Z_i = J[eqn][ex]
            else
                H_Z_i = hcat(H_Z_i,J[eqn][ex])
            end
        end
        # concatenate across rows (target equations)
        #if eqn == targetblk.outputs[1]
        if eqn == targets[1]
            H_Z = H_Z_i
        else
            H_Z = vcat(H_Z,H_Z_i)
        end
    end

    #Solve system
    ps = MKLPardisoSolver()
    G_U_Z = zeros(size(H_Z))
    solve!(ps, G_U_Z, -H_U, Matrix(H_Z))
    #@show size(H_U)
    #@show size(H_Z)
    #G_U_Z = -H_U \ Matrix(H_Z)

    #Recover response of unknowns wrt to exogenous variables into the G dict
    for z in 1:ne #number of exogenous variables
        G[exogenous[z]][exogenous[z]] = Matrix(I,periods,periods)
        for u in 1:nu # number of unknowns
            G[unknowns[u]][exogenous[z]] = G_U_Z[(u-1)*periods+1:u*periods,(z-1)*periods+1:z*periods]
        end
    end

    # Forward accumulate along general equilibrium responses
    for block in blks
        for var in exogenous
            for output in block.outputs
                mat = zeros(periods,periods)
                for input in block.inputs
                    mat += block.dF[output][input] * G[input][var]
                end
                G[output][var] = mat
            end
        end
    end

    J,G
end

function Perturb(funcs,inputs,steady,periods,eps)
    """
    Inputs:
        dF: Dict(Dict(String,Array{Float64,2})) Matrix to store derivative of output wrt inputs
        funcs: Dict(String,Function) with output name and function to differentiate
        inputs: Array{String} input names
        steady: Dict(String,FLoa64) with steady state
        periods: Integer with number of periods of impulse horizon
    Output:
        dF: filled object with derivatives
    """
    ### Get outputs
    outputs = collect(keys(funcs))

    ############################################################
    dF = Dict{String,Dict{String,SparseMatrixCSC{Float64,Int64}}}()
    for output in outputs
        dF_i = Dict{String,SparseMatrixCSC{Float64,Int64}}()
        dF[output] = dF_i
    end


    ### Create inputs variables
    inputvars = Dict{String,Dict{String,Float64}}()
    for input in inputs
        inputvars[input] = Dict("m" => steady[input], "i" => steady[input], "p" => steady[input])
    end

    ### Create a deep copy to be perturbed (copy does not work)
    vars = deepcopy(inputvars)
    ### compute derivatives of outputs wrt to inputs
    for output in outputs
        #dF_i = Dict{String,SparseMatrixCSC{Float64,Integer}}()
        for input in inputs
            ### previous period
            vars[input]["m"] = steady[input] + eps
            d_m = (funcs[output](vars) - funcs[output](inputvars))/eps
            if d_m != 0.0
                der_m = repeat([d_m],periods-1)
            end
            vars[input]["m"] = steady[input]
            ### current periods
            vars[input]["i"] = steady[input]  + eps
            d_i = (funcs[output](vars) - funcs[output](inputvars))/eps
            if d_i != 0.0
                der = repeat([d_i],periods)
            end
            vars[input]["i"] = steady[input]
            ### next periods
            vars[input]["p"] = steady[input]  + eps
            d_p = (funcs[output](vars) - funcs[output](inputvars))/eps
            if d_p != 0.0
                der_p = repeat([d_p],periods-1)
            end
            vars[input]["p"] = steady[input]
            if (d_m == 0.0) && (d_i == 0.0) && (d_p == 0.0)
                dF[output][input] = spzeros(periods,periods)

            #only one period
            elseif (d_m == 0.0) && (d_i == 0.0) && (d_p != 0.0)
                dF[output][input] = spdiagm(+1 => der_p)
            elseif (d_m == 0.0) && (d_i != 0.0) && (d_p == 0.0)
                dF[output][input] = spdiagm(0 => der)
            elseif (d_m != 0.0) && (d_i == 0.0) && (d_p == 0.0)
                dF[output][input] = spdiagm(-1 => der_m)

            #two periods
            elseif (d_m != 0.0) && (d_i != 0.0) && (d_p == 0.0)
                dF[output][input] = spdiagm(-1 => der_m,0 => der)
            elseif (d_m == 0.0) && (d_i != 0.0) && (d_p != 0.0)
                dF[output][input] = spdiagm(0 => der,+1 => der_p)
            elseif (d_m != 0.0) && (d_i == 0.0) && (d_p != 0.0)
                dF[output][input] = spdiagm(-1 => der_m,+1 => der_p)

            # all three periods
            else
                dF[output][input] = spdiagm(-1 => der_m,0 => der,+1 => der_p)
            end
        end
    end
    dF
end

function BackIte(Steady::SteadyState,
                 fake::FakeObj,
                 block::HetBlock,
                 UpdatePolicy::Function,
                 periods,
                 tol=1e-10,maxn=50,eps=0.0001)
    """
    This function uses the backward iteration described in Auclert et al. (2020) pg. 17
    Inputs:
    apol: endogenous grid => must be interpolated for individual asset position choice
    cpol: consumption
    dist: distribution over individual outcomes
    tran: Array to hold transition sparse matrix

    """
    @unpack grids,ind_pol,agg_pol,Pi = Steady
    agrid = grids["agrid"]
    @unpack inputs,outputs = block
    aggs = deepcopy(agg_pol)
    inds = deepcopy(ind_pol)

    na,ns = size(ind_pol["dist"])
    dist_ss = reshape(ind_pol["dist"],na*ns)

    # Initialize output matrices
    curlD = fake.curlD
    curlY = fake.curlY
    for output in outputs
        curlD_i = Dict{String,Array{Float64,2}}()
        curlY_i = Dict{String,Array{Float64,1}}()
        for input in inputs
            dY = zeros(periods)
            dD = zeros(na*ns,periods)
            for u = 0:periods-1
                t = periods-u-1 ### t from T-1 to 0
                if t == periods - 1
                    aggs[input] = agg_pol[input] + eps
                else
                    aggs[input] = agg_pol[input]
                end

                tran,y = UpdatePolicy(output,ind_pol,aggs,ind_pol["V"],Pi,grids)

                #aggregate output
                Y_ss = aggs[output]

                #Step 1 of Auclert's algorithm
                dD[:,u+1] = (tran * dist_ss - dist_ss)/eps
                dY[u+1] = (dot(y,dist_ss) - Y_ss)/eps
            end
            curlD_i[input] = dD
            curlY_i[input] = dY
        end
        curlD[output] = curlD_i
        curlY[output] = curlY_i
    end
    return curlD,curlY
end

function ForwIte(Steady::SteadyState,
                 fake::FakeObj,
                 block::HetBlock,
                 tran::AbstractArray,
                 periods,tol=1e-10,maxn=50,eps=0.0001)

    @unpack ind_pol,agg_pol,grids,Pi = Steady
    agrid = grids["agrid"]
    @unpack outputs = block
    na,ns = size(ind_pol["A"])

    ### Get Dict to store results
    curlP = fake.curlP

    #get the transition matrix
    #Lambda_ss = MakeTransMat(ind_pol["A"],Pi,agrid)'
    Lambda_ss = tran'
    for output in outputs
        dP = zeros(na*ns,periods-1)
        y = reshape(ind_pol[output],na*ns)
        dP[:,1] = y
        for i = 1:periods-2
            #if i == 0
            #    dP[:,i+1] = Lambda_ss * y
            #else
            dP[:,i+1] = Lambda_ss * dP[:,i]
            #end
        end
        curlP[output] = dP
    end
    curlP
end

function FakeFunc(Steady::SteadyState,
                  fake::FakeObj,
                  block::HetBlock,
                  UpdatePolicy::Function,
                  tran::AbstractArray,
                  periods,tol=1e-10,maxn=50,eps=0.0001)

    # Get lists of outputs and inputs
    @unpack inputs,outputs = block

    # step 1: Get curlyY,curlyD, and curlyP
    cD,cY = BackIte(Steady,fake,block,UpdatePolicy,periods)
    cP = ForwIte(Steady,fake,block,tran,periods)
    ni,no = length(inputs),length(outputs)

    curlF = fake.curlF
    curlJ = fake.curlJ
    # step 2: Iterate on evolution of outputs and distribution
    for output in outputs
        curlP = cP[output]
        cF_i = Dict{String,Array{Float64,2}}()
        cJ_i = Dict{String,Array{Float64,2}}()
        cF = zeros(periods,periods)
        cJ = zeros(periods,periods)
        for input in inputs
            cF = zeros(periods,periods)
            cJ = zeros(periods,periods)
            cF[1,:] = cY[output][input]
            cF[2:end,:] = cP[output]' * cD[output][input]
            cJ = cF
            for t = 2:periods
                cJ[2:end,t] += cJ[1:end-1,t-1]
            end
            cF_i[input] = cF
            cJ_i[input] = cJ
        end
        curlF[output] = cF_i
        curlJ[output] = cJ_i
    end
    fake
end

end #end module
