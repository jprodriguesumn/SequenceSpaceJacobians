
"""
Joao Fonseca Rodrigues May 27th, 2020
This code implements the fake news algorithm developed in Auclert et al. 2020.
It applies the algorithm to a simple Krusell-Smith type economy with idiosyncratic 
income risk and aggregate productivity shocks in the production technology.

Individual Problem: Use endogenous gridpoint method from Carroll (2006). 
Stationary distribution: Use an iterative power method described in the documentation 
    for IterativeSolvers.
Steady state: Use a bisection method on the asset market. Ensure solution by checking goods market
Transition Dynamics: Use the method described in Mittman and Krusell (2019) by taking a 
    numerical derivative of the asset market clearing condition. I then compare this with a solution
    based on Auclert et al. 2020
"""
module KS
# this is where SequenceSpace.jl file is
#push!(LOAD_PATH, "/home/joao/Dropbox/alisdair_work/MicroMacro/code")
# this is where KS file is
push!(LOAD_PATH, pwd())

export grid_fun, grid_fun2,  Solve!, EulerForward!, MakeTransMat, Steady!

using LinearAlgebra
using Parameters
using Roots
using SparseArrays
using IterativeSolvers

function uPrimeInv!(y,x,γ)
    for i in eachindex(y)
        @inbounds @fastmath y[i] = x[i]^(-1.0/γ)
    end
    y
end

function uPrimeInv(x,γ)
    x.^(-1.0/γ)
end

function uPrime!(y,x,γ)
    for i in eachindex(y)
        @inbounds @fastmath y[i] = x[i]^(-γ)
    end
end
function uPrime(x,γ)
    x.^(-γ)
end

function grid_fun(a_min,a_max,na, pexp) #this is how grid in built in Alisdair's notes (more points around 0)
    x = range(a_min,step=0.5,length=na)
    grid = a_min .+ (a_max-a_min)*(x.^pexp/maximum(x.^pexp))
    return grid
end

function grid_fun2(a_min,a_max,na) #this is how Auclert builds his grid
    grid = Real[]
    pivot = abs(a_min) + 0.25
    a1 = a_min + pivot
    an = a_max + pivot
    r = (an/a1)^(1/(na-1))
    push!(grid,a1)
    for n = 2:na
        push!(grid,a1*r^(n-1))
    end
    grid = grid .- pivot
    grid[1] = a_min
    return grid
end

function make_interp(xs::AbstractArray,
                     ys::AbstractArray) #this function could be improved upon to incorporate monotonicity
    @inline function fxn(x)
        np = searchsortedlast(xs,x)
        nend = length(xs)
        #If policy falls under undogenous grid, assign lower bound
        if np == 0
            return ys[1]
        elseif np == nend #actual grid
            xl,xh = xs[np-1],xs[np]
            yl,yh = ys[np-1],ys[np]
            return @inbounds @fastmath  yh + (x-xh)*(yh-yl)/(xh-xl)
        else
            xl,xh = xs[np],xs[np+1]
            yl,yh = ys[np],ys[np+1]
            return @inbounds @fastmath yl + (x-xl)*(yh-yl)/(xh-xl)
        end
    end
    return fxn
end

function MakeTransMat(pol::AbstractArray,
                      Pi::AbstractArray,
                      agrid::AbstractArray)
    na,ns = size(pol)
    trows = ones(Int64,2*ns*ns*na)
    tcols = ones(Int64,2*ns*ns*na)
    tvals = zeros(2*ns*ns*na)

    k = 0
    for si = 1:ns
        pol_si = pol[:,si]
        Pi_si = Pi[:,si]
        ssi = (si-1)*na
        #find where the policy falls (agrid => pol)
        for ai = 1:na
            poli = pol_si[ai]
            # find where in agrid the asset policy falls
            i = searchsortedlast(agrid,poli)
            i = min(max(i,1),na-1)
            p = (poli - agrid[i])/(agrid[i+1] - agrid[i])
            p = min(max(p,0.0),1.0)
            #for each possible state in the future
            for sj = 1:ns
                ssj = (sj-1)*na
                k += 1
                # probability if falls in the lower bin
                tvals[k] = (1.0-p) * Pi_si[sj]
                trows[k] = ssj+i
                tcols[k] = ssi+ai
                # probability if falls in the upper bin
                k += 1
                tvals[k] = p * Pi_si[sj]
                trows[k] = ssj+i+1
                tcols[k] = ssi+ai
            end
        end
    end
    return sparse(trows,tcols,tvals,na*ns,na*ns)
end

function EulerForward!(ind_pol,Va_P,Pi_P,γ,β,r,w,grids)
    agrid,sgrid = grids["agrid"],grids["sgrid"]
    na,ns = size(Va_P)
    #Va,a,c = ind_pol["V"],ind_pol["A"],ind_pol["C"]
    ### expected utility
    uc_nextgrid = β * Va_P * Pi_P

    ### implied consumption
    c_nextgrid = uPrimeInv(uc_nextgrid,γ)

    ### cash in hand
    cih = (1+r) * reshape(agrid,na,1) .+ w * reshape(sgrid,1,ns)

    ### interpolated savings policy
    for si = 1:ns
        itp = make_interp(c_nextgrid[:,si] + agrid,agrid)
        ind_pol["A"][:,si] = itp.(cih[:,si])
    end

    ### implied consumption
    ind_pol["C"] = cih - ind_pol["A"]

    ### implied value function
    ind_pol["V"] = (1+r) * uPrime(ind_pol["C"],γ)
    #Va,a,c
    #ind_pol
end

function Solve!(ind_pol,Pi_P,γ,β,r,w,grids,tol=1e-14)
    Va_P = copy(ind_pol["V"])
    ### iterate until Va = Va_P
    for i = 1:10000
        EulerForward!(ind_pol,Va_P,Pi_P,γ,β,r,w,grids)
        if (i-1) % 100 == 0
            test = abs.(ind_pol["V"] - Va_P)/(abs.(ind_pol["V"] + abs.(Va_P)))
            #println("iteration: ",i," ",maximum(test))
            if maximum(test) < tol
                #println("Solved in ",i," ","iterations")
                return ind_pol
                break
            end
        end
        Va_P = ind_pol["V"]
    end
    #println("Did not converge")
end

function StationaryDistribution(tran::SparseMatrixCSC,
                                dist::AbstractArray, #guess
                                maxit=100000,
                                tol=1e-11)
    powm!(tran, dist, maxiter = maxit,tol = tol)
    dist = dist./sum(dist)
end

function Steady!(ss,var,lb = 0.98,ub = 0.999)
    
    @inline function check_r(r,steady)
        #@show r
        @unpack grids,ind_pol,agg_pol,Pi = steady
        agrid = grids["agrid"]
        α,δ,Z,N,γ,β = agg_pol["α"],agg_pol["δ"],agg_pol["Z"],agg_pol["Lbar"],agg_pol["γ"],agg_pol["β"]
        na,ns = size(ind_pol["dist"])
        dist = reshape(ind_pol["dist"],na*ns)
        #Pi = updatePi(agg_pol)

        ### implied K and w
        K = ((r + δ)/(α*Z*N^(1-α)))^(1/(α-1))
        w = (1 - α) * Z * K^α * N^(-α)
        Y = Z * K^α * N^(1-α)

        ### update aggregates
        agg_pol["r"],agg_pol["w"],agg_pol["K"],agg_pol["Y"] = r,w,K,Y

        Solve!(steady.ind_pol,Pi,γ,β,r,w,grids)
        trans = MakeTransMat(steady.ind_pol["A"],Pi,agrid)
        dist = StationaryDistribution(trans,dist)
        d = reshape(dist,na,ns)
        A = sum(d .* reshape(agrid,na,1))
        #println("*******Asset markets*************")
        println("Assets: ",A)
        println("Capital: ",K)
        return A-K
    end

    ######### testing functions ###############
    @inline function check_beta(beta,ind_pol,Pi,K,γ,r,w,grids)
        agrid = grids["agrid"]
        na,ns = size(ind_pol["V"])
        Solve!(ind_pol,Pi,γ,beta,r,w,grids)
        trans = MakeTransMat(ind_pol["A"],Pi,agrid)

        ### distribution guess
        dist = ind_pol["dist"]
        dist = reshape(dist,na*ns)

        ### stationary distribution
        dist = StationaryDistribution(trans,dist)
        d = reshape(dist,na,ns)
        ind_pol["dist"] = d
        A = sum(d.*reshape(agrid,na,1))
        #println("*******Asset markets*************")
        println("Assets: ",A)
        println("Capital: ",K)
        return A-K
    end
    #dstar = Optim.optimize(X -> check(X,indpolicies,aggpolicies,params,params,modgrids),[K0], LBFGS(),Optim.Options(g_tol = 1e-10,iterations = 5000,show_trace = true))
    #pol = maxv(dstar.minimizer)[2]

    ### solve with roots
    #f(X) = check(X,indpolicies,aggpolicies,params,params,modgrids)
    #find_zero(f, (45, 47),Bisection(),rtol=1e-10)
    #@unpack grids,ind_pol,agg_pol = ss

    ### Check the betax
    if var == "beta"
        @show beta_min = lb / (ss.agg_pol["r"]+1)
        @show beta_max = ub / (ss.agg_pol["r"]+1)

        ### Inputs taken as fixed
        K,γ,r,w = ss.agg_pol["K"],ss.agg_pol["γ"],ss.agg_pol["r"],ss.agg_pol["w"]
        Va = ss.ind_pol["V"]
        agrid = ss.grids["agrid"]
        Pi = ss.Pi
        na,ns = size(Va)
        #dist = reshape(ss.ind_pol["dist"],na*ns)

        f(X) = check_beta(X,ss.ind_pol,Pi,K,γ,r,w,ss.grids)
        β = find_zero(f, (beta_min, beta_max),Bisection(),xtol=1e-8)
        ss.agg_pol["β"] = β
        Solve!(ss.ind_pol,Pi,γ,β,r,w,ss.grids)
        trans = MakeTransMat(ss.ind_pol["A"],ss.Pi,agrid)
        dist = StationaryDistribution(trans,dist)
    else
        @show ss.agg_pol["β"]
        @show r_min = 0.01
        @show r_max = 1/ss.agg_pol["β"] - 1
        g(X) = check_r(X,ss)
        rstar = find_zero(g, (r_min, r_max),Bisection(),xtol=1e-10)

        @unpack grids,ind_pol,agg_pol,Pi = ss
        agrid = grids["agrid"]
        α,δ,γ,w,r,β = agg_pol["α"], agg_pol["δ"],agg_pol["γ"],agg_pol["w"],agg_pol["r"],agg_pol["β"]
        na,ns = size(ind_pol["dist"])
            ### Update  ss objects not updated in iteration
        Solve!(ss.ind_pol,Pi,γ,β,r,w,ss.grids)
        trans = MakeTransMat(ss.ind_pol["A"],ss.Pi,agrid)
        dist = StationaryDistribution(trans,dist)
    end


    #### update variables 
    ss.agg_pol["C"] = sum(reshape(ss.ind_pol["C"],na*ns) .* dist)
    ss.agg_pol["A"] = sum(reshape(ss.ind_pol["A"],na*ns) .* dist)
    ss.ind_pol["dist"] = reshape(dist,na,ns)
    ss.agg_pol["X"] = ss.agg_pol["δ"] * ss.agg_pol["K"]
    @show ss.agg_pol["Y"] - ss.agg_pol["C"] - ss.agg_pol["X"]
    @assert abs(ss.agg_pol["Y"] - ss.agg_pol["C"] - ss.agg_pol["X"] ) < 1e-8
    println("steady state solved!")
    return ss
end


end
