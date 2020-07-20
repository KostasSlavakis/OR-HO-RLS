#############################################################################
using LinearAlgebra
using SparseArrays
using Compat
using Random
using Distributions
using Gurobi
using JuMP
using Printf
using Plots
using DelimitedFiles

# Play with the parameters marked by <--

###################################################
# Problem parameters:
###################################################
NoExp = 1; # <--
NoIter = 1000; # <--
P = 20;
L = 10;
SystemIsSparse = "no";
SystemSparsityLevel = 10/100;
SNR = 20.0; # In dB. The variance of the input signal is equal to 1. # <--
VarOutliers = 1.2e4; # <--
ARcoeffU = 0.95;
ARcoeffL = 0.0;
pBernoulli = 0.2; # <--
K0 = 500; # <--
#
Radius = 20;
gamma = 1;
#
alpha = 0.5;
lambda = 1e-1; # <--
UseBatchMethod = "FMHSDM"; # Choices are: ADMM, GIST, FMHSDM. # <--
alphaFMHSDM = 0.5;
lambdao = 1e-1; # <--
lambdaFMHSDM = 150; # <--
JFMHSDM = 100; # <--
eVarPi = 5e-2;
lambdaF = 1e-5;
rho = 1e2;
MultiplyGradBy = 1;
#
epsilonORRLS = 1e-1; # <--
lambdaORRLSl1 = 1e2; # <--
rhoADMM = 1e2; # <--
JADMM = 100; # <--
# Parameters of the GIST algorithm:
thetaMCP = 2.0; # <--
lambdaMCP = 1e2; # <--
tminMCP = 1e-30;
tmaxMCP = 1e30;
etaMCP = 2.0;
sigmaMCP = 1e-5;
mBuffMCP = 5;
JGIST = 100; # <--

#############################################################################
# Functions:
#############################################################################

struct GenerateProblemParams
    NoIter::Int64
    P::Int64
    L::Int64
    SystemIsSparse::String
    #
    Input::Array{Float64,2}
    System::Array{Float64,2}
    Output::Array{Float64,2}
    LastInput0::Array{Float64,1}
    LastOutput0::Array{Float64,1}
    Rnoise::Array{Float64,2}
    invRnoise::Array{Float64,2}
    MaxEigInvRnoise::Float64
    #
    eVarPi::Float64
    alpha::Float64
    Radius::Float64
    gamma::Float64
    epsilonORRLS::Float64
    F0::Array{Float64,2}
    o0::Array{Float64,1}
    F0HRLS::Array{Float64,2}
    o0HRLS::Array{Float64,1}
    lambdaF::Float64
    lambdao::Float64
    rho::Float64
    lambda::Float64
    MultiplyGradBy::Float64
    lambdaFMHSDM::Float64
    alphaFMHSDM::Float64
    JFMHSDM::Int64
    UseBatchMethod::String
    lambdaORRLSl1::Float64
    rhoADMM::Float64
    JADMM::Int64
    thetaMCP::Float64
    lambdaMCP::Float64
    tminMCP::Float64
    tmaxMCP::Float64
    etaMCP::Float64
    sigmaMCP::Float64
    mBuffMCP::Int64
    JGIST::Int64
end

#############################################################################

function GenerateModelMatrices(L::Int64, ARcoeffU::Float64, ARcoeffL::Float64, SNR::Float64, SystemIsSparse::String, SystemSparsityLevel::Float64)

    if SystemIsSparse == "yes"
        SystemVec = zeros(P*L);
        NumberNonZeroEntries = ceil(SystemSparsityLevel*P*L);
        NumberNonZeroEntries = convert(Int64,NumberNonZeroEntries);
        SystemVec[1:NumberNonZeroEntries] = ones(NumberNonZeroEntries);
        SystemVec = shuffle(SystemVec);
        System = reshape(SystemVec, (L,P));
    else
        System = randn(Float64,(L,P));
    end

    FAuxMat =  svd(randn(L,L));
    singularVal = [ARcoeffL*ones(L-1) + (ARcoeffU-ARcoeffL)*rand(L-1); ARcoeffU];
    ARmatrix = FAuxMat.U*Diagonal(singularVal)*FAuxMat.Vt;

    AuxVec = inv(Matrix{Float64}(I,L*L,L*L) - kron(ARmatrix, ARmatrix))*vec(Matrix{Float64}(I,L,L));
    AuxMat = reshape(AuxVec, (L,L));
    AuxMat = Symmetric((AuxMat + AuxMat')/2);
    VarWhiteNoise = 10^(-SNR/10)/opnorm(AuxMat,2);
    # VarWhiteNoise = (1-ARcoeffU^2)*10^(-SNR/10);

    Rnoise = VarWhiteNoise*AuxMat;
    invRnoise = Symmetric(inv(Rnoise));
    MaxEigRnoise = opnorm(Rnoise, 2);
    MaxEigInvRnoise = opnorm(invRnoise, 2);

    return System, Rnoise, invRnoise, MaxEigInvRnoise, VarWhiteNoise, ARmatrix;

end

#############################################################################

function InputOutput(P::Int64, L::Int64, NoIter::Int64, System::Matrix{Float64}, ARmatrix::Matrix{Float64}, pBernoulli::Float64, VarOutliers::Float64, VarWhiteNoise::Float64, FirstNoiseVec::Vector{Float64})

    Input = randn(Float64,(P,NoIter+1));

    # Coloring noise:
    Noise = Matrix{Float64}(undef,(L,NoIter+1));
    Noise[:,1] = FirstNoiseVec;
    for n in 1:NoIter
        Noise[:,n+1] = ARmatrix*Noise[:,n] + sqrt(VarWhiteNoise)*randn(Float64,L);
    end

    MaskOutliersPDF = Binomial(1,pBernoulli);
    MaskOutliers = rand(MaskOutliersPDF,L,NoIter+1);
    ModulusOutliers = sqrt(3*VarOutliers);
    Outliers = -ModulusOutliers*ones(L,NoIter+1) + 2*ModulusOutliers*rand(L,NoIter+1);
    Outliers = Outliers.*MaskOutliers;

    Output = System*Input + Outliers + Noise;

    return Input, Output, Noise[:,NoIter+1];
end

#############################################################################

function WarmStart(P, L, K0, Factor, invRnoise, lambda, Input, Output)

    MDL = Model(with_optimizer(Gurobi.Optimizer, PSDTol = 1e0, OutputFlag = 0, BarHomogeneous = 1, NumericFocus = 3));

    @variable(MDL, op[l=1:L, k=1:K0]);
    @variable(MDL, on[l=1:L, k=1:K0]);
    @variable(MDL, F[l=1:L, p=1:P]);

    @constraint(MDL, op .>= 0.0);
    @constraint(MDL, on .>= 0.0);

    AuxMat = F*Input + op - on - Output;
    OBJa = sum(AuxMat[:,k]'*invRnoise*AuxMat[:,k] for k=1:K0)/Factor;
    OBJb = lambda*sum(op+on);

    @objective(MDL, Min, OBJa+OBJb);
    optimize!(MDL);

    Fhat = value.(F);
    ohat = value.(op-on);

    # env = Gurobi.Env();
    # M = Model(solver = GurobiSolver(env, OutputFlag=0));

    # @variable(M, o[l=1:L, k=1:K0]);
    # @variable(M, op[l=1:L, k=1:K0]);
    # @variable(M, on[l=1:L, k=1:K0]);
    # @variable(M, z[l=1:L, k=1:K0]);
    # @variable(M, F[l=1:L, p=1:P]);

    # @constraint(M, z .== op + on);
    # @constraint(M, o .== op - on);
    # @constraint(M, op .>= 0);
    # @constraint(M, on .>= 0);

    # @objective(M, Min, sum(sum((Output[l,k] - sum(F[l,p]*Input[p,k] for p=1:P) - o[l,k])*sum(invRnoise[l,j]*(Output[j,k] - sum(F[j,p]*Input[p,k] for p=1:P) - o[j,k])/2 for j=1:L) for l=1:L) for k=1:K0) + lambda*sum(sum(z[l,k] for l=1:L) for k=1:K0));

    # solve(M);

    # Fhat = getvalue(F);
    # ohat = getvalue(o);

    return Fhat, ohat[:,K0];

end

#############################################################################

function UBoundLambdaORRLSl1(P, L, K0, invRnoise, Input, Output)

    Lambdas = Vector{Float64}(undef, K0);

    for k in 1:K0
        F = Output[:,k]*Input[:,k]'*pinv(Input[:,k]*Input[:,k]');
        Dummy = invRnoise*(Output[:,k] - F*Input[:,k]);
        Lambdas[k] = norm(Dummy, Inf);
    end

    return norm(Lambdas, Inf);

end

#############################################################################

function Proxl2(X::Array{Float64}, gamma::Float64)
    return X/(1+gamma);
end

#############################################################################

function Projl2(X::Matrix{Float64}, Radius::Float64)
    return X*Radius/max(Radius, norm(X,2));
end

#############################################################################

function Proxl1(X::Array{Float64}, gamma::Float64)
    return X .* (1 .- gamma ./ max.(gamma, abs.(X)));
end

#############################################################################

function Proxl1scalar(x::Float64, gamma::Float64)
    return x*(1 - gamma/max(gamma, abs(x)));
end

#############################################################################

function MCPloss(w::Float64, lambda::Float64, theta::Float64)

    if abs(w) <= theta*lambda
        loss = lambda*abs(w) - (w^2)/theta/2;
    else
        loss = theta*(lambda^2)/2;
    end

    return loss;

end

#############################################################################

function ProxMCP(u::Float64, lambda::Float64, t::Float64, theta::Float64)

    if (t*theta != 1)
        w1 = t*theta*(u - lambda/t)/(t*theta-1);
        w2 = t*theta*(u + lambda/t)/(t*theta-1);
        C = [0; w1; w2; u];
        Aux = Vector{Float64}(undef,length(C));
        for i in 1:(length(C)-1)
            Aux[i] = .5*(C[i]-u)^2 + lambda*abs(C[i])/t - ((C[i])^2)/2/theta/t;
        end
        Aux[length(C)] = theta*(lambda^2)/2/t;
        iStar = argmin(Aux);
        prox = C[iStar];
    else
        C = [0; theta*lambda; -theta*lambda; u];
        Aux = Vector{Float64}(undef,length(C));
        for i in 1:(length(C)-1)
            Aux[i] = - C[i]*u + lambda*abs(C[i])/t;
        end
        Aux[length(C)] = theta*(lambda^2)/2/t;
        iStar = argmin(Aux);
        prox = C[iStar];
    end

    return prox;

end

#############################################################################
# function sthresh_cd_MCP(x, l, R, theta)
#     if abs(x) <= theta*l
#         (R*theta/(R*theta-1))*sign(x)*max((abs(x)-l/R),0);
#     else
#         x;
#     end
# end
#############################################################################

function FMHSDMforLASSO(Resolvent::Matrix{Float64}, TargetVec::Array{Float64}, lambdao::Float64, lambda::Float64, alpha::Float64, JFMHSDM::Int64, StartingPoint::Array{Float64})

    X = repeat(StartingPoint, inner = [1,2]);
    Proj = sum(X, dims=2)/2;
    Tmap = repeat(Proj, inner = [1,2]);
    Xhalf = alpha*Tmap + (1-alpha)*X;
    Xprev = X;
    TmapPrev = Tmap;
    X[:,1] = Resolvent*(TargetVec + Xhalf[:,1]/lambda);
    X[:,2] = Proxl1(Xhalf[:,2], lambda*lambdao);

    for j in 1:JFMHSDM

        Proj = sum(X, dims=2)/2;
        Tmap = repeat(Proj, inner = [1,2]);
        Xhalf = Xhalf - alpha*TmapPrev - (1-alpha)*Xprev + Tmap;
        Xprev = X;
        TmapPrev = Tmap;
        X[:,1] = Resolvent*(TargetVec + Xhalf[:,1]/lambda);
        X[:,2] = Proxl1(Xhalf[:,2], lambda*lambdao);

    end

    return sum(X, dims=2)/2;

end

#############################################################################

function ADMM(b::Vector{Float64}, Resolvent::Matrix{Float64}, lambda::Float64, rho::Float64, J::Int64, x0::Vector{Float64})

    L = size(x0,1);
    x = x0;
    u = x0;
    z = x0;

    for j in 1:J
        x = Resolvent*(b + rho*(z-u));
        z = Proxl1(x+u, lambda/rho);
        u = u + x - z;
    end

    return x;

end

#############################################################################

function CD(R_inv, Dy, lambda, alpha, J, o0)

    o = Matrix{Float64}(undef, (L,J+1));
    o[:,1] = o0;

    for j in 1:J
        for d in 1:L
            R_inv_d1 = 0;
            R_inv_d2 = 0;
            for d1 in 1:(d-1)
                R_inv_d1 = R_inv_d1 + R_inv[d1,d]*o[d1,j+1];
            end
            for d2 in (d+1):L
                R_inv_d2 = R_inv_d2 + R_inv[d2,d]*o[d2,j];
            end
            gamma = (1/R_inv[d,d])*(alpha[d] - R_inv_d1 - R_inv_d2);
            o[d,j+1] = Proxl1scalar(gamma, lambda/R_inv[d,d]);
        end
    end
    return o[:,J+1];

end

#############################################################################

function CD_MCP(R_inv, Dy, lambda, alpha, theta, J, o0)

    o = Matrix{Float64}(undef, (L,J+1));
    o[:,1] = o0;

    for j in 1:J
        for d in 1:L
            R_inv_d1 = 0;
            R_inv_d2 = 0;
            for d1 in 1:(d-1)
                R_inv_d1 = R_inv_d1 + R_inv[d1,d]*o[d1,j+1];
            end
            for d2 in (d+1):L
                R_inv_d2 = R_inv_d2 + R_inv[d2,d]*o[d2,j];
            end
            gamma = (1/R_inv[d,d])*(alpha[d] - R_inv_d1 - R_inv_d2);

            # o[d,j+1] = sthresh_cd_MCP(gamma, lambda, R_inv[d,d], theta);
            o[d,j+1] = ProxMCP(gamma, lambda, R_inv[d,d], theta);

        end
    end
    return o[:,J+1];

end

#############################################################################

function GIST(w0::Vector{Float64}, L::Int64, b::Vector{Float64}, invRnoise::Matrix{Float64}, lambda::Float64, theta::Float64, tmin::Float64, tmax::Float64, eta::Float64, sigma::Float64, mBuff::Int64, JGIST::Int64)

    w = w0;
    wPrev = zeros(L);
    MaxFbuff = zeros(mBuff);
    Regularizer = 0;
    for l in 1:L
        Regularizer = Regularizer + MCPloss(w[l], lambda, theta);
    end
    MaxFbuff[mBuff] = (w-b)'*invRnoise*(w-b)/2 + Regularizer;
    MaxF = maximum(MaxFbuff);

    for k in 1:JGIST

        # Sec. 2.3.1:
        x = w-wPrev;
        y = invRnoise*x;
        if norm(x) == 0
            t = 1.0;
        else
            # Barzilai-Borwein rule:
            t = x'*y/(x'*x);
            t = max(min(t,tmax), tmin);
        end
        wPrev = w;

        Criterion = 1;
        tau = t;
        wNext = Vector{Float64}(undef,L);
        CurrValF = 0;
        while (Criterion > 0) && (tau < tmax)
            u = w - invRnoise*(w-b)/tau;
            Regularizer = 0;
            for l in 1:L
                wNext[l] = ProxMCP(u[l], lambda, tau, theta);
                Regularizer = Regularizer + MCPloss(wNext[l], lambda, theta);
            end
            CurrValF = (wNext-b)'*invRnoise*(wNext-b)/2 + Regularizer;
            Criterion = CurrValF - MaxF + sigma*tau*(norm(wNext-w)^2)/2;
            tau = eta*tau;
        end
        w = wNext;
        MaxFbuff[1:(mBuff-1)] = MaxFbuff[2:mBuff];
        MaxFbuff[mBuff] = CurrValF;
        MaxF = maximum(MaxFbuff);

    end

    return w;

end

#############################################################################

function RLS(ProblemParams)

    L = ProblemParams.L;
    NoIter = ProblemParams.NoIter;
    Input = ProblemParams.Input;
    Output = ProblemParams.Output;
    Rnoise = ProblemParams.Rnoise;
    invRnoise = ProblemParams.invRnoise;
    epsilonORRLS = ProblemParams.epsilonORRLS;
    System = ProblemParams.System;
    F0 = ProblemParams.F0;
    gamma = ProblemParams.gamma;

    Phi = Matrix{Float64}(I,L*P,L*P)/epsilonORRLS;
    Phi = Symmetric(Phi);

    NRMSE = Vector{Float64}(undef,NoIter);

    # F = F0;
    # F = randn(L,P);
    Fvec = vec(F0);

    BEGIN = time();
    for k in 2:(NoIter+1)

        H = kron(Input[:,k]', Matrix{Float64}(I,L,L));

        # RLS updates:
        DummyMat = H*Phi*H'/gamma;
        DummyMat = (DummyMat + DummyMat')/2;
        DummyMat = Symmetric(DummyMat);
        W = (1/gamma)*Phi*H'*inv(DummyMat + Rnoise);
        Fvec = Fvec + W*(Output[:,k] - H*Fvec);
        Phi = (1/gamma)*(Phi - W*H*Phi);
        Phi = (Phi+Phi')/2;
        Phi = Symmetric(Phi);

        F = reshape(Fvec,L,P);
        NRMSE[k-1] = norm(F-System,2)/norm(System,2);

    end
    END = time();
    Time = (END-BEGIN)/NoIter;

    return NRMSE, Time;

end

#############################################################################

function ORRLSl1(ProblemParams)

    L = ProblemParams.L;
    P = ProblemParams.P;
    NoIter = ProblemParams.NoIter;
    Input = ProblemParams.Input;
    Output = ProblemParams.Output;
    lambdaORRLSl1 = ProblemParams.lambdaORRLSl1;
    Rnoise = ProblemParams.Rnoise;
    invRnoise = ProblemParams.invRnoise;
    rhoADMM = ProblemParams.rhoADMM;
    JADMM = ProblemParams.JADMM;
    epsilonORRLS = ProblemParams.epsilonORRLS;
    System = ProblemParams.System;
    F0 = ProblemParams.F0;
    o0 = ProblemParams.o0;

    NRMSE = Vector{Float64}(undef,NoIter);

    Fvec = vec(F0);
    # F = randn(L,P);
    o = o0;
    Resolvent = inv(invRnoise + rhoADMM*Matrix{Float64}(I,L,L));
    Phi = Matrix{Float64}(I,L*P,L*P)/epsilonORRLS;
    SystemVec = vec(System);

    BEGIN = time();
    for k in 2:(NoIter+1)

        H = kron(Input[:,k]', Matrix{Float64}(I,L,L));

        auxVec = invRnoise*(Output[:,k] - H*Fvec);
        o = ADMM(auxVec, Resolvent, lambdaORRLSl1, rhoADMM, JADMM, o);

        # RLS updates:
        DummyMat = H*Phi*H'/gamma;
        DummyMat = (DummyMat + DummyMat')/2;
        DummyMat = Symmetric(DummyMat);
        W = (1/gamma)*Phi*H'*inv(DummyMat + Rnoise);
        Fvec = Fvec + W*(Output[:,k] - o - H*Fvec);
        Phi = (1/gamma)*(Phi - W*H*Phi);
        Phi = (Phi+Phi')/2;
        Phi = Symmetric(Phi);

        NRMSE[k-1] = norm(Fvec-SystemVec)/norm(SystemVec);

    end
    END = time();
    Time = (END-BEGIN)/NoIter;

    return NRMSE, Time;

end

#############################################################################

function ORRLSCD_L1(ProblemParams)

    L = ProblemParams.L;
    P = ProblemParams.P;
    NoIter = ProblemParams.NoIter;
    Input = ProblemParams.Input;
    Output = ProblemParams.Output;
    lambdaORRLSl1 = ProblemParams.lambdaORRLSl1;
    Rnoise = ProblemParams.Rnoise;
    invRnoise = ProblemParams.invRnoise;
    epsilonORRLS = ProblemParams.epsilonORRLS;
    System = ProblemParams.System;
    F0 = ProblemParams.F0;
    o0 = ProblemParams.o0;
    thetaMCP = ProblemParams.thetaMCP;
    lambdaMCP = ProblemParams.lambdaMCP;
    tminMCP = ProblemParams.tminMCP;
    tmaxMCP = ProblemParams.tmaxMCP;
    etaMCP = ProblemParams.etaMCP;
    sigmaMCP = ProblemParams.sigmaMCP;
    mBuffMCP = ProblemParams.mBuffMCP;
    JGIST = ProblemParams.JGIST;

    NRMSE = Vector{Float64}(undef,NoIter);

    Fvec = vec(F0);
    # F = randn(L,P);
    o = o0;
    Phi = Matrix{Float64}(I,L*P,L*P)/epsilonORRLS;


    SystemVec = vec(System);

    BEGIN = time();
    for k in 2:(NoIter+1)

        H = kron(Input[:,k]', Matrix{Float64}(I,L,L));

        auxVec = invRnoise*(Output[:,k] - H*Fvec);
        o = CD(invRnoise, L, lambdaORRLSl1, auxVec, JADMM, o);

        # RLS updates:
        DummyMat = H*Phi*H'/gamma;
        DummyMat = (DummyMat + DummyMat')/2;
        DummyMat = Symmetric(DummyMat);
        W = (1/gamma)*Phi*H'*inv(DummyMat + Rnoise);
        Fvec = Fvec + W*(Output[:,k] - o - H*Fvec);
        Phi = (1/gamma)*(Phi - W*H*Phi);
        Phi = (Phi+Phi')/2;
        Phi = Symmetric(Phi);

        NRMSE[k-1] = norm(Fvec-SystemVec)/norm(SystemVec);

    end
    END = time();
    Time = (END-BEGIN)/NoIter;

    return NRMSE, Time;
end

#############################################################################

function ORRLSCD_MCP(ProblemParams)

    L = ProblemParams.L;
    P = ProblemParams.P;
    NoIter = ProblemParams.NoIter;
    Input = ProblemParams.Input;
    Output = ProblemParams.Output;
    lambdaORRLSl1 = ProblemParams.lambdaORRLSl1;
    Rnoise = ProblemParams.Rnoise;
    invRnoise = ProblemParams.invRnoise;
    epsilonORRLS = ProblemParams.epsilonORRLS;
    System = ProblemParams.System;
    F0 = ProblemParams.F0;
    o0 = ProblemParams.o0;
    thetaMCP = ProblemParams.thetaMCP;
    lambdaMCP = ProblemParams.lambdaMCP;
    tminMCP = ProblemParams.tminMCP;
    tmaxMCP = ProblemParams.tmaxMCP;
    etaMCP = ProblemParams.etaMCP;
    sigmaMCP = ProblemParams.sigmaMCP;
    mBuffMCP = ProblemParams.mBuffMCP;
    JGIST = ProblemParams.JGIST;

    NRMSE = Vector{Float64}(undef,NoIter);

    Fvec = vec(F0);
    # F = randn(L,P);
    o = o0;
    Resolvent = inv(invRnoise + rhoADMM*Matrix{Float64}(I,L,L));
    Phi = Matrix{Float64}(I,L*P,L*P)/epsilonORRLS;
    SystemVec = vec(System);

    BEGIN = time();
    for k in 2:(NoIter+1)

        H = kron(Input[:,k]', Matrix{Float64}(I,L,L));

        auxVec = invRnoise*(Output[:,k] - H*Fvec);
        o = CD_MCP(invRnoise, L, lambdaMCP, auxVec, thetaMCP, JGIST, o);

        # RLS updates:
        DummyMat = H*Phi*H'/gamma;
        DummyMat = (DummyMat + DummyMat')/2;
        DummyMat = Symmetric(DummyMat);
        W = (1/gamma)*Phi*H'*inv(DummyMat + Rnoise);
        Fvec = Fvec + W*(Output[:,k] - o - H*Fvec);
        Phi = (1/gamma)*(Phi - W*H*Phi);
        Phi = (Phi+Phi')/2;
        Phi = Symmetric(Phi);

        NRMSE[k-1] = norm(Fvec-SystemVec)/norm(SystemVec);

    end
    END = time();
    Time = (END-BEGIN)/NoIter;

    return NRMSE, Time;
end

#############################################################################

function ORRLSMCP(ProblemParams)

    L = ProblemParams.L;
    P = ProblemParams.P;
    NoIter = ProblemParams.NoIter;
    Input = ProblemParams.Input;
    Output = ProblemParams.Output;
    lambdaORRLSl1 = ProblemParams.lambdaORRLSl1;
    Rnoise = ProblemParams.Rnoise;
    invRnoise = ProblemParams.invRnoise;
    epsilonORRLS = ProblemParams.epsilonORRLS;
    System = ProblemParams.System;
    F0 = ProblemParams.F0;
    o0 = ProblemParams.o0;
    thetaMCP = ProblemParams.thetaMCP;
    lambdaMCP = ProblemParams.lambdaMCP;
    tminMCP = ProblemParams.tminMCP;
    tmaxMCP = ProblemParams.tmaxMCP;
    etaMCP = ProblemParams.etaMCP;
    sigmaMCP = ProblemParams.sigmaMCP;
    mBuffMCP = ProblemParams.mBuffMCP;
    JGIST = ProblemParams.JGIST;

    NRMSE = Vector{Float64}(undef,NoIter);

    Fvec = vec(F0);
    # F = randn(L,P);
    o = o0;
    Phi = Matrix{Float64}(I,L*P,L*P)/epsilonORRLS;
    SystemVec = vec(System);

    BEGIN = time();
    for k in 2:(NoIter+1)

        H = kron(Input[:,k]', Matrix{Float64}(I,L,L));

        auxVec = Output[:,k] - H*Fvec;
        o = GIST(o, L, auxVec, invRnoise, lambdaMCP, thetaMCP, tminMCP, tmaxMCP, etaMCP, sigmaMCP, mBuffMCP, JGIST);

        # RLS updates:
        DummyMat = H*Phi*H'/gamma;
        DummyMat = (DummyMat + DummyMat')/2;
        DummyMat = Symmetric(DummyMat);
        W = (1/gamma)*Phi*H'*inv(DummyMat + Rnoise);
        Fvec = Fvec + W*(Output[:,k] - o - H*Fvec);
        Phi = (1/gamma)*(Phi - W*H*Phi);
        Phi = (Phi+Phi')/2;
        Phi = Symmetric(Phi);

        NRMSE[k-1] = norm(Fvec-SystemVec)/norm(SystemVec);

    end
    END = time();
    Time = (END-BEGIN)/NoIter;

    return NRMSE, Time;

end

#############################################################################
function HRLS_ADMM(ProblemParams)

    ###################################################
    # Parameters:
    ###################################################

    eVarPi = ProblemParams.eVarPi;
    alpha = ProblemParams.alpha;
    lambda = ProblemParams.lambda;
    MultiplyGradBy = ProblemParams.MultiplyGradBy;

    NoIter = ProblemParams.NoIter;
    P = ProblemParams.P;
    L = ProblemParams.L;
    lambdaF = ProblemParams.lambdaF;
    Radius = ProblemParams.Radius;
    Input = ProblemParams.Input;
    Output = ProblemParams.Output;
    LastInput0 = ProblemParams.LastInput0;
    LastOutput0 = ProblemParams.LastOutput0;
    MaxEigInvRnoise = ProblemParams.MaxEigInvRnoise;
    invRnoise = ProblemParams.invRnoise;
    System = ProblemParams.System;
    F0 = ProblemParams.F0HRLS;
    o0 = ProblemParams.o0HRLS;
    lambdao = ProblemParams.lambdao;
    rho = ProblemParams.rho;
    lambdaFMHSDM = ProblemParams.lambdaFMHSDM;
    alphaFMHSDM = ProblemParams.alphaFMHSDM;
    JFMHSDM = ProblemParams.JFMHSDM;
    thetaMCP = ProblemParams.thetaMCP;
    lambdaMCP = ProblemParams.lambdaMCP;
    tminMCP = ProblemParams.tminMCP;
    tmaxMCP = ProblemParams.tmaxMCP;
    etaMCP = ProblemParams.etaMCP;
    sigmaMCP = ProblemParams.sigmaMCP;
    mBuffMCP = ProblemParams.mBuffMCP;
    JGIST = ProblemParams.JGIST;
    rhoADMM = ProblemParams.rhoADMM;
    JADMM = ProblemParams.JADMM;
    UseBatchMethod = ProblemParams.UseBatchMethod;
    SystemIsSparse = ProblemParams.SystemIsSparse;

    NRMSE = Array{Float64}(undef,NoIter);

    ###################################################
    # Initialization:
    ###################################################

    Fvec = vec(F0);
    o = o0;

    SystemVec = vec(System);

    H = kron(Input[:,1]', Matrix{Float64}(I,L,L));

    auxVec = invRnoise*(Output[:,1] - H*Fvec);
    Resolvent = inv(invRnoise + rhoADMM*Matrix{Float64}(I,L,L));
    o = ADMM(auxVec, Resolvent, lambdaORRLSl1, rhoADMM, JADMM, o);

    K = H'*invRnoise*H; K = (K+K')/2; K = Symmetric(K);
    Q = H'*invRnoise*(Output[:,1]-o);

    Gamma = 1.0;
    VV = eigen(K);
    vVec = VV.vectors[:,P*L];

    FvecPrev = Fvec;

    varpi = MaxEigInvRnoise*(norm(Input[:,1])^2)/Gamma;
    Grad = MultiplyGradBy*(1/Gamma)*(K*Fvec - Q)/varpi;
    FvecHalf = Fvec - alpha*Grad;

    if SystemIsSparse == "yes"
        Fvec = Proxl1(FvecHalf, lambda);
    else
        Fvec = FvecHalf;
    end

    GradPrev = Grad;

    BEGIN = time();
    for n in 2:(NoIter+1)

        H = kron(Input[:,n]', Matrix{Float64}(I,L,L));

        auxVec = invRnoise*(Output[:,n] - H*Fvec);
        o = ADMM(auxVec, Resolvent, lambdaORRLSl1, rhoADMM, JADMM, o);

        Gamma = gamma*Gamma + 1.0;
        K = gamma*K + H'*invRnoise*H;
        Q = gamma*Q + H'*invRnoise*(Output[:,n]-o);

        uVec = K*vVec/Gamma;
        vVec = uVec/norm(uVec);
        varpi = vVec'*K*vVec/Gamma + eVarPi;

        Grad = MultiplyGradBy*(1/Gamma)*(K*Fvec - Q)/varpi;

        FvecHalf = FvecHalf + Fvec - FvecPrev + alpha*GradPrev - Grad;

        FvecPrev = Fvec;

        Fvec = FvecHalf;

        if SystemIsSparse == "yes"
            Fvec = Proxl1(FvecHalf, lambda);
        else
            Fvec = FvecHalf;
        end
        GradPrev = Grad;

        NRMSE[n-1] = norm(Fvec-SystemVec)/norm(SystemVec);

    end
    END = time();
    Time = (END-BEGIN)/NoIter;

    return NRMSE, Time;

end

##############################################################################

function HRLS_GIST(ProblemParams)

    ###################################################
    # Parameters:
    ###################################################

    eVarPi = ProblemParams.eVarPi;
    alpha = ProblemParams.alpha;
    lambda = ProblemParams.lambda;
    MultiplyGradBy = ProblemParams.MultiplyGradBy;

    NoIter = ProblemParams.NoIter;
    P = ProblemParams.P;
    L = ProblemParams.L;
    lambdaF = ProblemParams.lambdaF;
    Radius = ProblemParams.Radius;
    Input = ProblemParams.Input;
    Output = ProblemParams.Output;
    LastInput0 = ProblemParams.LastInput0;
    LastOutput0 = ProblemParams.LastOutput0;
    MaxEigInvRnoise = ProblemParams.MaxEigInvRnoise;
    invRnoise = ProblemParams.invRnoise;
    System = ProblemParams.System;
    F0 = ProblemParams.F0HRLS;
    o0 = ProblemParams.o0HRLS;
    lambdao = ProblemParams.lambdao;
    rho = ProblemParams.rho;
    lambdaFMHSDM = ProblemParams.lambdaFMHSDM;
    alphaFMHSDM = ProblemParams.alphaFMHSDM;
    JFMHSDM = ProblemParams.JFMHSDM;
    thetaMCP = ProblemParams.thetaMCP;
    lambdaMCP = ProblemParams.lambdaMCP;
    tminMCP = ProblemParams.tminMCP;
    tmaxMCP = ProblemParams.tmaxMCP;
    etaMCP = ProblemParams.etaMCP;
    sigmaMCP = ProblemParams.sigmaMCP;
    mBuffMCP = ProblemParams.mBuffMCP;
    JGIST = ProblemParams.JGIST;
    rhoADMM = ProblemParams.rhoADMM;
    JADMM = ProblemParams.JADMM;
    UseBatchMethod = ProblemParams.UseBatchMethod;
    SystemIsSparse = ProblemParams.SystemIsSparse;

    NRMSE = Array{Float64}(undef,NoIter);

    ###################################################
    # Initialization:
    ###################################################

    Fvec = vec(F0);
    o = o0;

    SystemVec = vec(System);

    H = kron(Input[:,1]', Matrix{Float64}(I,L,L));

    auxVec = Output[:,1] - H*Fvec;
    o = GIST(o, L, auxVec, invRnoise, lambdaMCP, thetaMCP, tminMCP, tmaxMCP, etaMCP, sigmaMCP, mBuffMCP, JGIST);

    K = H'*invRnoise*H; K = (K+K')/2; K = Symmetric(K);

    Q = H'*invRnoise*(Output[:,1]-o);

    Gamma = 1.0;
    VV = eigen(K);
    vVec = VV.vectors[:,P*L];

    FvecPrev = Fvec;
    varpi = MaxEigInvRnoise*(norm(Input[:,1])^2)/Gamma;
    Grad = MultiplyGradBy*(1/Gamma)*(K*Fvec - Q)/varpi;
    FvecHalf = Fvec - alpha*Grad;

    if SystemIsSparse == "yes"
        Fvec = Proxl1(FvecHalf, lambda);
    else
        Fvec = FvecHalf;
    end

    GradPrev = Grad;

    BEGIN = time();
    for n in 2:(NoIter+1)

        H = kron(Input[:,n]', Matrix{Float64}(I,L,L));

        auxVec = Output[:,n] - H*Fvec;
        o = GIST(o, L, auxVec, invRnoise, lambdaMCP, thetaMCP, tminMCP, tmaxMCP, etaMCP, sigmaMCP, mBuffMCP, JGIST);

        Gamma = gamma*Gamma + 1.0;
        K = gamma*K + H'*invRnoise*H;

        Q = gamma*Q + H'*invRnoise*(Output[:,n]-o);

        uVec = K*vVec/Gamma;
        vVec = uVec/norm(uVec);
        varpi = vVec'*K*vVec/Gamma + eVarPi;

        Grad = MultiplyGradBy*(1/Gamma)*(K*Fvec - Q)/varpi;

        FvecHalf = FvecHalf + Fvec - FvecPrev + alpha*GradPrev - Grad;

        FvecPrev = Fvec;

        Fvec = FvecHalf;

        if SystemIsSparse == "yes"
            Fvec = Proxl1(FvecHalf, lambda);
        else
            Fvec = FvecHalf;
        end

        GradPrev = Grad;

        NRMSE[n-1] = norm(Fvec-SystemVec)/norm(SystemVec);

    end
    END = time();
    Time = (END-BEGIN)/NoIter;

    return NRMSE, Time;

end

############################################################################

function HRLS_FMHSDM(ProblemParams)

    ###################################################
    # Parameters:
    ###################################################

    eVarPi = ProblemParams.eVarPi;
    alpha = ProblemParams.alpha;
    lambda = ProblemParams.lambda;
    MultiplyGradBy = ProblemParams.MultiplyGradBy;

    NoIter = ProblemParams.NoIter;
    P = ProblemParams.P;
    L = ProblemParams.L;
    lambdaF = ProblemParams.lambdaF;
    Radius = ProblemParams.Radius;
    Input = ProblemParams.Input;
    Output = ProblemParams.Output;
    LastInput0 = ProblemParams.LastInput0;
    LastOutput0 = ProblemParams.LastOutput0;
    MaxEigInvRnoise = ProblemParams.MaxEigInvRnoise;
    invRnoise = ProblemParams.invRnoise;
    System = ProblemParams.System;
    F0 = ProblemParams.F0HRLS;
    o0 = ProblemParams.o0HRLS;
    lambdao = ProblemParams.lambdao;
    rho = ProblemParams.rho;
    lambdaFMHSDM = ProblemParams.lambdaFMHSDM;
    alphaFMHSDM = ProblemParams.alphaFMHSDM;
    JFMHSDM = ProblemParams.JFMHSDM;
    thetaMCP = ProblemParams.thetaMCP;
    lambdaMCP = ProblemParams.lambdaMCP;
    tminMCP = ProblemParams.tminMCP;
    tmaxMCP = ProblemParams.tmaxMCP;
    etaMCP = ProblemParams.etaMCP;
    sigmaMCP = ProblemParams.sigmaMCP;
    mBuffMCP = ProblemParams.mBuffMCP;
    JGIST = ProblemParams.JGIST;
    rhoADMM = ProblemParams.rhoADMM;
    JADMM = ProblemParams.JADMM;
    UseBatchMethod = ProblemParams.UseBatchMethod;
    SystemIsSparse = ProblemParams.SystemIsSparse;

    NRMSE = Array{Float64}(undef,NoIter);

    ###################################################
    # Initialization:
    ###################################################

    Fvec = vec(F0);
    o = o0;

    SystemVec = vec(System);

    H = kron(Input[:,1]', Matrix{Float64}(I,L,L));

    Resolvent = inv(invRnoise + Matrix{Float64}(I,L,L)/lambdaFMHSDM);
    auxVec = invRnoise*(Output[:,1] - H*Fvec);
    o = FMHSDMforLASSO(Resolvent, auxVec, lambdao, lambdaFMHSDM, alphaFMHSDM, JFMHSDM, o);

    K = H'*invRnoise*H; K = (K+K')/2; K = Symmetric(K);
    Q = H'*invRnoise*(Output[:,1]-o);
    Gamma = 1.0;
    VV = eigen(K);
    vVec = VV.vectors[:,P*L];

    FvecPrev = Fvec;

    varpi = MaxEigInvRnoise*(norm(Input[:,1])^2)/Gamma;
    Grad = MultiplyGradBy*(1/Gamma)*(K*Fvec - Q)/varpi;
    FvecHalf = Fvec - alpha*Grad;
    if SystemIsSparse == "yes"
        Fvec = Proxl1(FvecHalf, lambda);
    else
        Fvec = FvecHalf;
    end

    GradPrev = Grad;

    BEGIN = time();
    for n in 2:(NoIter+1)

        H = kron(Input[:,n]', Matrix{Float64}(I,L,L));

        auxVec = invRnoise*(Output[:,n] - H*Fvec);
        o = FMHSDMforLASSO(Resolvent, auxVec, lambdao, lambdaFMHSDM, alphaFMHSDM, JFMHSDM, o);

        Gamma = gamma*Gamma + 1.0;
        K = gamma*K + H'*invRnoise*H;
        Q = gamma*Q + H'*invRnoise*(Output[:,n]-o);

        uVec = K*vVec/Gamma;
        vVec = uVec/norm(uVec);
        varpi = vVec'*K*vVec/Gamma + eVarPi;

        Grad = MultiplyGradBy*(1/Gamma)*(K*Fvec - Q)/varpi;
        FvecHalf = FvecHalf + Fvec - FvecPrev + alpha*GradPrev - Grad;
        FvecPrev = Fvec;
        Fvec = FvecHalf;
        if SystemIsSparse == "yes"
            Fvec = Proxl1(FvecHalf, lambda);
        else
            Fvec = FvecHalf;
        end
        GradPrev = Grad;

        NRMSE[n-1] = norm(Fvec-SystemVec)/norm(SystemVec);

    end
    END = time();
    Time = (END-BEGIN)/NoIter;

    return NRMSE, Time;

end

############################################################################

