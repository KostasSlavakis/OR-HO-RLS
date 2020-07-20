############################################
###########################################
##########################################
# Main course:
##########################################
#############################################
##############################################
include("./Params_And_Functions.jl")

NRMSE_RLS = Matrix{Float64}(undef, NoIter,NoExp);
NRMSE_ORRLSl1 = Matrix{Float64}(undef, NoIter,NoExp);
NRMSE_ORRLSMCP = Matrix{Float64}(undef, NoIter,NoExp);
NRMSE_ORRLSCDl1 = Matrix{Float64}(undef, NoIter,NoExp);
NRMSE_ORRLSCDMCP = Matrix{Float64}(undef, NoIter,NoExp);
NRMSE_HRLS_ADMM = Matrix{Float64}(undef, NoIter,NoExp);
NRMSE_HRLS_GIST = Matrix{Float64}(undef, NoIter,NoExp);
NRMSE_HRLS_FMHSDM = Matrix{Float64}(undef, NoIter,NoExp);

TimeRLS = Vector{Float64}(undef, NoExp);
TimeORRLSl1 = Vector{Float64}(undef, NoExp);
TimeORRLSMCP = Vector{Float64}(undef, NoExp);
TimeORRLSCDl1 = Vector{Float64}(undef, NoExp);
TimeORRLSCDMCP = Vector{Float64}(undef, NoExp);
TimeORHRLSADMM = Vector{Float64}(undef, NoExp);
TimeORHRLSGIST = Vector{Float64}(undef, NoExp);
TimeORHRLSFMHSDM = Vector{Float64}(undef, NoExp);

for iEXP in 1:NoExp

    @printf("********************** Experiment %d/%d **********************\n", iEXP, NoExp);
    print("-- Initializing..\n");

    System, Rnoise, invRnoise, MaxEigInvRnoise, VarWhiteNoise, ARmatrix = GenerateModelMatrices(L, ARcoeffU, ARcoeffL, SNR, SystemIsSparse, SystemSparsityLevel);
    Input0, Output0, LastNoiseVec = InputOutput(P,#
                                                L,#
                                                K0,#
                                                System,#
                                                ARmatrix,#
                                                pBernoulli,#
                                                VarOutliers,#
                                                VarWhiteNoise,#
                                                sqrt(VarWhiteNoise)*randn(L));
    print("-- Warm start via Gurobi..\n");
    F0, o0 = WarmStart(P, L, K0+1, 1, invRnoise, lambdaORRLSl1, Input0, Output0);
    F0HRLS, o0HRLS = WarmStart(P, L, K0+1, 2*(K0+1), invRnoise, lambdao, Input0, Output0);
    print("-- Generating input-output data..\n");
    Input, Output, _ = InputOutput(P,#
                                   L,#
                                   NoIter,#
                                   System,#
                                   ARmatrix,#
                                   pBernoulli,#
                                   VarOutliers,#
                                   VarWhiteNoise,#
                                   LastNoiseVec);
    ProblemParams = GenerateProblemParams(NoIter,#
                                          P,#
                                          L,#
                                          SystemIsSparse,#
                                          Input,#
                                          System,#
                                          Output,#
                                          Input0[:, K0+1],#
                                          Output0[:, K0+1],#
                                          Rnoise,#
                                          invRnoise,#
                                          MaxEigInvRnoise,#
                                          eVarPi,#
                                          alpha,#
                                          Radius,#
                                          gamma,#
                                          epsilonORRLS,#
                                          F0,#
                                          o0,#
                                          F0HRLS,#
                                          o0HRLS,#
                                          lambdaF,#
                                          lambdao,#
                                          rho,#
                                          lambda,#
                                          MultiplyGradBy,#
                                          lambdaFMHSDM,#
                                          alphaFMHSDM,#
                                          JFMHSDM,#
                                          UseBatchMethod,#
                                          lambdaORRLSl1,#
                                          rhoADMM,#
                                          JADMM,#
                                          thetaMCP,#
                                          lambdaMCP,#
                                          tminMCP,#
                                          tmaxMCP,#
                                          etaMCP,#
                                          sigmaMCP,#
                                          mBuffMCP,#
                                          JGIST);

    # Methods:

    NRMSE_RLS[:, iEXP], TimeRLS[iEXP] = RLS(ProblemParams);
    @printf("-- RLS: %esecs/iteration.\n", TimeRLS[iEXP]);

    NRMSE_ORRLSl1[:, iEXP], TimeORRLSl1[iEXP] = ORRLSl1(ProblemParams);
    @printf("-- OR-RLS(LASSO): %esecs/iteration.\n", TimeORRLSl1[iEXP]);

    NRMSE_ORRLSMCP[:,iEXP], TimeORRLSMCP[iEXP] = ORRLSMCP(ProblemParams);
    @printf("-- OR-RLS(MCP): %esecs/iteration.\n", TimeORRLSMCP[iEXP]);

    NRMSE_ORRLSCDl1[:,iEXP], TimeORRLSCDl1[iEXP] =  ORRLSCD_L1(ProblemParams);
    @printf("-- OR-RLS(CDl1): %esecs/iteration.\n", TimeORRLSCDl1[iEXP]);

    NRMSE_ORRLSCDMCP[:,iEXP], TimeORRLSCDMCP[iEXP] = ORRLSCD_MCP(ProblemParams);
    @printf("-- OR-RLS(CDMCP): %esecs/iteration.\n", TimeORRLSCDMCP[iEXP]);

    NRMSE_HRLS_ADMM[:,iEXP], TimeORHRLSADMM[iEXP] = HRLS_ADMM(ProblemParams);
    @printf("-- OR-HRLS(ADMM): %esecs/iteration.\n", TimeORHRLSADMM[iEXP]);

    NRMSE_HRLS_GIST[:,iEXP], TimeORHRLSGIST[iEXP] = HRLS_GIST(ProblemParams);
    @printf("-- OR-HRLS(GIST): %esecs/iteration.\n", TimeORHRLSGIST[iEXP]);

    NRMSE_HRLS_FMHSDM[:,iEXP],  TimeORHRLSFMHSDM[iEXP] = HRLS_FMHSDM(ProblemParams);
    @printf("-- OR-HRLS(FMHSDM): %esecs/iteration.\n", TimeORHRLSFMHSDM[iEXP]);

end

NRMSE_RLSav =  Array{Int64}(undef, NoIter)
NRMSE_ORRLSl1av =  Array{Int64}(undef, NoIter)
NRMSE_ORRLSMCPav =  Array{Int64}(undef, NoIter)
NRMSE_ORRLSCDl1av =  Array{Int64}(undef, NoIter)
NRMSE_ORRLSCDMCPav =  Array{Int64}(undef, NoIter)
NRMSE_HRLS_ADMMav =  Array{Int64}(undef, NoIter)
NRMSE_HRLS_GISTav =  Array{Int64}(undef, NoIter)
NRMSE_HRLS_FMHSDMav =  Array{Int64}(undef, NoIter)

NRMSE_RLSav = sum(NRMSE_RLS, dims=2)/NoExp;
NRMSE_ORRLSl1av = sum(NRMSE_ORRLSl1, dims=2)/NoExp;
NRMSE_ORRLSMCPav = sum(NRMSE_ORRLSMCP, dims=2)/NoExp;
NRMSE_ORRLSCDl1av = sum(NRMSE_ORRLSCDl1, dims=2)/NoExp;
NRMSE_ORRLSCDMCPav = sum(NRMSE_ORRLSCDMCP, dims=2)/NoExp;
NRMSE_HRLS_ADMMav = sum(NRMSE_HRLS_ADMM, dims=2)/NoExp;
NRMSE_HRLS_GISTav = sum(NRMSE_HRLS_GIST, dims=2)/NoExp;
NRMSE_HRLS_FMHSDMav = sum(NRMSE_HRLS_FMHSDM, dims=2)/NoExp;

TimeRLSav = sum(TimeRLS)/NoExp;
TimeORRLSl1av = sum(TimeORRLSl1)/NoExp;
TimeORRLSMCPav = sum(TimeORRLSMCP)/NoExp;
TimeORRLSCDl1av = sum(TimeORRLSCDl1)/NoExp;
TimeORRLSCDMCPav = sum(TimeORRLSCDMCP)/NoExp;
TimeORHRLSADMMav = sum(TimeORHRLSADMM)/NoExp;
TimeORHRLSGISTav = sum(TimeORHRLSGIST)/NoExp;
TimeORHRLSFMHSDMav = sum(TimeORHRLSFMHSDM)/NoExp;

Sq_TimeRLS = Vector{Float64}(undef, NoExp);
Sq_TimeORRLSl1 = Vector{Float64}(undef, NoExp);
Sq_TimeORRLSMCP = Vector{Float64}(undef, NoExp);
Sq_TimeORRLSCDl1 = Vector{Float64}(undef, NoExp);
Sq_TimeORRLSCDMCP = Vector{Float64}(undef, NoExp);
Sq_TimeORHRLSADMM = Vector{Float64}(undef, NoExp);
Sq_TimeORHRLSGIST = Vector{Float64}(undef, NoExp);
Sq_TimeORHRLSFMHSDM = Vector{Float64}(undef, NoExp);

for iEXP in 1:NoExp
    Sq_TimeRLS[iEXP] = (TimeRLSav - TimeRLS[iEXP])^2
    Sq_TimeORRLSl1[iEXP] = (TimeORRLSl1av - TimeORRLSl1[iEXP])^2
    Sq_TimeORRLSMCP[iEXP] = (TimeORRLSMCPav - TimeORRLSMCP[iEXP])^2
    Sq_TimeORRLSCDl1[iEXP] = (TimeORRLSCDl1av - TimeORRLSCDl1[iEXP])^2
    Sq_TimeORRLSCDMCP[iEXP] = (TimeORRLSCDMCPav - TimeORRLSCDMCP[iEXP])^2
    Sq_TimeORHRLSADMM[iEXP] = (TimeORHRLSADMMav - TimeORHRLSADMM[iEXP])^2
    Sq_TimeORHRLSGIST[iEXP] = (TimeORHRLSGISTav - TimeORHRLSGIST[iEXP])^2
    Sq_TimeORHRLSFMHSDM[iEXP] = (TimeORHRLSFMHSDMav - TimeORHRLSFMHSDM[iEXP])^2
end

SD_TimeRLS = sum(Sq_TimeRLS)/NoExp
SD_TimeORRLSl1 = sum(Sq_TimeORRLSl1)/NoExp
SD_TimeORRLSMCP = sum(Sq_TimeORRLSMCP)/NoExp
SD_TimeORRLSCDl1 = sum(Sq_TimeORRLSCDl1)/NoExp
SD_TimeORRLSCDMCP = sum(Sq_TimeORRLSCDMCP)/NoExp
SD_TimeORHRLSADMM = sum(Sq_TimeORHRLSADMM)/NoExp
SD_TimeORHRLSGIST = sum(Sq_TimeORHRLSGIST)/NoExp
SD_TimeORHRLSFMHSDM = sum(Sq_TimeORHRLSFMHSDM)/NoExp

# Data = SharedArray{Int64};
Data = [NRMSE_RLSav NRMSE_ORRLSCDl1av NRMSE_ORRLSCDMCPav NRMSE_ORRLSl1av NRMSE_ORRLSMCPav NRMSE_HRLS_ADMMav NRMSE_HRLS_GISTav NRMSE_HRLS_FMHSDMav];

print("-- Plotting..\n");

# gr()
plotlyjs()
plot(1:NoIter, Data, #
     label = ["OR-RLS(RLS)" "OR-RLS(CD-L1)" "OR-RLS(CD-MCP)" "OR-RLS(LASSO)" "OR-RLS(GIST)" "OR-HRLS(ADMM)" "OR-HRLS(GIST)" "OR-HRLS(FMHSDM)"],#
     xlabel = "Iteration/time index",#
     ylabel = "NRMSE",#
     yaxis = :log10,#
     ylims = (-Inf, 10),#
     lw = 2#
     );

# # display(p)
# # print(Data)
# # savefig("fig")

#  open("VarLinParam_1.txt", "w") do io
#            print("###################################################\r\n")
#            print(io, "Problems Parameters\r\n")
#            print(io,"####################################################\r\n")
#            print(io,"SNR (DB):")
#            writedlm(io,SNR)
#            print(io,"No. of experiments:")
#            writedlm(io,NoExp)
#            print(io,"Number of iterations:")
#            writedlm(io,NoIter)
#            print(io,"Variance for outliers:")
#            writedlm(io,VarOutliers)
#            print(io,"pBernoulli:")
#            writedlm(io,pBernoulli)
#            print(io,"K0:")
#            writedlm(io,K0)
#            print(io,"lambda:")
#            writedlm(io,lambda)
#            print(io,"lambdao:")
#            writedlm(io,lambdao)
#            print(io,"lambdaFMHSDM:")
#            writedlm(io,lambdaFMHSDM)
#            print(io,"JFMHSDM:")
#            writedlm(io,JFMHSDM)
#            print(io,"epsilonORRLS:")
#            writedlm(io,epsilonORRLS)
#            print(io,"lambdaORRLSl1:")
#            writedlm(io,lambdaORRLSl1)
#            print(io,"rhoADMM:")
#            writedlm(io,rhoADMM)
#            print(io,"JADMM:")
#            writedlm(io,JADMM)
#            print(io,"thetaMCP:")
#            writedlm(io,thetaMCP)
#            print(io,"lambdaMCP:")
#            writedlm(io,lambdaMCP)
#            print(io,"JGIST:")
#            writedlm(io,JGIST)
#            print(io,"SD_TimeRLS:")
#            writedlm(io,SD_TimeRLS)

#        end;

#  writedlm("VarLinData_1.txt", Data)
 
#  open("VarLinTime_1.txt", "w") do inou
#     print("###################################################\r\n")
#     print(inou, "Average Time\r\n")
#     print(inou,"####################################################\r\n")
#     print(inou, "Average_TimeRLS:")
#     writedlm(inou,TimeRLSav)
#     print(inou,"Average_TimeORRLSl1:")
#     writedlm(inou,TimeORRLSl1av)
#     print(inou,"Average_TimeORRLSMCP:")
#     writedlm(inou,TimeORRLSMCPav)
#     print(inou,"Average_TimeORRLSCDl1:")
#     writedlm(inou,TimeORRLSCDl1av)
#     print(inou,"Average_TimeORRLSCDMCP:")
#     writedlm(inou,TimeORRLSCDMCPav)
#     print(inou,"Average_TimeORHRLSADMM:")
#     writedlm(inou,TimeORHRLSADMMav)
#     print(inou,"Average_TimeORHRLSGIST:")
#     writedlm(inou,TimeORHRLSGISTav)
#     print(inou,"Average_TimeORHRLSFMHSDM:")
#     writedlm(inou,TimeORHRLSFMHSDMav)
#     print("###################################################\r\n")
#     print(inou, "Standard Deviation\r\n")
#     print(inou,"####################################################\r\n")
#     print(inou, "SD_TimeRLS:")
#     writedlm(inou,SD_TimeRLS)
#     print(inou, "SD_TimeORRLSl1:")
#     writedlm(inou,SD_TimeORRLSl1)
#     print(inou, "SD_TimeORRLSMCP:")
#     writedlm(inou,SD_TimeORRLSMCP)
#     print(inou, "SD_TimeORRLSCDl1:")
#     writedlm(inou,SD_TimeORRLSCDl1)
#     print(inou, "SD_TimeORRLSCDMCP:")
#     writedlm(inou,SD_TimeORRLSCDMCP)
#     print(inou, "SD_TimeORHRLSADMM:")
#     writedlm(inou,SD_TimeORHRLSADMM)
#     print(inou, "SD_TimeORHRLSGIST:")
#     writedlm(inou,SD_TimeORHRLSGIST)
#     print(inou, "SD_TimeORHRLSFMHSDM:")
#     writedlm(inou,SD_TimeORHRLSFMHSDM)
# end;

