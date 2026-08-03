//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2026 Han Song
// Modified from Copyright (c) 2021 Yoshiya Usui
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//-------------------------------------------------------------------------------------------------------
#ifndef DBLDEF_ANALYSIS_CONTROL
#define DBLDEF_ANALYSIS_CONTROL

#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <time.h>
#include <set>
#include <vector>
#include "Inversion.h"
#include "Forward3D.h"
#include "Forward3DBrickElement0thOrder.h"
#include "Forward3DTetraElement0thOrder.h"
#include "Forward3DNonConformingHexaElement0thOrder.h"
#include "MeshData.h"

// Class of Analysis Control
class AnalysisControl
{

public:
	// Type of boundary condition at the bottom of the model
	static const int BOUNDARY_BOTTOM_ONE_DIMENSIONAL = 0;	// Boundary condition on which one dimensional relation holds for the EM field at the bottom
	static const int BOUNDARY_BOTTOM_PERFECT_CONDUCTOR = 1; // Boundary condition on which electric field is null at the bottom

	// Selection of Regularized Parameters (Trade-Off Parameter (TO))
	enum TypeOfRegularizedParameters
	{
		Parameter_TYPE_UNDEFINED = -1,
		TO_Fixed = 0, // 0: fixed constant
		TO_ABIC_LS = 1,	  // 1: ABIC line search
		TO_OCCAM_LS = 2,  // 2: OCCAM line search
		TO_LINEAR_LCURVE = 3,	 // 3: linear cubic-spline L-curve selection
		TO_NONLINEAR_LCURVE = 4, // 4: nonlinear cubic-spline L-curve selection
		TO_DATA_FIT_COOLING = 5, // 5: data-fit-bracketed cooling
	};

	enum TypeOfReferenceModel
	{
		MOD_TYPE_UNDEFINED = -1,
		Fixed = 0,		 // 0: fixed reference model provided by users
		AfterAdjustment, // 2: reference model is updated model after inversion, also with step length adjustment
	};

	enum TypeOfCGOperator
	{
		CG_TYPE_UNDEFINED = -1,
		FD_CG = 0, // 0: ForwardDifference
		CD_CG,	   // 1: CentralDifference
		MS_CG,	   // 2: MnimumCrossGradientSupport
	};

	// Post-inversion model appraisal mode. This is independent from INV_METHOD.
	enum AppraisalMode
	{
		APPRAISAL_DISABLED = -1,
		APPRAISAL_RESOLUTION_AND_COVARIANCE_DIAGONALS = 0,
		APPRAISAL_RESOLUTION_DIAGONAL = 1,
		APPRAISAL_COVARIANCE_DIAGONAL = 2,
	};

	enum ABICSearchMode
	{
		ABIC_SEARCH_EXACT = 0,
		ABIC_SEARCH_INEXACT = 1,
	};
	// Get weighting factor of alpha
	double getAlphaWeight(const int iDir) const;

	//-----------------------
	//--- ABIC inversion ---
	//-----------------------
	double sign(double val, double ref);
	bool OCCAMinversion() const;
	bool ABICinversion() const;
	bool MinNormInv() const;
	bool LMdamping() const;
	double getstepcutOCC() const;

	//---------------------
	//--- femtic v4.2 ---
	//---------------------
	// IDs of output parameter which can be outputed to VTK file
	enum outputParameterIDsForVTK
	{
		OUTPUT_RESISTIVITY_VALUES_TO_VTK = 0,
		OUTPUT_ELECTRIC_FIELD_VECTORS_TO_VTK,
		OUTPUT_MAGNETIC_FIELD_VECTORS_TO_VTK,
		OUTPUT_CURRENT_DENSITY,
		OUTPUT_SENSITIVITY,
		OUTPUT_SENSITIVITY_DENSITY,
	};

	// The way of numbering
	enum numbering
	{
		NOT_ASSIGNED = -1,
		XYZ = 0,
		YZX = 1,
		ZXY = 2,
	};

	// Flags specifing which backward or forward element is used for calculating EM field
	enum BackwardOrForwardElement
	{
		BACKWARD_ELEMENT = 0,
		FORWARD_ELEMENT,
	};

	// Flags specifing the way of creating roughning matrix
	enum TypeOfRoughningMatrix
	{
		USER_DEFINED_ROUGHNING = -1,
		USE_ELEMENTS_SHARE_FACES = 0,
		USE_RESISTIVITY_BLOCKS_SHARE_FACES,
		USE_ELEMENTS_SHARE_FACES_AREA_VOL_RATIO,
		EndOfTypeOfRoughningMatrix // This must be written at the end
	};

	// Type of galvanic distortion
	enum TypeOfDistortion
	{
		DISTORTION_TYPE_UNDEFINED = -1,
		NO_DISTORTION = 0, // ���õ������
		ESTIMATE_DISTORTION_MATRIX_DIFFERENCE,
		ESTIMATE_GAINS_AND_ROTATIONS,
		ESTIMATE_GAINS_ONLY,
	};

	// Flag of convergence behaviors
	enum ConvergenceBehaviors
	{
		DURING_RETRIALS = 0,
		GO_TO_NEXT_ITERATION,
		INVERSIN_CONVERGED,
	};

	struct UseBackwardOrForwardElement
	{
		enum BackwardOrForwardElement directionX;
		enum BackwardOrForwardElement directionY;
	};

	// Type of electric field used to calculate response functions
	enum TypeOfElectricField
	{
		USE_HORIZONTAL_ELECTRIC_FIELD = 0,
		USE_TANGENTIAL_ELECTRIC_FIELD = 1,
	};

	// Type of owner element
	enum TypeOfOwnerElement
	{
		USE_LOWER_ELEMENT = 0,
		USE_UPPER_ELEMENT = 1,
	};

	// Option about treatment of apparent resistivity & phase
	enum AppResPhaseTreatmentOption
	{
		NO_SPECIAL_TREATMENT_APP_AND_PHASE = 0,
		USE_Z_IF_SIGN_OF_RE_Z_DIFFER = 1,
	};

	// Type of data space algorithm
	enum TypeOfDataSpaceAlgorithm
	{
		NEW_DATA_SPACE_ALGORITHM = 1,
		NEW_DATA_SPACE_ALGORITHM_USING_INV_RTR_MATRIX = 2,
	};

#ifdef _ANISOTOROPY
	// Type of anisotropy
	enum TypeOfAnisotropy
	{
		NO_ANISOTROPY = 0,
		AXIAL_ANISOTROPY = 1,
	};
#endif

	// Return the the instance of the class
	static AnalysisControl *getInstance();

	// Run analysis
	void run();

	// SONG2023 (2/2)
	double m_beta;

	// Calculate and output elapsed time
	std::string outputElapsedTime() const;
	double getElapsedTimeInSeconds() const;

	// Get type of boundary condition at the bottom of the model
	int getBoundaryConditionBottom() const;

	// Get order of finite element
	int getOrderOfFiniteElement() const;

	// Get process ID
	int getMyPE() const;

	// Get total number of processes
	int getTotalPE() const;

	// Get total number of threads
	int getNumThreads() const;

	// Get flag specifing either incore or out-of-core version of PARDISO is used
	int getModeOfPARDISO() const;

	// Get flag specifing the way of numbering of edges or nodess
	int getNumberingMethod() const;

	// Get flag specifing whether the results of 2D forward calculations are outputed
	bool getIsOutput2DResult() const;

	// Get initial iteration number
	int getIterationNumInit() const;

	// Get current iteration number
	int getIterationNumCurrent() const;

	// Get maximum iteration number
	int getIterationNumMax() const;

	// Get residual updated or not
	int getresidualupdate() const;

	// Get type of Cross-Gradient operator
	int gettypeofCG() const;

	// Get member variable specifing which backward or forward element is used for calculating EM field
	const UseBackwardOrForwardElement getUseBackwardOrForwardElement() const;

	// Get whether the specified parameter is outputed to VTK file
	bool doesOutputToVTK(const int paramID) const;

	// Get trade-off parameter for resistivity value
	double getTradeOffParameterForResistivityValue() const;
	int getNumTO() const;
	bool getloglog() const;
	bool getnorm() const;
	double get_ithTradeOffParameterForResistivityValue(const int ito) const;
	void recordLCurveSelectionDiagnostics(
		const int iteration,
		const std::string& mode,
		const std::string& roughnessOperator,
		const double selectedAlpha,
		const double selectedPredictedDataMisfit,
		const double selectedPredictedRms,
		const double selectedModelRoughness,
		const double maxCurvature,
		const std::string& failureIndicators);
	void setLCurveFinalTradeOffParameterForDiagnostics(const double finalAlpha);

	double getTradeOffParameterForMinNorm() const;

	double getDampingofLM() const;

	double getTradeOffParameterForCrossGradient() const;

	// Get datamisfit
	double getdatamisfit() const;

	// Get trade-off parameter for distortion matrix complexity
	double getTradeOffParameterForDistortionMatrixComplexity() const;

	// Get trade-off parameter for gains of distortion matrix
	double getTradeOffParameterForGainsOfDistortionMatrix() const;

	// Get trade-off parameter for rotations of distortion matrix
	double getTradeOffParameterForRotationsOfDistortionMatrix() const;

	// Get current factor of step length damping
	double getStepLengthDampingFactorCur() const;

	// Get small value for Cross-Gradient
	double getSmallvalueforCrossGradient() const;

	// Get maximum number of cutbacks.
	int getNumCutbackMax() const;

	// Get flag whether memory of solver is held after forward calculation
	bool holdMemoryForwardSolver() const;

	bool runCG() const;

	// Get type of mesh
	int getTypeOfMesh() const;

	// Get flag specifing whether distortion matrix is estimated or not
	bool estimateDistortionMatrix() const;

	// Get type of galvanic distortion
	int getTypeOfDistortion() const;

	// Get flag specifing the way of creating roughning matrix
	int geTypeOfRoughningMatrix() const;

	// Get type of the electric field used to calculate response functions
	int getTypeOfElectricField() const;

	// Flag specifing whether type of the electric field of each site is specified indivisually
	bool isTypeOfElectricFieldSetIndivisually() const;

	// Tyep of owner element of observation sites
	int getTypeOfOwnerElement() const;

	// Flag specifing whether the type of owner element of each site is specified indivisually
	bool isTypeOfOwnerElementSetIndivisually() const;

	// Get division number of right-hand sides at solve phase in forward calculation
	int getDivisionNumberOfMultipleRHSInForward() const;

	// Get division number of right-hand sides at solve phase in inversion
	int getDivisionNumberOfMultipleRHSInInversion() const;

	// Get flag specifing whether the coefficient matrix of the normal equation is positive definite or not
	bool getPositiveDefiniteNormalEqMatrix() const;

	// Get flag specifing whether output file for paraview is binary or ascii
	bool writeBinaryFormat() const;

	// Get inversion method
	int getInversionMethod() const;

	// Get flag specifing whether observation point is moved to the horizontal center of the element including it
	int getIsObsLocMovedToCenter() const;

	// Get option about treatment of apparent resistivity & phase
	int getApparentResistivityAndPhaseTreatmentOption() const;

	// Get flag specifing whether roughening matrix is outputted
	bool getIsRougheningMatrixOutputted() const;

	// Get type of data space algorithm
	int getTypeOfDataSpaceAlgorithm() const;

	// Get flag specifing whether Lp optimization with difference filter is used
	bool useDifferenceFilter() const;

	// Get degree of Lp optimization
	int getDegreeOfLpOptimization() const;
	int getDegreeOfLpMinimumNorm() const;
	double getSmallValueOfMinimumSupport() const;
	double getSmallValueOfMinimumGradientSupport() const;

	// Get lower limit of the difference of log10(rho) for Lp optimization
	double getLowerLimitOfDifflog10RhoForLpOptimization() const;
	double getLowerLimitOfDifflog10RhoForLpMinimumNorm() const;

	// Get upper limit of the difference of log10(rho) for Lp optimization
	double getUpperLimitOfDifflog10RhoForLpOptimization() const;
	double getUpperLimitOfDifflog10RhoForLpMinimumNorm() const;

	// Get maximum iteration number of IRWLS for Lp optimization
	int getMaxIterationIRWLSForLpOptimization() const;

	// Get threshold value for deciding convergence about IRWLS for Lp optimization
	double getThresholdIRWLSForLpOptimization() const;

	void getResidualVectorOfDataThisPE(double *vector) const;

	void seticut(int value);

	int geticut() const;

	// Get directory of out-of-core files for the sensitivitry matrix
	std::string getDirectoryOfOutOfCoreFilesForSensitivityMatrix() const;

	int getAppraisalMode() const;
	bool isAppraisalEnabled() const;
	int getNumRandomVectorsForAppraisal() const;
	const std::vector<int>& getAppraisalCheckpoints() const;
	std::string getAppraisalInputSensitivityDirectory() const;
	std::string getAppraisalOutputDirectory() const;
	bool writeLegacyAppraisalDsdkFiles() const;

	// Get pointer to the object of class Forward3D
	Forward3D *getPointerOfForward3D() const;

#ifdef _ANISOTOROPY
	// Get type of anisotropy
	int getTypeOfAnisotropy() const;

	// Get flag specifing whether anisotropy of conductivity is taken into account
	bool isAnisotropyConsidered() const;
#endif

	// Get pointer to the object of class MeshData
	const MeshData *getPointerOfMeshData() const;

	// Get pointer to the object of class MeshDataBrickElement
	const MeshDataBrickElement *getPointerOfMeshDataBrickElement() const;

	// Get pointer to the object of class MeshDataTetraElement
	const MeshDataTetraElement *getPointerOfMeshDataTetraElement() const;

	// Get pointer to the object of class MeshDataNonConformingHexaElement
	const MeshDataNonConformingHexaElement *getPointerOfMeshDataNonConformingHexaElement() const;

private:
	// Constructor
	AnalysisControl();

	// Destructor
	~AnalysisControl();

	// Copy constructor
	AnalysisControl(const AnalysisControl &rhs)
	{
		std::cerr << "Error : Copy constructor of the class AnalysisControl is not implemented." << std::endl;
		exit(1);
	};

	// Copy assignment operator
	AnalysisControl &operator=(const AnalysisControl &rhs)
	{
		std::cerr << "Error : Assignment operator of the class AnalysisControl is not implemented." << std::endl;
		exit(1);
	}

	// Read analysis control data from "control.dat"
	void inputControlData();

	// Flag specifing whether each parameter has already read from control.dat
	enum controlParameterID
	{
		BOUNDARY_CONDITION_BOTTOM = 0,
		NUM_THREADS,
		FWD_SOLVER,
		MEM_LIMIT,
		OUTPUT_PARAM,
		NUMBERING_METHOD,
		OUTPUT_OPTION,
		OUTPUT_2D_RESULTS,
		TRADE_OFF_PARAM,
		ITERATION,
		DECREASE_THRESHOLD,
		CONVERGE,
		RETRIAL,
		STEP_LENGTH,
		MESH_TYPE,
		DISTORTION,
		WEIGHT_OF_DISTORTION,	 // SONG(2024/11/5 10:00)
		TYPE_OF_TRADE_OFF_PARAMETER,			 // SONG(2024/11/5 10:00)
		WEIGHT_OF_REFERENCE, // Reference Model SONG(2025/09/11 10:00)
		TYPE_OF_REFERENCE,	 // Reference Model TYPE SONG(2026/04/28 10:00)
		NORM_OF_MINIMUMNORM, // MinimumNorm SONG(2025/09/11 10:00)
		TRADE_OFF_CG,		 // Cross-Gradient SONG(2025/01/23 10:00)
		TYPE_OF_CG,			 // SONG(2024/11/5 10:00)
		ROUGH_MATRIX,
		ELEC_FIELD,
		DIV_NUM_RHS_FWD,
		DIV_NUM_RHS_INV,
		RESISTIVITY_BOUNDS,
		OFILE_TYPE,
		HOLD_FWD_MEM,
		ALPHA_WEIGHT,
		INV_MAT_POSITIVE_DEFINITE,
		BOTTOM_RESISTIVITY,
		BOTTOM_ROUGHNING_FACTOR,
		INV_METHOD,
		ABIC_SEARCH_MODE,
		RUN_INEXACT_LINE_SEARCH, // SONG(2026/05/27 10:00)
		RUN_INEXACT_OCCAM_LINE_SEARCH,
		ALPHA_COOLING,
		APPRAISAL_MODE,
		APPRAISAL_RANDOM_VECTORS,
		BOUNDS_DIST_THLD,
		IDW,
		SMALL_VALUE,
		Levenberg_Marquardt, // SONG(2026/05/22 10:00)
		MOVE_OBS_LOC,
		OWNER_ELEMENT,
		APP_PHS_OPTION,
		OUTPUT_ROUGH_MATRIX,
		DATA_SPACE_METHOD,
#ifdef _ANISOTOROPY
		ANISOTROPY,
#endif
		EndOfControlParameterID // This must be written at the end of controlParameterID
	};

	// Total number of the parameters written in control.dat
	static const int numParamWrittenInControlFile = EndOfControlParameterID;
	// Process ID
	int m_myPE;
	// Total number of processes
	int m_totalPE;
	// Total number of threads
	int m_numThreads;
	// Type of boundary condition at the bottom of the model
	int m_boundaryConditionBottom;
	// Order of finite element
	int m_orderOfFiniteElement;
	// Flag specifing either incore or out-of-core version of PARDISO is used
	int m_modeOfPARDISO;
	// Flag specifing the way of numbering of edges or nodess
	int m_numberingMethod;
	// Parameters to be outputed to the file for visualization
	std::set<int> m_outputParametersForVis;
	// Flag specifing whether the results of 2D forward calculations are outputed
	bool m_isOutput2DResult;
	int icut;
	//// Flag specifing whether only forward computation is performed or inversion is performed
	// bool m_performForwardOnly;

	// Trade-off parameter for resistivity value
	int m_NumOF_TO;
	bool m_lCurveUseLogLog;
	bool m_lCurveUseRootNorm;
	bool m_hasLCurveSelectionDiagnostics;
	int m_lCurveSelectionIteration;
	std::string m_lCurveModeName;
	std::string m_lCurveRoughnessOperator;
	std::string m_lCurveFailureIndicators;
	double m_lCurveSelectedAlpha;
	double m_lCurveFinalAlpha;
	double m_lCurveSelectedPredictedDataMisfit;
	double m_lCurveSelectedPredictedRms;
	double m_lCurveSelectedModelRoughness;
	double m_lCurveMaxCurvature;
	bool m_leavingABIC;
	double m_tradeOffParameterForResistivityValue;
	double m_tradeOffParameterForMinNorm;
	double m_tradeOffParameterForCrossGradient;
	double m_smallvalueForCrossGradient;
	bool m_MinNormInv;
	bool m_Levenberg_Marquardt;
	double m_dampingof_LM;
	bool m_CrossGradientInv;
	double m_datamisfit;
	double m_tradeOffParameterInitial;
	double m_tolreq;
	double m_tradeOffParameterForResistivityValuePre;
	double m_RatioForRoughness;
	double m_tradeOffParameterABICA;
	double m_tradeOffParameterABICB;
	double m_tradeOffParameterABICC;
	double m_tradeOffParameterABIClb;
	double m_tradeOffParameterABICub;
	std::vector<double> m_ABICA;
	std::vector<double> m_ABICB;
	std::vector<double> m_ABICC;
	std::vector<double> m_abic;
	std::vector<double> m_abicpre;
	std::vector<double> m_ABIClb;
	std::vector<double> m_ABICub;
	bool m_ABICconverage;
	double m_updatedmean;

	// OCCAM line-search state restored from the legacy v1.0 control flow.
	bool m_OCCAMinversion;
	bool m_OCCAMsmoothing;
	bool m_leavingOCCAM;
	double m_tradeOffParameterOCCA;
	double m_tradeOffParameterOCCB;
	double m_tradeOffParameterOCCC;
	double m_tradeOffParameterOCClb;
	double m_tradeOffParameterOCCub;
	double m_rmsOCCA;
	double m_rmsOCCB;
	double m_rmsOCCC;
	double m_rmsOCClb;
	double m_rmsOCCub;
	double m_stepcutOCC;

	double m_stepsizelb;
	double m_stepsizeub;
	double m_rms;
	double m_rmsPre;
	int m_residualupdated;
	// Array of Trade-off parameter
	double *m_tradeOffParameters;

	// Array of Trade-off parameter
	double *m_residualVectorThisPE;

	// Trade-off parameter for distortion matrix complexity
	double m_tradeOffParameterForDistortionMatrixComplexity;

	// Trade-off parameter for gains of distortion matrix
	double m_tradeOffParameterForDistortionGain;

	// Trade-off parameter for rotation of distortion matrix
	double m_tradeOffParameterForDistortionRotation;

	// The time the class instanced
	time_t m_startTime;
	int m_germanYear;
	int m_germanMonth;
	int m_germanDay;
	int m_germanHour;
	int m_germanMin;
	int m_germanSec;

	// Backward/forward element flags used for EM-field calculation.
	UseBackwardOrForwardElement m_useBackwardOrForwardElement;

	// Initial iteration number
	int m_iterationNumInit;

	// Current iteration number
	int m_iterationNumC;

	// Maximum iteration number
	int m_iterationNumMax;

	// Current iteration number
	int m_iterationNumCurrent;

	// Threshold value for decreasing
	double m_thresholdValueForDecreasing;

	// Criterion of decrease ratio [%].
	// Iteration If the objective functions and its components decrease by more than criterion,
	// the inversion is considered to be converged.
	double m_decreaseRatioForConvegence;

	// Current factor of step length damping
	double m_stepLengthDampingFactorCur;

	double m_stepLengthDampingFactorPre;

	// Minimum factor of step length damping
	double m_stepLengthDampingFactorMin;

	// Maximum factor of step length damping
	double m_stepLengthDampingFactorMax;

	// If value of objective function decrease specified times in a row, factor of step length damping is increases.
	int m_numOfIterIncreaseStepLength;

	// Factor decreasing step length damping factor
	double m_factorDecreasingStepLength;

	// Factor increasing step length damping factor
	double m_factorIncreasingStepLength;

	// Maximum number of cutbacks.
	int m_numCutbackMax;

	// Hold memory of solver after forward calculation
	bool m_holdMemoryForwardSolver;

	// Pointer to the object of class Forward3DBrickElement0thOrder
	Forward3DBrickElement0thOrder *m_ptrForward3DBrickElement0thOrder;

	// Pointer to the object of class Forward3DTetraElement0thOrder
	Forward3DTetraElement0thOrder *m_ptrForward3DTetraElement0thOrder;

	// Pointer to the object of class Forward3DNonConformingHexaElement0thOrder
	Forward3DNonConformingHexaElement0thOrder *m_ptrForward3DNonConformingHexaElement0thOrder;

	// Pointer to the object of class Inversion
	Inversion *m_ptrInversion;

	// Pointer to the object of class Inversion
	Inversion *m_ptrInversiondataspace;

	// Value of objective functional of previous iteration
	double m_objectFunctionalPre;
	double m_objPre;
	double m_objPreiter;

	// Value of objective functional of previous previous iteration //SONG240412-2RatiosForConverage
	double m_objectFunctionalPre2;

	// Data misifit of previous iteration
	double m_dataMisfitPre;

	// Model roughness of previous iteration
	double m_modelRoughnessPre;

	// Norm of distortion matrix differences of previous iteration
	double m_normOfDistortionMatrixDifferencesPre;

	// Norm of the gains of distortion matrices of previous iteration
	double m_normOfGainsPre;

	// Norm of the rotations of distortion matrices of previous iteration
	double m_normOfRotationsPre;

	// Number of consecutive iteration of which the value of objective functional decrase from the one of previous iteration
	int m_numConsecutiveIterFunctionalDecreasing;

	// Flag specifing whether increment iteration without cutback
	bool m_continueWithoutCutback;

	// Maximum value of the memory used by out-of-core mode of PARDISO
	int m_maxMemoryPARDISO;

	// Type of mesh
	int m_typeOfMesh;

	// Flag specifing the way of creating roughning matrix
	int m_typeOfRoughningMatrix;

	// Type of the electric field used to calculated response functions
	int m_typeOfElectricField;

	// Flag specifing whether type of the electric field of each site is specified indivisually
	bool m_isTypeOfElectricFieldSetIndivisually;

	// Type of owner element of observation sites
	int m_typeOfOwnerElement;

	// Flag specifing whether the owner element (upper or lower) of each site is specified indivisually
	bool m_isTypeOfOwnerElementSetIndivisually;

	// Division number of right-hand sides at solve phase in forward calculation
	int m_divisionNumberOfMultipleRHSInForward;

	// Division number of right-hand sides at solve phase in inversion
	int m_divisionNumberOfMultipleRHSInInversion;

	// Flag specifing whether output file for paraview is binary or ascii
	bool m_binaryOutput;

	// Type of galvanic distortion
	int m_typeOfDistortion;

	// Type of regularization parameter selection scheme
	int m_typeOfTradeOffParam;
	int m_typeOfReferenceModel;

	// Type of Cross-Gradient operator
	int m_typeOfCG;

	// Weighting factor of alpha
	double m_alphaWeight[3];

	// Flag specifing whether the coefficient matrix of the normal equation is positive definite or not
	bool m_positiveDefiniteNormalEqMatrix;

	// Inversion method
	int m_inversionMethod;

	// Flag specifing whether the observation point is moved to the center of the element
	bool m_isObsLocMovedToCenter;

	// Option about treatment of apparent resistivity & phase
	int m_apparentResistivityAndPhaseTreatmentOption;

	// Flag specifing whether roughening matrix is outputed
	bool m_isRougheningMatrixOutputted;

	// Type of data space algorithm
	int m_typeOfDataSpaceAlgorithm;

	// Flag specifing whether Lp optimization with difference filter is used
	bool m_useDifferenceFilter;

	// Flag specifing whether ABIC inversion is used
	bool m_ABICinversion;
	ABICSearchMode m_abicSearchMode;
	bool m_inexactMinimizationOfOCCAM;
	double m_dataFitCoolingInitialAlpha;
	double m_dataFitCoolingInitialRmsDecreaseThreshold;
	double m_dataFitCoolingTriggerThreshold;
	double m_dataFitCoolingFactor;
	double m_dataFitCoolingMinimumAlpha;
	bool m_dataFitCoolingHasSelectedAlpha;
	double m_dataFitCoolingPersistentAlpha;
	double m_dataFitCoolingPreviousAcceptedRms;
	double m_dataFitCoolingTrialRms;
	int m_dataFitCoolingCount;
	int m_dataFitCoolingCurrentUpdateIteration;
	std::string m_dataFitCoolingCurrentAlphaSource;
	bool m_stopAfterDataFitCooling;

	// Degree of Lp optimization
	int m_degreeOfLpOptimization;
	int m_degreeOfLpMinimumNorm;

	// Lower limit of the difference of log10(rho) for Lp optimization
	double m_lowerLimitOfDifflog10RhoForLpOptimization;
	double m_lowerLimitOfDifflog10RhoForLpMinimumNorm;
	double m_smallvauleOfMinimumSupport;
	double m_smallvauleOfMinimumGradientSupport;

	// Upper limit of the difference of log10(rho) for Lp optimization
	double m_upperLimitOfDifflog10RhoForLpOptimization;
	double m_upperLimitOfDifflog10RhoForLpMinimumNorm;

	// Maximum iteration number of IRWLS for Lp optimization
	int m_maxIterationIRWLSForLpOptimization;

	// Threshold value for deciding convergence about IRWLS for Lp optimization
	double m_thresholdIRWLSForLpOptimization;

	// Directory of out-of-core files for the sensitivitry matrix
	std::string m_directoryOfOutOfCoreFilesForSensitivityMatrix;

	// Post-inversion appraisal input/output contract.
	int m_appraisalMode;
	int m_numRandomVectorsForAppraisal;
	std::vector<int> m_appraisalCheckpoints;
	std::string m_appraisalInputSensitivityDirectory;
	std::string m_appraisalOutputDirectory;
	bool m_writeLegacyAppraisalDsdkFiles;

	// Terminate the inversion loop if nonlinear L-curve diagnostics cannot select an alpha
	bool m_stopAfterNonlinearLCurveDiagnostics;

#ifdef _ANISOTOROPY
	// Type of anisotropy
	int m_typeOfAnisotropy;
#endif

	// Calculate forward computation
	void calcForwardComputation(const int iter, const bool reuseSelectedTrialForwardResponse);

	// Calculate forward responses for one nonlinear L-curve trial without derivative or sensitivity work
	void calcForwardResponseForNonlinearLCurveTrial(const int iter, const int trialIndex, const double tradeOffParameter);

	// Run actual-response diagnostics for nonlinear cubic-spline L-curve selection
	void runNonlinearLCurveDiagnostics(const int iter, const char* regularizationLabel);

	// Return whether the standalone data-fit cooling method is active
	bool isDataFitCoolingMode() const;

	// Calculate the global RMS represented by the current response state
	double calculateCurrentGlobalRms() const;

	// Evaluate one cooling candidate using a full forward response without sensitivity work
	void runDataFitCoolingTrial(const double alpha, const int trialIndex, const char* alphaSource);

	// Select the largest tested initial alpha that produces a meaningful RMS decrease
	bool runInitialDataFitCoolingBracket();

	// Select the next full-step model using only bounded alpha cooling retries
	bool runPersistentDataFitCoolingAlpha();

	// Record an evaluated update and apply cooling only after acceptance
	void applyAcceptedDataFitCoolingDecision(
		const int responseIteration,
		const int cutbackCount,
		const double currentRms,
		const double stepLengthUsed,
		const bool accepted,
		const bool terminating,
		const char* terminationReason);

	// Append one actual forward trial to the maintained cooling diagnostic CSV
	void appendDataFitCoolingTrialSummary(
		const int iteration,
		const int trialIndex,
		const char* alphaSource,
		const double trialAlpha,
		const double acceptedAlpha,
		const double nextAlpha,
		const double previousRms,
		const double trialRms,
		const bool coolingTriggered,
		const double stepLength,
		const int cutbackCount,
		const char* acceptanceStatus,
		const char* terminationReason) const;

	// BRACKET THE MINIMUM USING MINBRK AND TWO GUESSES
	void minbrkABIC();

	// Return the iteration-aware log10(alpha) span between the first two ABIC bracket points
	double getInitialABICLog10BracketSpan() const;

	// Return whether the current ABIC search uses inexact bracket selection
	bool usesInexactABICSearch() const;

	// Return whether the current ABIC search should stop after bracket selection
	bool shouldUseInexactABICBracketOnly(const int cutbackCount) const;

	// Return whether cutback reuses the current inexact alpha without a new line search
	bool shouldReuseInexactABICAlphaOnCutback(const int cutbackCount) const;

	// Run one reduced-step ABIC trial using the currently selected alpha
	void runReducedStepTrialWithCurrentABICAlpha();

	// RETURNS THE MINIMUM VALUE OF A FUNCTION
	std::vector<double> fminbrentABIC(const int icut);

	// RETURNS THE root OF A FUNCTION
	double frootABIC();

	// BRACKET THE MINIMUM USING MINBRK AND TWO GUESSES FOR OCCAM
	void minbrkOCC();

	// RETURNS THE MINIMUM RMS VALUE FOR OCCAM
	double fminbrentOCC();

	// RETURNS THE RMS-TOLERANCE ROOT FOR OCCAM
	double frootOCC();

	// Run OCCAM Phase I/II trade-off search and restore the selected model
	void runOCCAMLineSearch(const char* regularizationLabel);

	// Adjust factor of step length damping and output convergence data to cnv file
	AnalysisControl::ConvergenceBehaviors adjustStepLengthDampingFactor(const int iterCur, const int iCutbackCur);

	// Return whether convergence data should be written to the CNV file
	bool shouldWriteConvergenceDataToCnv(const int iterCur) const;

	// Exit if the CNV file is not open before convergence-data output
	void ensureCnvFileIsOpenForConvergence() const;

	// Append accepted-model diagnostics for the latest L-curve selection
	void appendLCurveNonlinearCheckDiagnostics(
		const int iteration,
		const int retrial,
		const double actualDataMisfit,
		const double actualRms,
		const double acceptedModelRoughness,
		const double stepLengthFactorUsed);

	bool checkConvergence(const double objectFunctionalCur, const int iterCur); // SONG240412-2RatiosForConverage

	bool checkConvergence(const double objectFunctionalCur, const double dataMisft,
						  const double modelRoughness, const double normDist1 = 0.0, const double normDist2 = 0.0);

	// Return flag specifing whether sensitivity is calculated or not
	bool doesCalculateSensitivity(const int iter) const;

	// Return whether selected-trial forward-response cache can replace forward response calculation
	bool canUseSelectedTrialForwardResponseCache(const int iter) const;

	// Return whether selected-trial forward-state cache can replace forward-field and sensitivity-state work
	bool canUseSelectedTrialForwardStateCache(const int iter) const;
};

#endif
