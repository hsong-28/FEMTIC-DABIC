CXX           = mpiicpx
# oneAPI 2025 MPI compiler wrappers
CC            = mpiicpx
GCC           = mpigxx

# Default release option
CXXFLAGS      = -O3 \
                -qopenmp \
                -D_LINUX \
                -DMKL_ILP64 \
                -D_USE_OMP \
                -DNDEBUG \
		-diag-disable=10441

# debug option (added -g, changed to -O0)
DEBUG_CXXFLAGS = -g \
                -O0 \
                -qopenmp \
                -D_LINUX \
                -DMKL_ILP64 \
                -D_USE_OMP \
		-diag-disable=10441

DEST          = ./
LDFLAGS       =
MKLROOT ?= /opt/intel/oneapi/mkl/latest
OBJ_DIR       = build/obj
DEPFLAGS      = -MMD -MP

LIBS = -Wl,--start-group \
       $(MKLROOT)/lib/intel64/libmkl_intel_ilp64.so \
       $(MKLROOT)/lib/intel64/libmkl_intel_thread.so \
       $(MKLROOT)/lib/intel64/libmkl_core.so \
       $(MKLROOT)/lib/intel64/libmkl_blacs_intelmpi_ilp64.so \
       -Wl,--end-group \
       -liomp5 -lpthread -lm -ldl
#LIBS          =  -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_ilp64 -liomp5 -lpthread -lm -ldl

SRC_DIRS      = core \
                control \
                io \
                mesh \
                forward \
                inversion \
                model \
                data \
                linalg \
                solver

VPATH         = $(SRC_DIRS)
INCLUDES      = $(addprefix -I,$(SRC_DIRS))

CORE_OBJS     = main.o \
                Util.o

CONTROL_OBJS  = AnalysisControl.o \
                AnalysisControlABICLineSearch.o \
                AnalysisControlConvergence.o \
				AnalysisControlDataFitCooling.o \
                AnalysisControlForward.o \
                AnalysisControlGetters.o \
                AnalysisControlLCurveDiagnostics.o \
                AnalysisControlNonlinearLCurve.o \
                AnalysisControlOCCAMLineSearch.o \
                ControlKeywords.o

IO_OBJS       = FemticDabicFileNames.o \
                FemticDabicRunSummary.o \
                OutputFiles.o

MESH_OBJS     = MeshData.o \
                MeshDataBrickElement.o \
                MeshDataNonConformingHexaElement.o \
                MeshDataTetraElement.o

FORWARD_OBJS  = Forward2D.o \
                Forward2DQuadrilateralElement.o \
                Forward2DNonConformingQuadrilateralElement0thOrderEdgeBased.o \
                Forward2DQuadrilateralElementEdgeBased.o \
                Forward2DSquareElement.o \
                Forward2DSquareElement0thOrderEdgeBased.o \
                Forward2DSquareElement1stOrderEdgeBased.o \
                Forward2DSquareElement1stOrderNodeBased.o \
                Forward2DSquareElement2ndOrderNodeBased.o \
                Forward2DSquareElementEdgeBased.o \
                Forward2DSquareElementNodeBased.o \
                Forward2DTriangleElement.o \
                Forward2DTriangleElement0thOrderEdgeBased.o \
                Forward2DTriangleElement1stOrderNodeBased.o \
                Forward2DTriangleElementEdgeBased.o \
                Forward2DTriangleElementNodeBased.o \
                Forward3D.o \
                Forward3DBrickElement0thOrder.o \
                Forward3DNonConformingHexaElement0thOrder.o \
                Forward3DTetraElement0thOrder.o

INVERSION_OBJS = Inversion.o \
                 LCurveCubicSpline.o \
                 AppraisalSensitivityMatrix.o \
                 AppraisalRougheningState.o \
                 AppraisalResolutionCovarianceOperatorChain.o \
                 AppraisalResolutionCovarianceProduction.o \
                 SensitivityDiagnostics.o \
                 InversionGaussNewtonModelSpace.o \
                 InversionGaussNewtonDataSpace.o \
                 InversionGaussNewtonDataSpaceLCurve.o \
                 InversionGaussNewtonDataSpace_OCCAM.o \
                 InversionGaussNewtonDataSpace_ABIC.o

MODEL_OBJS    = ConstrainingModel.o \
                ResistivityBlock.o \
                RougheningMatrix.o \
                RougheningSquareMatrix.o

DATA_OBJS     = AdditionalOutputPoint.o \
                ObservedDataStation.o \
                ObservedDataStationPoint.o \
                ObservedDataStationMT.o \
                ObservedDataStationApparentResistivityAndPhase.o \
                ObservedDataStationHTF.o \
                ObservedDataStationVTF.o \
                ObservedDataStationPT.o \
                ObservedDataStationNMT.o \
                ObservedDataStationNMT2.o \
                ObservedDataStationNMT2ApparentResistivityAndPhase.o \
                ObservedData.o

LINALG_OBJS   = ComplexSparseMatrix.o \
                ComplexSparseSquareMatrix.o \
                ComplexSparseSquareSymmetricMatrix.o \
                ComplexSparseSquareUnsymmetricMatrix.o \
                DoubleSparseMatrix.o \
                DoubleSparseSquareMatrix.o \
                DoubleSparseSquareSymmetricMatrix.o \
                DoubleSparseSquareUnsymmetricMatrix.o

SOLVER_OBJS   = PARDISOSolver.o \
                PARDISOSolverComplex.o \
                PARDISOSolverDouble.o

OBJS          = $(addprefix $(OBJ_DIR)/,$(CORE_OBJS) \
                $(CONTROL_OBJS) \
                $(IO_OBJS) \
                $(MESH_OBJS) \
                $(FORWARD_OBJS) \
                $(INVERSION_OBJS) \
                $(MODEL_OBJS) \
                $(DATA_OBJS) \
                $(LINALG_OBJS) \
                $(SOLVER_OBJS))
DEPS          = $(OBJS:.o=.d)
PROGRAM       = femtic-dabic
CUTBACK_TEST_PROGRAM = femtic-dabic-cutback-test
CUTBACK_TEST_OBJ_DIR = build/cutback-test

.PHONY: all release debug cutback-test clean print-config check-env

all:            $(PROGRAM)

release:        all

# default: release mode
$(PROGRAM):     $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(LDFLAGS) $(LIBS) -o $(PROGRAM)

debug: CXXFLAGS = $(DEBUG_CXXFLAGS)
debug: $(PROGRAM)

cutback-test:
	$(MAKE) OBJ_DIR=$(CUTBACK_TEST_OBJ_DIR) PROGRAM=$(CUTBACK_TEST_PROGRAM) \
		CXXFLAGS='$(CXXFLAGS) -DFEMTIC_DABIC_TEST_FORCE_FIRST_INEXACT_ABIC_CUTBACK' all

print-config:
	@echo "PROGRAM=$(PROGRAM)"
	@echo "CXX=$(CXX)"
	@echo "CC=$(CC)"
	@echo "MKLROOT=$(MKLROOT)"
	@echo "OBJ_DIR=$(OBJ_DIR)"
	@echo "SRC_DIRS=$(SRC_DIRS)"
	@echo "INCLUDES=$(INCLUDES)"
	@echo "CXXFLAGS=$(CXXFLAGS)"
	@echo "DEBUG_CXXFLAGS=$(DEBUG_CXXFLAGS)"

check-env:
	@command -v $(CXX) >/dev/null 2>&1 || { echo "ERROR: $(CXX) was not found in PATH."; exit 1; }
	@test -d "$(MKLROOT)" || { echo "ERROR: MKLROOT does not exist: $(MKLROOT)"; exit 1; }
	@test -f "$(MKLROOT)/lib/intel64/libmkl_intel_ilp64.so" || { echo "ERROR: MKL ILP64 library not found under $(MKLROOT)."; exit 1; }

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEPFLAGS) -c $< -o $@

clean:
	rm -rf build
	rm -f *.o *.d *~ $(PROGRAM) $(CUTBACK_TEST_PROGRAM)

-include $(DEPS)
