# HDF5 support by Volker Rath (DIAS; 2026-08-05 to 2026-10-02).
# Single build entry point: make [flags].
# ---------------------------------------------------------------------------
# Intel oneAPI toolchain (oneAPI 2024.0 or later): the compiler-agnostic MPI
# wrapper scripts mpiicpx/mpiicc invoke the LLVM-based icpx/icx directly.
#
# Usage:
#   make [flags]
# ---------------------------------------------------------------------------

CXX           = mpiicpx
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
                -fno-omit-frame-pointer -Wall -Wextra -Wpedantic \
                -traceback \
                -qopenmp \
                -D_LINUX \
                -DMKL_ILP64 \
                -D_USE_OMP \
		-diag-disable=10441

# ---------------------------------------------------------------------------
# Optional sanitizer instrumentation (use together with the debug target):
#   make debug SANITIZE=yes
# Adds AddressSanitizer + UndefinedBehaviorSanitizer. Catches the classic
# causes of an "unspecified segfault" — out-of-bounds writes into the
# BlockRow/DataRow vectors, use-after-free, mismatched array sizes between
# nElem/nNodes/nNPE and what's actually allocated, etc. Slower and uses
# more memory, so use it on a small/single-node test case first.
# Note: MKL's static libs are usually fine under ASan, but if you see
# spurious ASan reports from inside MKL/MPI internals, wrap them with
# ASAN_OPTIONS=verify_asan_link_order=0 at runtime.
# ---------------------------------------------------------------------------
SANITIZE ?= no
ifeq ($(SANITIZE),yes)
    DEBUG_CXXFLAGS += -fsanitize=address,undefined -fno-sanitize-recover=all
    LDFLAGS        += -fsanitize=address,undefined
endif

# icpx/icx use their own bundled libstdc++ headers by default; on systems
# where the module-loaded/system GCC is a different major version than the
# one oneAPI was validated against, --gcc-install-dir tells icpx/icx which
# GCC installation's headers and libs to use for compatibility.
#
# No flag is passed by default; set GCC_INSTALL_DIR=/path/to/gcc/version-dir
# (e.g. /usr/lib/gcc/x86_64-linux-gnu/13) to pass --gcc-install-dir.
GCC_INSTALL_DIR ?=
ifneq ($(strip $(GCC_INSTALL_DIR)),)
CXXFLAGS       += --gcc-install-dir=$(GCC_INSTALL_DIR)
DEBUG_CXXFLAGS += --gcc-install-dir=$(GCC_INSTALL_DIR)
endif

# Optional runtime remapping of hard-coded input file names via "inputfiles.dat"
# (see io/InputFileMap.h). Default: disabled. Enable with INPUT_FILE_MAP=yes.
INPUT_FILE_MAP ?= no
ifeq ($(INPUT_FILE_MAP),yes)
CXXFLAGS       += -D_INPUT_FILE_MAP
DEBUG_CXXFLAGS += -D_INPUT_FILE_MAP
endif

# ---------------------------------------------------------------------------
# Optional HDF5 outputs — two independent flags, each defaulting to 'no'.
# Ported from femtic_v4_src/femtic_v5_src (2026-08-21); reduced from four
# flags to two on 2026-10-01.
#
#   HDF5_OUT   = yes  →  -D_HDF5_OUT    write results_iterX.h5 (model+data+distortion)
#   HDF5_JAC   = yes  →  -D_HDF5_JAC    write exchange.h5 (Jacobian + roughening matrix
#                                       + mesh in one file, rewritten each time the
#                                       Jacobian is written)
#
# The flags are independent; OutputHDF5.o is added to OBJS whenever either is set.
#
# Since 2026-10-02 these flags only decide whether the HDF5 support is compiled
# and linked in. Whether a file is actually written in a given run is chosen in
# control.dat with the keywords ACTIVATE_HDF5_RESULTS (needs HDF5_OUT=yes) and
# ACTIVATE_HDF5_EXCHANGE (needs HDF5_JAC=yes); both default to off. Asking for a
# feature the executable was not built with stops the run with an error.
# (HDF5_ROUGH / HDF5_MESH no longer exist: rough and mesh are part of exchange.h5.
# Passing either as 'yes' prints a warning and enables HDF5_JAC instead.)
#
# Usage examples:
#   make HDF5_OUT=yes
#   make HDF5_OUT=yes HDF5_JAC=yes
#   make HDF5_JAC=yes
#   make                                    # no HDF5, no extra deps
#
# HDF5 root auto-detection order (first match wins):
#   1. HDF5_ROOT   set on command line or in environment
#   2. HDF5_DIR    Cray / NERSC systems
#   3. HDF5HOME    some MPI-IO stacks
#   4. EBROOTHDF5  EasyBuild / Lmod (ICHEC Kay, DKRZ Levante, ...)
#   5. pkg-config  workstations / conda fallback
#
# On HPC clusters 'module load HDF5' typically sets one of the above.
# ---------------------------------------------------------------------------

HDF5_OUT   ?= no
HDF5_JAC   ?= no

# Retired flags (2026-10-01): rough.h5 / mesh.h5 are now groups of exchange.h5.
ifeq ($(HDF5_ROUGH),yes)
    $(warning HDF5_ROUGH is obsolete: the roughening matrix is part of exchange.h5; enabling HDF5_JAC)
    HDF5_JAC := yes
endif
ifeq ($(HDF5_MESH),yes)
    $(warning HDF5_MESH is obsolete: the mesh is part of exchange.h5; enabling HDF5_JAC)
    HDF5_JAC := yes
endif

# Determine whether any HDF5 feature is requested
HDF5_ANY := no
ifneq ($(HDF5_OUT),no)
    HDF5_ANY := yes
endif
ifneq ($(HDF5_JAC),no)
    HDF5_ANY := yes
endif

HDF5_STRING =
ifeq ($(HDF5_ANY),yes)
	HDF5_STRING := _h5
endif

# Build a name suffix reflecting active features, e.g. "_h5_results_exchange"
# HDF5_OUT adds _results; HDF5_JAC adds _exchange. Preprocessor names are unchanged.
ifeq ($(HDF5_OUT),yes)
    HDF5_STRING := $(HDF5_STRING)_results
endif
ifeq ($(HDF5_JAC),yes)
    HDF5_STRING := $(HDF5_STRING)_exchange
endif


# Build the -D flags reflecting active features; applied to both release
# and debug flag sets, matching this Makefile's INPUT_FILE_MAP convention.
ifeq ($(HDF5_OUT),yes)
    CXXFLAGS       += -D_HDF5_OUT
    DEBUG_CXXFLAGS += -D_HDF5_OUT
endif
ifeq ($(HDF5_JAC),yes)
    CXXFLAGS       += -D_HDF5_JAC
    DEBUG_CXXFLAGS += -D_HDF5_JAC
endif

# HDF5_OBJS is set here (compile-time deps); library resolution (LIBS/-L/-l)
# is deferred until after the main LIBS definition below, since LIBS is
# reassigned with plain '=' later and would otherwise wipe out an earlier
# '+=' from this block (see "Resolve HDF5 library" block below).
ifeq ($(HDF5_ANY),yes)
    HDF5_OBJS = OutputHDF5.o
else
    HDF5_OBJS =
endif

DEST          = $(HOME)/bin
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

# Resolve HDF5 include/library flags when any HDF5 feature is enabled.
# Must come AFTER the main LIBS assignment above: LIBS is set with plain
# '=' there, which would otherwise wipe out an earlier '+=' from this block.
ifeq ($(HDF5_ANY),yes)
    ifndef HDF5_ROOT
        ifdef HDF5_DIR
            HDF5_ROOT := $(HDF5_DIR)
        else ifdef HDF5HOME
            HDF5_ROOT := $(HDF5HOME)
        else ifdef EBROOTHDF5
            HDF5_ROOT := $(EBROOTHDF5)
        endif
    endif

    ifdef HDF5_ROOT
        CXXFLAGS       += -I$(HDF5_ROOT)/include
        DEBUG_CXXFLAGS += -I$(HDF5_ROOT)/include
        LIBS           += -L$(HDF5_ROOT)/lib -lhdf5
    else
        CXXFLAGS       += $(shell pkg-config --cflags hdf5 2>/dev/null)
        DEBUG_CXXFLAGS += $(shell pkg-config --cflags hdf5 2>/dev/null)
        LIBS           += $(shell pkg-config --libs   hdf5 2>/dev/null)
    endif
endif

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
                InputFileMap.o \
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
                $(SOLVER_OBJS) \
                $(HDF5_OBJS))
DEPS          = $(OBJS:.o=.d)

PROGRAM       = femtic-dabic
ifeq ($(HDF5_ANY),yes)
PROGRAM       = femtic-dabic$(HDF5_STRING).x
endif

CUTBACK_TEST_PROGRAM = femtic-dabic-cutback-test
CUTBACK_TEST_OBJ_DIR = build/cutback-test

.PHONY: all release debug cutback-test clean print-config check-env

all:            $(PROGRAM)
install:	$(PROGRAM)
	install -m 755 $(PROGRAM) $(DEST)

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
	@echo "GCC_INSTALL_DIR=$(GCC_INSTALL_DIR)"
	@echo "MKLROOT=$(MKLROOT)"
	@echo "OBJ_DIR=$(OBJ_DIR)"
	@echo "SRC_DIRS=$(SRC_DIRS)"
	@echo "INCLUDES=$(INCLUDES)"
	@echo "INPUT_FILE_MAP=$(INPUT_FILE_MAP)"
	@echo "HDF5_OUT=$(HDF5_OUT)"
	@echo "HDF5_JAC=$(HDF5_JAC)"
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

# ---------------------------------------------------------------------------
# Rebuild-on-flag-change guard (2026-09-10).
#
# The dependency files generated by -MMD/-MP (included via "-include
# $(DEPS)" below) track header-file changes correctly, but NOT changes to
# preprocessor macros passed on the `make` command line: they only record
# which files were #included, not which -D flags were active when a .o was
# compiled. So a flag such as -D_HDF5_OUT or -D_HDF5_JAC changing between
# invocations -- e.g. `make HDF5_OUT=yes` run right after
# `make HDF5_JAC=yes` with no `make clean` in between -- is invisible to
# both the mtime check and the header-dependency check: no source or
# header changed, so every .o in build/obj/ is considered up to date and
# silently reused with the OLD flags. The newly requested feature then
# never actually gets compiled in, even though the build reports success.
# This is almost certainly why a newly enabled HDF5_* output (e.g.
# mesh.h5) can appear to silently never be written.
#
# Fix: record the flag values that select #ifdef branches (plus whether
# the "debug" goal was requested, since CXXFLAGS is only swapped to
# DEBUG_CXXFLAGS for that target) in .cxxflags. If they differ from the
# previous build, rewrite the file, which is then newer than every .o,
# forcing GNU make to rebuild everything. Placed after "all"/"$(PROGRAM)"
# above (both already appear earlier in this file, and "all" is also
# declared .PHONY) so the default goal is unaffected.
# ---------------------------------------------------------------------------
FLAGS_SIGNATURE := $(HDF5_OUT)-$(HDF5_JAC)-$(INPUT_FILE_MAP)-$(SANITIZE)-$(filter debug,$(MAKECMDGOALS))

.PHONY: FORCE
.cxxflags: FORCE
	@if [ ! -f $@ ] || [ "$$(cat $@ 2>/dev/null)" != "$(FLAGS_SIGNATURE)" ]; then \
		echo "$(FLAGS_SIGNATURE)" > $@ ; \
	fi

$(OBJS): .cxxflags

clean:
	rm -rf build
	rm -f *.o *.d *~ $(PROGRAM) $(CUTBACK_TEST_PROGRAM) .cxxflags

-include $(DEPS)
