//-------------------------------------------------------------------------------------------------------
// FemticHDF5CalcTypes.h
//
// Plain-old-data row types shared between:
//   - the ObservedData / ObservedDataStation* collector methods (which have
//     no HDF5 or MPI dependency, so they can be called from any translation
//     unit without pulling in hdf5.h/mpi.h), and
//   - OutputHDF5.cpp (which does the actual H5Dwrite calls) and
//     AnalysisControl.cpp (which does the MPI_Gatherv of calculated values
//     across PEs before results_iterX.h5 is written).
//
// Kept deliberately dependency-free (no hdf5.h, no mpi.h, no FEMTIC headers
// beyond nothing) so it is cheap to include from every station header.
//
// HDF5 support by Volker Rath (DIAS; 2026-09-13).
//-------------------------------------------------------------------------------------------------------
#ifndef DBLDEF_FEMTIC_HDF5_CALC_TYPES
#define DBLDEF_FEMTIC_HDF5_CALC_TYPES

// Station/data-type codes used throughout the HDF5 output (results_iterX.h5
// /data group) and in the calculated-value collectors below. MUST stay in
// sync with the component-ordering documented in OutputHDF5.h.
namespace FemticHDF5 {
    const int DTYPE_MT    = 0;
    const int DTYPE_APP   = 1;
    const int DTYPE_HTF   = 2;
    const int DTYPE_VTF   = 3;
    const int DTYPE_PT    = 4;
    const int DTYPE_NMT   = 5;
    const int DTYPE_NMT2  = 6;
    const int DTYPE_NMT2A = 7;
}

// One calculated-response value for one (station, frequency, component).
// Frequencies are partitioned across PEs, so each PE can only fill in the
// rows for the frequencies it actually computed; collectCalculatedValuesFor
// HDF5() on each station class appends only those. AnalysisControl.cpp then
// MPI_Gatherv's these (as raw bytes -- this struct is POD, no pointers) onto
// PE 0 collectively, on every PE, before results_iterX.h5 is written -- see
// the comment above outputResultsToHDF5() in OutputHDF5.h for why this must
// never be done from inside a "PE 0 only" branch.
struct FemticHDF5CalcRow {
    int    site_id;
    int    datatype;   // one of FemticHDF5::DTYPE_*
    double freq;
    int    component;  // meaning depends on datatype -- see OutputHDF5.h
    double cal_re;
    double cal_im;      // 0 for real-valued datatypes (APP, PT)
};

// One row of estimated distortion parameters for one station, mirroring
// ObservedData::outputDistortionParams()'s text output (distortion_iterN.dat)
// so the same numbers can also be embedded in results_iterN.h5's
// /distortion group. Which of param1..param4 are meaningful, and their
// units, depend on the run's distortion type -- see the /distortion/metadata
// "type" attribute (AnalysisControl::TypeOfDistortion) written alongside
// this dataset, and ObservedData::outputDistortionParams() for the
// per-type field meanings.
struct FemticHDF5DistortionRow {
    int    site_id;
    double param1;
    double param2;
    double param3;
    double param4;   // unused (0.0) for ESTIMATE_GAINS_ONLY
    int    isFixed;
};

#endif // DBLDEF_FEMTIC_HDF5_CALC_TYPES
