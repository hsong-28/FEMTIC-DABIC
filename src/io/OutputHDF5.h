//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
// Copyright (c) 2021 Yoshiya Usui
// Modified from Copyright (c) 2025 Han Song
// (HDF5 output extension added 2025-06-23, in femtic_v4_src)
//-------------------------------------------------------------------------------------------------------
// New file: ported from femtic_v4_src/OutputHDF5.h to femtic_dabic_v2.7_src by
// Volker Rath (DIAS) with the help of Claude Sonnet 5, 2026-08-21. Adapted only
// for this tree's per-subdirectory include layout (VPATH covers all of them,
// so include lines are unchanged); no API or on-disk format changes.
// Further modified (outputModelToHDF5 signature changed to take a
// pre-reduced sensitivity array instead of an Inversion* — avoids an
// MPI_Allreduce deadlock; see .cpp changelog and femtic_v4_src fix) by
// Volker Rath (DIAS) with the help of Claude Sonnet 5 (Anthropic), 2026-09-09.
// Further modified (merged model_iterX.h5 + data_iterX.h5 +
// distortion_iterX.dat into one results_iterX.h5, and added calculated
// response values -- cal_re/cal_im -- to /data, gathered across all PEs via
// MPI_Gatherv since the calculation is frequency-partitioned; ported from
// femtic_v4_src) by Volker Rath (DIAS) with the help of Claude Sonnet 5
// (Anthropic), 2026-09-13.
#ifndef DBLDEF_OUTPUT_HDF5
#define DBLDEF_OUTPUT_HDF5

#include <vector>
#include "FemticHDF5CalcTypes.h"

// Write results_iterX.h5 -- combines what used to be three separate
// artifacts (model_iterX.h5, data_iterX.h5, distortion_iterX.dat) into one
// file, as sibling top-level groups. distortion_iterX.dat itself is STILL
// written unchanged by ObservedData::outputDistortionParams() (human-
// readable / back-compat) -- /distortion just carries the same numbers,
// structured for HDF5 tools.
//
//   /model/metadata            – scalar attributes: nElem, nBlocks, nNodes, iterNum
//   /model/element_block_map   – int[nElem]  : blockID of each element
//   /model/blocks              – compound[nBlocks]: blockID, resistivity, rho_min,
//                                 rho_max, weight, type
//   /model/mesh/node_coords    – double[nNodes][3]: X, Y, Z of every mesh node
//   /model/mesh/elem_nodes     – int[nElem][nNodesPerElem]: node IDs composing each element
//   /model/sensitivity/raw               – double[nBlocks] (only when sensitivity was computed this iter)
//   /model/sensitivity/volume_normalised – double[nBlocks]
//
//   /data/metadata             – scalar attributes: nRows, iterNum, nMissingCalc
//   /data/data                 – compound[nRows]: freq, datatype, site_id,
//                                 site_x, site_y, site_z,
//                                 re_val, im_val, re_err, im_err,
//                                 cal_re, cal_im, component
//
//   /distortion/metadata       – scalar attributes: iterNum, nRows, type
//                                 (type = AnalysisControl::TypeOfDistortion of this run)
//   /distortion/params         – compound[nRows]: site_id, param1..param4, isFixed
//                                 (only written when distortion estimation is
//                                 enabled for this run -- see
//                                 AnalysisControl::estimateDistortionMatrix())
//
// datatype codes (int):
//   0=MT(Z), 1=APP_RES_AND_PHS, 2=HTF, 3=VTF, 4=PT, 5=NMT, 6=NMT2, 7=NMT2_APP_RES_AND_PHS
//
// component codes (int) per datatype:
//   MT / NMT2 :  0=Zxx  1=Zxy  2=Zyx  3=Zyy
//   APP / NMT2APP: 0=rhoXX 1=rhoXY 2=rhoYX 3=rhoYY 4=phsXX 5=phsXY 6=phsYX 7=phsYY
//   HTF :        0=Txx  1=Txy  2=Tyx  3=Tyy
//   VTF :        0=Tzx  1=Tzy
//   PT  :        0=PTxx 1=PTxy 2=PTyx 3=PTyy
//   NMT :        0=Yx   1=Yy
//
// For real-valued quantities (APP_RES_AND_PHS, PT) im_val = 0, im_err = 0,
// and cal_im = 0. cal_re/cal_im are NaN only if no PE reported a calculated
// value for that row, which should not happen in a normal run (see
// /data/metadata attr "nMissingCalc" and the WARNING logged by
// writeDataGroup() in OutputHDF5.cpp if it ever is nonzero).
//
// param1..4 under /distortion/params depend on the run's distortion type:
//   ESTIMATE_DISTORTION_MATRIX_DIFFERENCE: param1..4 = Cxx, Cxy, Cyx, Cyy
//   ESTIMATE_GAINS_AND_ROTATIONS:          param1..4 = ExGain, EyGain,
//                                           ExRotation(deg), EyRotation(deg)
//   ESTIMATE_GAINS_ONLY:                   param1..2 = ExGain, EyGain;
//                                           param3..4 unused (0.0)
//
// iterNum: current iteration number
// sensitivityScalarValuesReduced: pointer to an ALREADY MPI-reduced (summed
//             over all PEs) array of length numModel, or NULL if sensitivity
//             is unavailable/not requested this iteration.
//   IMPORTANT (fixed 2026-09-09): the /model-writing part of this function
//   only ever runs on PE 0 (see call site), so it must NEVER itself call an
//   MPI collective such as MPI_Allreduce — doing so previously deadlocked
//   the whole run, since PE 0 would block waiting for other PEs that never
//   reached a matching call. The reduction must be performed by ALL PEs
//   *before* branching into the PE-0-only block that calls this function;
//   the caller passes in the already-reduced result.
// calcRowsAll: the FULL, cross-PE-merged list of calculated response values
//   (added 2026-09-13). Calculation is frequency-partitioned across PEs, so
//   -- exactly as with sensitivityScalarValuesReduced above -- this MUST be
//   the result of an MPI_Gatherv performed by the caller on EVERY PE (see
//   ObservedData::collectCalculatedValuesForHDF5() and
//   FemticHDF5CalcTypes.h), never assembled from PE 0's own data alone.
//   Added datasets when sensitivityScalarValuesReduced != NULL:
//   /model/sensitivity/raw              – double[nBlocks]  sum|J[:,imdl]| per block (0 for fixed)
//   /model/sensitivity/volume_normalised – double[nBlocks]  raw / block_volume (m^-3)
#ifdef _HDF5_OUT
void outputResultsToHDF5( const int iterNum,
                           const double* sensitivityScalarValuesReduced,
                           const std::vector<FemticHDF5CalcRow>& calcRowsAll );
#endif // _HDF5_OUT

#ifdef _HDF5_JAC
// Write exchange.h5 -- Jacobian, roughening matrix and mesh in ONE file
// (2026-10-01; replaces jacobian.h5, rough.h5 and mesh.h5):
//
//   /metadata                attrs: iterNum, exchangeVersion (=1)
//
//   /jacobian/metadata       attrs: iterNum, nData, nModel,
//                                   weighted (=1, J already divided by SD)
//   /jacobian/values         double[nData][nModel], row=datum col=model param
//                            (the SD-weighted sensitivity matrix Cd^{-1/2} dF/dm)
//   /jacobian/data_errors    double[nData], SD denominator of each datum row
//                            (same ordering as the Jacobian rows)
//
//   /rough/metadata          attrs: nRows, nNonZeros, format="CSR"
//   /rough/row_ptr           int[nRows+1]
//   /rough/col_ind           int[nNonZeros]
//   /rough/values            double[nNonZeros]
//                            Square [nModel x nModel] roughening matrix,
//                            nModel = number of resistivity blocks.
//
//   /mesh/metadata           attrs: meshType (0=HEXA, 1=TETRA, 2=NONCONFORMING_HEXA),
//                                   nNodes, nElem, nNodesPerElem,
//                                   neighborFormat (0=dense, 1=CSR),
//                                   nNeighborElem (only when neighborFormat==0)
//   /mesh/node_coords        double[nNodes][3]
//   /mesh/elem_nodes         int[nElem][nNodesPerElem]
//   /mesh/neighbor_elements  neighborFormat==0: int[nElem][nNeighborElem]
//                            neighborFormat==1: int[nnz], flat CSR values
//   /mesh/neighbor_face_ptr  (neighborFormat==1 only) int[nElem*6+1]
//
// Written whenever (and only when) the Jacobian is written: called from
// Inversion::assembleAndWriteJacobianToHDF5() (and the legacy trade-off
// search paths) on PE 0 after the full Jacobian is assembled. The mesh and
// roughening matrix are fetched from AnalysisControl / ResistivityBlock, so
// no extra arguments are needed. Fixed filename, truncated on each call.
// Needs only _HDF5_JAC (independent of _HDF5_OUT).
//
// sensitivityMatrix: row-major double[nData * nModel] on PE 0.
// dataErrorsGlobal:  double[nData] on PE 0, SD values.
void outputJacobianToHDF5( const int iterNum,
                           const int numDataTotal,
                           const int numModel,
                           const double* sensitivityMatrix,
                           const double* dataErrorsGlobal );
#endif // _HDF5_JAC

#endif // DBLDEF_OUTPUT_HDF5
