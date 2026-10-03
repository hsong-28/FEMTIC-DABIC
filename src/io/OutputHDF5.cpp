//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
// Original FEMTIC source:
// Copyright (c) 2021 Yoshiya Usui
//
// FEMTIC-DABIC modifications and extensions:
// Copyright (c) 2025-2026 Han Song
//
// HDF5 support by Volker Rath (DIAS; 2026-08-21 to 2026-10-01).
// Writes are checked and staged before publication; see OutputHDF5.h for the schema.
//-------------------------------------------------------------------------------------------------------
#include "OutputHDF5.h"

#include <hdf5.h>
#include <sstream>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <limits>
#include <utility>
#include <cstdio>
#include <cerrno>
#include <stdexcept>

#include "AnalysisControl.h"
#include "ResistivityBlock.h"
#include "ObservedData.h"
#include "MeshData.h"
#include "MeshDataNonConformingHexaElement.h"
#include "OutputFiles.h"
#include "CommonParameters.h"
#include "FemticHDF5CalcTypes.h"

// Station headers – observed data members are exposed via the inline getters
// added to each station header (see v4_/v5_ prefixed header files).
#include "ObservedDataStationMT.h"
#include "ObservedDataStationApparentResistivityAndPhase.h"
#include "ObservedDataStationHTF.h"
#include "ObservedDataStationVTF.h"
#include "ObservedDataStationPT.h"
#include "ObservedDataStationNMT.h"
#include "ObservedDataStationNMT2.h"
#include "ObservedDataStationNMT2ApparentResistivityAndPhase.h"
#include "Inversion.h"
#ifdef _HDF5_JAC
#include "RougheningSquareMatrix.h"
#endif

//===========================================================================
// Internal helpers
//===========================================================================
namespace {

// Optional exports must not terminate one MPI rank on an I/O failure.
static void hdf5check( hid_t id, const char* msg )
{
    if( id < 0 ) throw std::runtime_error(msg);
}

// Release every local HDF5 object during unwinding; check normal closes too.
class HdfHandle {
public:
    HdfHandle() : id(-1), closer(NULL) {}
    HdfHandle( hid_t value, herr_t (*closeFunction)(hid_t) ) : id(value), closer(closeFunction) {
        hdf5check(id, "HDF5 object creation");
    }
    HdfHandle( HdfHandle&& other ) noexcept : id(other.id), closer(other.closer) { other.id = -1; }
    HdfHandle& operator=( HdfHandle&& other ) {
        close();
        id = other.id; closer = other.closer; other.id = -1;
        return *this;
    }
    ~HdfHandle() {
        if( id >= 0 && closer(id) < 0 )
            OutputFiles::m_logFile << "# Warning: HDF5 cleanup close failed." << std::endl;
    }
    operator hid_t() const { return id; }
    void close() {
        if( id >= 0 ) {
            hdf5check(closer(id), "HDF5 object close");
            id = -1;
        }
    }
    HdfHandle( const HdfHandle& ) = delete;
    HdfHandle& operator=( const HdfHandle& ) = delete;
private:
    hid_t id;
    herr_t (*closer)(hid_t);
};

// Publish only a fully written and closed file. Never truncate the old export.
template <typename Writer>
static bool writeHdfFile( const std::string& filename, int iteration, Writer writer )
{
    const std::string temporary = filename + ".tmp";
    bool created = false;
    try {
        HdfHandle file(H5Fcreate(temporary.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
        created = true;
        writer(file);
        file.close();
        if( std::rename(temporary.c_str(), filename.c_str()) != 0 )
            throw std::runtime_error(std::string("publishing completed file: ") + std::strerror(errno));
        return true;
    } catch( const std::exception& error ) {
        OutputFiles::m_logFile << "# Warning: skipping HDF5 export " << filename
                               << " for iteration " << iteration << " (" << error.what()
                               << "). Any existing final file is unchanged and may be stale; inversion continues." << std::endl;
        if( created && std::remove(temporary.c_str()) != 0 )
            OutputFiles::m_logFile << "# Warning: could not remove incomplete temporary file " << temporary << std::endl;
        return false;
    }
}

// Write a scalar int attribute on an open group/dataset.
static void writeIntAttr( hid_t obj, const char* name, int val )
{
    HdfHandle sp( H5Screate( H5S_SCALAR ), H5Sclose );
    HdfHandle at( H5Acreate2( obj, name, H5T_NATIVE_INT, sp, H5P_DEFAULT, H5P_DEFAULT ), H5Aclose );
    hdf5check( H5Awrite( at, H5T_NATIVE_INT, &val ), "H5Awrite" );
    at.close();
    sp.close();
}

// Compound data row for the /blocks dataset.
struct BlockRow {
    int    blockID;
    double resistivity;
    double rho_min;
    double rho_max;
    double weight;
    int    type;    // ResistivityBlock::FREE_AND_CONSTRAINED etc.
};

// Compound data row for the /data dataset.
struct DataRow {
    double freq;
    int    datatype;
    int    site_id;
    double site_x;
    double site_y;
    double site_z;
    double re_val;
    double im_val;
    double re_err;
    double im_err;
    double cal_re;   // added 2026-09-13 -- see collectCalculatedValuesForHDF5()
    double cal_im;   // 0 for real-valued datatypes (APP, PT); NaN if no PE
                     // reported a calculated value for this row (should not
                     // happen in a normal run -- see outputResultsToHDF5())
    int    component;
};

// /distortion/params rows use FemticHDF5DistortionRow (FemticHDF5CalcTypes.h)
// directly -- HOFFSET() works against any POD struct visible in this
// translation unit, so no local duplicate type is needed here.

// Create the HDF5 compound type for BlockRow.
static HdfHandle makeBlockType()
{
    HdfHandle t( H5Tcreate( H5T_COMPOUND, sizeof(BlockRow) ), H5Tclose );
    hdf5check( H5Tinsert( t, "blockID",     HOFFSET(BlockRow,blockID),     H5T_NATIVE_INT    ), "H5Tinsert" );
    hdf5check( H5Tinsert( t, "resistivity", HOFFSET(BlockRow,resistivity), H5T_NATIVE_DOUBLE ), "H5Tinsert" );
    hdf5check( H5Tinsert( t, "rho_min",     HOFFSET(BlockRow,rho_min),     H5T_NATIVE_DOUBLE ), "H5Tinsert" );
    hdf5check( H5Tinsert( t, "rho_max",     HOFFSET(BlockRow,rho_max),     H5T_NATIVE_DOUBLE ), "H5Tinsert" );
    hdf5check( H5Tinsert( t, "weight",      HOFFSET(BlockRow,weight),      H5T_NATIVE_DOUBLE ), "H5Tinsert" );
    hdf5check( H5Tinsert( t, "type",        HOFFSET(BlockRow,type),        H5T_NATIVE_INT    ), "H5Tinsert" );
    return t;
}

// Create the HDF5 compound type for DataRow.
static HdfHandle makeDataType()
{
    HdfHandle t( H5Tcreate( H5T_COMPOUND, sizeof(DataRow) ), H5Tclose );
    hdf5check( H5Tinsert( t, "freq",      HOFFSET(DataRow,freq),      H5T_NATIVE_DOUBLE ), "H5Tinsert" );
    hdf5check( H5Tinsert( t, "datatype",  HOFFSET(DataRow,datatype),  H5T_NATIVE_INT    ), "H5Tinsert" );
    hdf5check( H5Tinsert( t, "site_id",   HOFFSET(DataRow,site_id),   H5T_NATIVE_INT    ), "H5Tinsert" );
    hdf5check( H5Tinsert( t, "site_x",    HOFFSET(DataRow,site_x),    H5T_NATIVE_DOUBLE ), "H5Tinsert" );
    hdf5check( H5Tinsert( t, "site_y",    HOFFSET(DataRow,site_y),    H5T_NATIVE_DOUBLE ), "H5Tinsert" );
    hdf5check( H5Tinsert( t, "site_z",    HOFFSET(DataRow,site_z),    H5T_NATIVE_DOUBLE ), "H5Tinsert" );
    hdf5check( H5Tinsert( t, "re_val",    HOFFSET(DataRow,re_val),    H5T_NATIVE_DOUBLE ), "H5Tinsert" );
    hdf5check( H5Tinsert( t, "im_val",    HOFFSET(DataRow,im_val),    H5T_NATIVE_DOUBLE ), "H5Tinsert" );
    hdf5check( H5Tinsert( t, "re_err",    HOFFSET(DataRow,re_err),    H5T_NATIVE_DOUBLE ), "H5Tinsert" );
    hdf5check( H5Tinsert( t, "im_err",    HOFFSET(DataRow,im_err),    H5T_NATIVE_DOUBLE ), "H5Tinsert" );
    hdf5check( H5Tinsert( t, "cal_re",    HOFFSET(DataRow,cal_re),    H5T_NATIVE_DOUBLE ), "H5Tinsert" );
    hdf5check( H5Tinsert( t, "cal_im",    HOFFSET(DataRow,cal_im),    H5T_NATIVE_DOUBLE ), "H5Tinsert" );
    hdf5check( H5Tinsert( t, "component", HOFFSET(DataRow,component), H5T_NATIVE_INT    ), "H5Tinsert" );
    return t;
}

// Create the HDF5 compound type for FemticHDF5DistortionRow.
static HdfHandle makeDistortionType()
{
    HdfHandle t( H5Tcreate( H5T_COMPOUND, sizeof(FemticHDF5DistortionRow) ), H5Tclose );
    hdf5check( H5Tinsert( t, "site_id", HOFFSET(FemticHDF5DistortionRow,site_id), H5T_NATIVE_INT    ), "H5Tinsert" );
    hdf5check( H5Tinsert( t, "param1",  HOFFSET(FemticHDF5DistortionRow,param1),  H5T_NATIVE_DOUBLE ), "H5Tinsert" );
    hdf5check( H5Tinsert( t, "param2",  HOFFSET(FemticHDF5DistortionRow,param2),  H5T_NATIVE_DOUBLE ), "H5Tinsert" );
    hdf5check( H5Tinsert( t, "param3",  HOFFSET(FemticHDF5DistortionRow,param3),  H5T_NATIVE_DOUBLE ), "H5Tinsert" );
    hdf5check( H5Tinsert( t, "param4",  HOFFSET(FemticHDF5DistortionRow,param4),  H5T_NATIVE_DOUBLE ), "H5Tinsert" );
    hdf5check( H5Tinsert( t, "isFixed", HOFFSET(FemticHDF5DistortionRow,isFixed), H5T_NATIVE_INT    ), "H5Tinsert" );
    return t;
}

// Look up r's calculated value in calcLookup (keyed on datatype/site_id/
// freq/component) and fill r.cal_re/r.cal_im; increments *nMissingCalc and
// leaves them NaN if this row has no corresponding gathered calc value
// (should not happen in a normal run -- see outputResultsToHDF5()).
static void lookupCalc( const std::map< std::vector<double>, std::pair<double,double> >& calcLookup,
                         DataRow& r, int* nMissingCalc )
{
    std::vector<double> key(4);
    key[0] = (double)r.datatype; key[1] = (double)r.site_id;
    key[2] = r.freq;             key[3] = (double)r.component;
    std::map< std::vector<double>, std::pair<double,double> >::const_iterator itc = calcLookup.find(key);
    if( itc != calcLookup.end() ){
        r.cal_re = itc->second.first;
        r.cal_im = itc->second.second;
    } else {
        r.cal_re = std::numeric_limits<double>::quiet_NaN();
        r.cal_im = std::numeric_limits<double>::quiet_NaN();
        ++(*nMissingCalc);
    }
}

// Datatype code constants (must match header comment) now live in
// FemticHDF5CalcTypes.h (namespace FemticHDF5) so the station-class
// collectors and this file agree on them; pull them in unqualified here
// since every reference below predates that header (DTYPE_MT etc.).
using namespace FemticHDF5;

} // anonymous namespace

#ifdef _HDF5_OUT   // results_iterN.h5 writer; exchange.h5 (below) needs only _HDF5_JAC
//===========================================================================
// writeModelGroup -- writes the /model group of results_iterN.h5
//===========================================================================
// Formerly outputModelToHDF5(), which wrote its own model_iterN.h5. Now
// writes into a "/model" group of an already-open file (created and closed
// by the caller, outputResultsToHDF5()) so model/data/distortion all land
// in one results_iterN.h5 -- see OutputHDF5.h. Still PE-0-only and still
// must never itself call an MPI collective (see header comment /
// 2026-09-09 deadlock fix, unchanged by this reorganisation).
static void writeModelGroup( hid_t fid, const int iterNum, const double* sensitivityScalarValuesReduced )
{
    const AnalysisControl* const pAC = AnalysisControl::getInstance();

    HdfHandle grpModel( H5Gcreate2( fid, "/model", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ), H5Gclose );

    //------------------------------------------------------------------
    // /model/metadata  (group with scalar attributes)
    //------------------------------------------------------------------
    const MeshData* const pMesh = pAC->getPointerOfMeshData();
    const ResistivityBlock* const pRB = ResistivityBlock::getInstance();

    const int nElem   = pMesh->getNumElemTotal();
    const int nNodes  = pMesh->getNumNodeTotal();
    const int nBlocks = pRB->getNumResistivityBlockTotal();

    HdfHandle grpMeta( H5Gcreate2( grpModel, "metadata", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ), H5Gclose );
    writeIntAttr( grpMeta, "iterNum", iterNum );
    writeIntAttr( grpMeta, "nElem",   nElem   );
    writeIntAttr( grpMeta, "nNodes",  nNodes  );
    writeIntAttr( grpMeta, "nBlocks", nBlocks );
    grpMeta.close();

    //------------------------------------------------------------------
    // /model/element_block_map  – int[nElem]
    //------------------------------------------------------------------
    {
        std::vector<int> ebmap( nElem );
        for( int i = 0; i < nElem; ++i )
            ebmap[i] = pRB->getBlockIDFromElemID(i);

        hsize_t dims[1] = { (hsize_t)nElem };
        HdfHandle sp( H5Screate_simple( 1, dims, NULL ), H5Sclose );
        HdfHandle ds( H5Dcreate2( grpModel, "element_block_map", H5T_NATIVE_INT,
                                 sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ), H5Dclose );
        hdf5check( H5Dwrite( ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, ebmap.data() ), "H5Dwrite" );
        ds.close(); sp.close();
    }

    //------------------------------------------------------------------
    // /model/blocks  – compound[nBlocks]
    //------------------------------------------------------------------
    {
        std::vector<BlockRow> rows( nBlocks );
        for( int i = 0; i < nBlocks; ++i ){
            rows[i].blockID     = i;
            rows[i].resistivity = pRB->getResistivityValuesFromBlockID(i);
            rows[i].rho_min     = pRB->getResistivityValuesMinFromBlockID(i);
            rows[i].rho_max     = pRB->getResistivityValuesMaxFromBlockID(i);
            rows[i].weight      = pRB->getWeightingConstantFromBlockID(i);
            rows[i].type        = pRB->getTypeOfResistivityBlockHDF5(i);
        }

        hsize_t dims[1] = { (hsize_t)nBlocks };
        HdfHandle memType = makeBlockType();
        HdfHandle sp( H5Screate_simple( 1, dims, NULL ), H5Sclose );
        HdfHandle ds( H5Dcreate2( grpModel, "blocks", memType, sp,
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ), H5Dclose );
        hdf5check( H5Dwrite( ds, memType, H5S_ALL, H5S_ALL, H5P_DEFAULT, rows.data() ), "H5Dwrite" );
        ds.close(); sp.close(); memType.close();
    }

    //------------------------------------------------------------------
    // /model/mesh/node_coords  – double[nNodes][3]
    //------------------------------------------------------------------
    HdfHandle grpMesh( H5Gcreate2( grpModel, "mesh", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ), H5Gclose );
    {
        std::vector<double> coords( nNodes * 3 );
        for( int n = 0; n < nNodes; ++n ){
            coords[ n*3 + 0 ] = pMesh->getXCoordinatesOfNodes(n);
            coords[ n*3 + 1 ] = pMesh->getYCoordinatesOfNodes(n);
            coords[ n*3 + 2 ] = pMesh->getZCoordinatesOfNodes(n);
        }
        hsize_t dims[2] = { (hsize_t)nNodes, 3 };
        HdfHandle sp( H5Screate_simple( 2, dims, NULL ), H5Sclose );
        HdfHandle ds( H5Dcreate2( grpMesh, "node_coords", H5T_NATIVE_DOUBLE,
                                sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ), H5Dclose );
        hdf5check( H5Dwrite( ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, coords.data() ), "H5Dwrite" );
        ds.close(); sp.close();
    }

    //------------------------------------------------------------------
    // /model/mesh/elem_nodes  – int[nElem][nNodesPerElem]
    //------------------------------------------------------------------
    {
        // Determine nodes-per-element from first element.
        // MeshData::getNumNodePerElem() is available in both v4 and v5.
        const int nNPE = pMesh->getNumNodePerElement();
        std::vector<int> enodes( nElem * nNPE );
        for( int e = 0; e < nElem; ++e )
            for( int k = 0; k < nNPE; ++k )
                enodes[ e*nNPE + k ] = pMesh->getNodesOfElements(e,k);

        hsize_t dims[2] = { (hsize_t)nElem, (hsize_t)nNPE };
        HdfHandle sp( H5Screate_simple( 2, dims, NULL ), H5Sclose );
        HdfHandle ds( H5Dcreate2( grpMesh, "elem_nodes", H5T_NATIVE_INT,
                                sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ), H5Dclose );
        hdf5check( H5Dwrite( ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, enodes.data() ), "H5Dwrite" );
        ds.close(); sp.close();
    }

    grpMesh.close();

    //------------------------------------------------------------------
    // /model/sensitivity  (only when a pre-reduced sensitivity array was passed in)
    //------------------------------------------------------------------
#ifdef _HDF5_OUT
    if( sensitivityScalarValuesReduced != nullptr ){
        // NOTE: sensitivityScalarValuesReduced must already be the
        // MPI-reduced (summed over all PEs) array — this function runs on
        // PE 0 only and must not perform any MPI collective itself (see
        // header comment / 2026-09-09 deadlock fix).
        const double* const sensReduced = sensitivityScalarValuesReduced;

        const int nBlocks2 = pRB->getNumResistivityBlockTotal();
        std::vector<double> sensRaw( nBlocks2, 1.0e-20 );
        std::vector<double> sensVol( nBlocks2, 1.0e-20 );
        const double criteria = 1.0e-20;

        for( int iblk = 0; iblk < nBlocks2; ++iblk ){
            if( pRB->isFixedResistivityValue( iblk ) ) continue;
            const int    imdl = pRB->getModelIDFromBlockID( iblk );
            const double raw  = std::fabs( sensReduced[imdl] );
            const double vol  = pRB->calcVolumeOfBlock( iblk );
            sensRaw[iblk] = (raw > criteria) ? raw : criteria;
            sensVol[iblk] = (raw > criteria) ? raw / vol : criteria;
        }

        HdfHandle grpSens( H5Gcreate2( grpModel, "sensitivity",
                                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ), H5Gclose );
        hsize_t dimB = (hsize_t)nBlocks2;
        {
            HdfHandle sp( H5Screate_simple( 1, &dimB, NULL ), H5Sclose );
            HdfHandle ds( H5Dcreate2( grpSens, "raw", H5T_NATIVE_DOUBLE,
                                    sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ), H5Dclose );
            hdf5check( H5Dwrite( ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, sensRaw.data() ), "H5Dwrite" );
            ds.close(); sp.close();
        }
        {
            HdfHandle sp( H5Screate_simple( 1, &dimB, NULL ), H5Sclose );
            HdfHandle ds( H5Dcreate2( grpSens, "volume_normalised", H5T_NATIVE_DOUBLE,
                                    sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ), H5Dclose );
            hdf5check( H5Dwrite( ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, sensVol.data() ), "H5Dwrite" );
            ds.close(); sp.close();
        }
        grpSens.close();
    }
#endif // _HDF5_OUT

    grpModel.close();

    OutputFiles::m_logFile << "# Written HDF5 /model group (iter " << iterNum << ")" << std::endl;
}

//===========================================================================
// writeDataGroup -- writes the /data group of results_iterN.h5
//===========================================================================
// Formerly outputDataToHDF5(), which wrote its own data_iterN.h5. Now
// writes into a "/data" group of an already-open file -- see
// writeModelGroup() above and outputResultsToHDF5() below.
//
// calcRowsAll: the FULL, cross-PE-merged list of calculated response values
// (already MPI_Gatherv'd onto PE 0 by the caller -- see AnalysisControl.cpp
// and FemticHDF5CalcTypes.h). This function only runs on PE 0 and must not
// itself perform any MPI collective, same rule as writeModelGroup().
static void writeDataGroup( hid_t fid, const int iterNum, const std::vector<FemticHDF5CalcRow>& calcRowsAll )
{
    // Collect all rows first, then write in one shot.
    std::vector<DataRow> rows;
    rows.reserve( 4096 );

    // Build a lookup from (datatype, site_id, freq, component) to the
    // calculated value gathered for that datum. Keyed on the exact double
    // frequency value: every frequency ultimately comes from the same
    // per-station m_freq array read from observe.dat by every PE, so the
    // bit patterns compared here always originate from the same read, never
    // from two independent computations that could differ by rounding.
    std::map< std::vector<double>, std::pair<double,double> > calcLookup;
    // NOTE: std::map<std::vector<double>,...> is a convenient way to get a
    // 4-field (datatype, site_id, freq, component) key with an ordering
    // comparator "for free" from std::vector's operator<, without writing a
    // custom tuple comparator; nRows is at most a few 10^5 for any
    // realistic FEMTIC run, so the log-n lookup cost here is negligible
    // next to the HDF5 I/O this function already does.
    for( std::size_t i = 0; i < calcRowsAll.size(); ++i ){
        const FemticHDF5CalcRow& c = calcRowsAll[i];
        std::vector<double> key(4);
        key[0] = (double)c.datatype; key[1] = (double)c.site_id;
        key[2] = c.freq;             key[3] = (double)c.component;
        calcLookup[key] = std::make_pair( c.cal_re, c.cal_im );
    }
    int nMissingCalc = 0;
    const double NaN = std::numeric_limits<double>::quiet_NaN();

    const ObservedData* const pOD = ObservedData::getInstance();

    //----------------------------------------------------------------------
    // MT stations  (Zxx, Zxy, Zyx, Zyy)
    //----------------------------------------------------------------------
    {
        const int nSta = pOD->getNumStationsMT();
        for( int i = 0; i < nSta; ++i ){
            const ObservedDataStationMT& sta = pOD->getStationMT(i);
            const int sid   = sta.getStationID();
            const double sx = sta.getLocationOfPoint().X;
            const double sy = sta.getLocationOfPoint().Y;
            const double sz = sta.getZCoordOfPoint();
            const int nf    = sta.getTotalNumberOfFrequency();

            // Component order: 0=Zxx 1=Zxy 2=Zyx 3=Zyy
            for( int ifreq = 0; ifreq < nf; ++ifreq ){
                const double freq = sta.getFrequencyValues(ifreq);
                struct { std::complex<double> obs; double re_e; double im_e; } comps[4] = {
                    { sta.getZxxObserved(ifreq), sta.getZxxSDRe(ifreq), sta.getZxxSDIm(ifreq) },
                    { sta.getZxyObserved(ifreq), sta.getZxySDRe(ifreq), sta.getZxySDIm(ifreq) },
                    { sta.getZyxObserved(ifreq), sta.getZyxSDRe(ifreq), sta.getZyxSDIm(ifreq) },
                    { sta.getZyyObserved(ifreq), sta.getZyySDRe(ifreq), sta.getZyySDIm(ifreq) },
                };
                for( int c = 0; c < 4; ++c ){
                    DataRow r;
                    r.freq      = freq;
                    r.datatype  = DTYPE_MT;
                    r.site_id   = sid;
                    r.site_x    = sx;
                    r.site_y    = sy;
                    r.site_z    = sz;
                    r.re_val    = comps[c].obs.real();
                    r.im_val    = comps[c].obs.imag();
                    r.re_err    = comps[c].re_e;
                    r.im_err    = comps[c].im_e;
                    r.component = c;
                    lookupCalc( calcLookup, r, &nMissingCalc );
                    rows.push_back(r);
                }
            }
        }
    }

    //----------------------------------------------------------------------
    // Apparent resistivity & phase stations (rhoXX..YY, phsXX..YY)
    //----------------------------------------------------------------------
    {
        const int nSta = pOD->getNumStationsApparentResistivityAndPhase();
        for( int i = 0; i < nSta; ++i ){
            const ObservedDataStationApparentResistivityAndPhase& sta =
                pOD->getStationApparentResistivityAndPhase(i);
            const int sid   = sta.getStationID();
            const double sx = sta.getLocationOfPoint().X;
            const double sy = sta.getLocationOfPoint().Y;
            const double sz = sta.getZCoordOfPoint();
            const int nf    = sta.getTotalNumberOfFrequency();

            // comp 0-3: rhoXX,XY,YX,YY  comp 4-7: phsXX,XY,YX,YY
            for( int ifreq = 0; ifreq < nf; ++ifreq ){
                const double freq = sta.getFrequencyValues(ifreq);
                struct { double obs; double err; } comps[8] = {
                    { sta.getApparentResistivityXXObserved(ifreq), sta.getApparentResistivityXXSD(ifreq) },
                    { sta.getApparentResistivityXYObserved(ifreq), sta.getApparentResistivityXYSD(ifreq) },
                    { sta.getApparentResistivityYXObserved(ifreq), sta.getApparentResistivityYXSD(ifreq) },
                    { sta.getApparentResistivityYYObserved(ifreq), sta.getApparentResistivityYYSD(ifreq) },
                    { sta.getPhaseXXObserved(ifreq), sta.getPhaseXXSD(ifreq) },
                    { sta.getPhaseXYObserved(ifreq), sta.getPhaseXYSD(ifreq) },
                    { sta.getPhaseYXObserved(ifreq), sta.getPhaseYXSD(ifreq) },
                    { sta.getPhaseYYObserved(ifreq), sta.getPhaseYYSD(ifreq) },
                };
                for( int c = 0; c < 8; ++c ){
                    DataRow r;
                    r.freq      = freq;
                    r.datatype  = DTYPE_APP;
                    r.site_id   = sid;
                    r.site_x    = sx;
                    r.site_y    = sy;
                    r.site_z    = sz;
                    r.re_val    = comps[c].obs;
                    r.im_val    = 0.0;
                    r.re_err    = comps[c].err;
                    r.im_err    = 0.0;
                    r.component = c;
                    lookupCalc( calcLookup, r, &nMissingCalc );
                    rows.push_back(r);
                }
            }
        }
    }

    //----------------------------------------------------------------------
    // HTF stations (Txx, Txy, Tyx, Tyy)
    //----------------------------------------------------------------------
    {
        const int nSta = pOD->getNumStationsHTF();
        for( int i = 0; i < nSta; ++i ){
            const ObservedDataStationHTF& sta = pOD->getStationHTF(i);
            const int sid   = sta.getStationID();
            const double sx = sta.getLocationOfPoint().X;
            const double sy = sta.getLocationOfPoint().Y;
            const double sz = sta.getZCoordOfPoint();
            const int nf    = sta.getTotalNumberOfFrequency();

            for( int ifreq = 0; ifreq < nf; ++ifreq ){
                const double freq = sta.getFrequencyValues(ifreq);
                struct { std::complex<double> obs; double re_e; double im_e; } comps[4] = {
                    { sta.getTxxObserved(ifreq), sta.getTxxSDRe(ifreq), sta.getTxxSDIm(ifreq) },
                    { sta.getTxyObserved(ifreq), sta.getTxySDRe(ifreq), sta.getTxySDIm(ifreq) },
                    { sta.getTyxObserved(ifreq), sta.getTyxSDRe(ifreq), sta.getTyxSDIm(ifreq) },
                    { sta.getTyyObserved(ifreq), sta.getTyySDRe(ifreq), sta.getTyySDIm(ifreq) },
                };
                for( int c = 0; c < 4; ++c ){
                    DataRow r;
                    r.freq = freq; r.datatype = DTYPE_HTF;
                    r.site_id = sid; r.site_x = sx; r.site_y = sy; r.site_z = sz;
                    r.re_val = comps[c].obs.real(); r.im_val = comps[c].obs.imag();
                    r.re_err = comps[c].re_e;       r.im_err = comps[c].im_e;
                    r.component = c;
                    lookupCalc( calcLookup, r, &nMissingCalc );
                    rows.push_back(r);
                }
            }
        }
    }

    //----------------------------------------------------------------------
    // VTF stations (Tzx, Tzy)
    //----------------------------------------------------------------------
    {
        const int nSta = pOD->getNumStationsVTF();
        for( int i = 0; i < nSta; ++i ){
            const ObservedDataStationVTF& sta = pOD->getStationVTF(i);
            const int sid   = sta.getStationID();
            const double sx = sta.getLocationOfPoint().X;
            const double sy = sta.getLocationOfPoint().Y;
            const double sz = sta.getZCoordOfPoint();
            const int nf    = sta.getTotalNumberOfFrequency();

            for( int ifreq = 0; ifreq < nf; ++ifreq ){
                const double freq = sta.getFrequencyValues(ifreq);
                struct { std::complex<double> obs; double re_e; double im_e; } comps[2] = {
                    { sta.getTzxObserved(ifreq), sta.getTzxSDRe(ifreq), sta.getTzxSDIm(ifreq) },
                    { sta.getTzyObserved(ifreq), sta.getTzySDRe(ifreq), sta.getTzySDIm(ifreq) },
                };
                for( int c = 0; c < 2; ++c ){
                    DataRow r;
                    r.freq = freq; r.datatype = DTYPE_VTF;
                    r.site_id = sid; r.site_x = sx; r.site_y = sy; r.site_z = sz;
                    r.re_val = comps[c].obs.real(); r.im_val = comps[c].obs.imag();
                    r.re_err = comps[c].re_e;       r.im_err = comps[c].im_e;
                    r.component = c;
                    lookupCalc( calcLookup, r, &nMissingCalc );
                    rows.push_back(r);
                }
            }
        }
    }

    //----------------------------------------------------------------------
    // PT stations (PTxx, PTxy, PTyx, PTyy) – real-valued
    //----------------------------------------------------------------------
    {
        const int nSta = pOD->getNumStationsPT();
        for( int i = 0; i < nSta; ++i ){
            const ObservedDataStationPT& sta = pOD->getStationPT(i);
            const int sid   = sta.getStationID();
            const double sx = sta.getLocationOfPoint().X;
            const double sy = sta.getLocationOfPoint().Y;
            const double sz = sta.getZCoordOfPoint();
            const int nf    = sta.getTotalNumberOfFrequency();

            for( int ifreq = 0; ifreq < nf; ++ifreq ){
                const double freq = sta.getFrequencyValues(ifreq);
                struct { double obs; double err; } comps[4] = {
                    { sta.getPTxxObserved(ifreq), sta.getPTxxSD(ifreq) },
                    { sta.getPTxyObserved(ifreq), sta.getPTxySD(ifreq) },
                    { sta.getPTyxObserved(ifreq), sta.getPTyxSD(ifreq) },
                    { sta.getPTyyObserved(ifreq), sta.getPTyySD(ifreq) },
                };
                for( int c = 0; c < 4; ++c ){
                    DataRow r;
                    r.freq = freq; r.datatype = DTYPE_PT;
                    r.site_id = sid; r.site_x = sx; r.site_y = sy; r.site_z = sz;
                    r.re_val = comps[c].obs; r.im_val = 0.0;
                    r.re_err = comps[c].err; r.im_err = 0.0;
                    r.component = c;
                    lookupCalc( calcLookup, r, &nMissingCalc );
                    rows.push_back(r);
                }
            }
        }
    }

    //----------------------------------------------------------------------
    // NMT stations (Yx, Yy)  – dipole; use midpoint as site location
    //----------------------------------------------------------------------
    {
        const int nSta = pOD->getNumStationsNMT();
        for( int i = 0; i < nSta; ++i ){
            const ObservedDataStationNMT& sta = pOD->getStationNMT(i);
            const int sid = sta.getStationID();
            const CommonParameters::locationDipole& loc = sta.getLocationOfStation();
            const double sx = 0.5*(loc.startPoint.X + loc.endPoint.X);
            const double sy = 0.5*(loc.startPoint.Y + loc.endPoint.Y);
            const double sz = 0.0; // NMT has no Z stored separately
            const int nf    = sta.getTotalNumberOfFrequency();

            for( int ifreq = 0; ifreq < nf; ++ifreq ){
                const double freq = sta.getFrequencyValues(ifreq);
                struct { std::complex<double> obs; double re_e; double im_e; } comps[2] = {
                    { sta.getYxObserved(ifreq), sta.getYxSDRe(ifreq), sta.getYxSDIm(ifreq) },
                    { sta.getYyObserved(ifreq), sta.getYySDRe(ifreq), sta.getYySDIm(ifreq) },
                };
                for( int c = 0; c < 2; ++c ){
                    DataRow r;
                    r.freq = freq; r.datatype = DTYPE_NMT;
                    r.site_id = sid; r.site_x = sx; r.site_y = sy; r.site_z = sz;
                    r.re_val = comps[c].obs.real(); r.im_val = comps[c].obs.imag();
                    r.re_err = comps[c].re_e;       r.im_err = comps[c].im_e;
                    r.component = c;
                    lookupCalc( calcLookup, r, &nMissingCalc );
                    rows.push_back(r);
                }
            }
        }
    }

    //----------------------------------------------------------------------
    // NMT2 stations (Zxx, Zxy, Zyx, Zyy for triangle-area dipoles)
    //----------------------------------------------------------------------
    {
        const int nSta = pOD->getNumStationsNMT2();
        for( int i = 0; i < nSta; ++i ){
            const ObservedDataStationNMT2& sta = pOD->getStationNMT2(i);
            const int sid   = sta.getStationID();
            // NMT2 uses two dipoles; report midpoint of dipole 0 as site location
            const CommonParameters::locationDipole& loc2 = sta.getLocationOfStation(0);
            const double sx = 0.5*(loc2.startPoint.X + loc2.endPoint.X);
            const double sy = 0.5*(loc2.startPoint.Y + loc2.endPoint.Y);
            const double sz = sta.getZCoordOfPoint(0, 0);
            const int nf    = sta.getTotalNumberOfFrequency();

            for( int ifreq = 0; ifreq < nf; ++ifreq ){
                const double freq = sta.getFrequencyValues(ifreq);
                struct { std::complex<double> obs; double re_e; double im_e; } comps[4] = {
                    { sta.getNMT2ZxxObserved(ifreq), sta.getNMT2ZxxSDRe(ifreq), sta.getNMT2ZxxSDIm(ifreq) },
                    { sta.getNMT2ZxyObserved(ifreq), sta.getNMT2ZxySDRe(ifreq), sta.getNMT2ZxySDIm(ifreq) },
                    { sta.getNMT2ZyxObserved(ifreq), sta.getNMT2ZyxSDRe(ifreq), sta.getNMT2ZyxSDIm(ifreq) },
                    { sta.getNMT2ZyyObserved(ifreq), sta.getNMT2ZyySDRe(ifreq), sta.getNMT2ZyySDIm(ifreq) },
                };
                for( int c = 0; c < 4; ++c ){
                    DataRow r;
                    r.freq = freq; r.datatype = DTYPE_NMT2;
                    r.site_id = sid; r.site_x = sx; r.site_y = sy; r.site_z = sz;
                    r.re_val = comps[c].obs.real(); r.im_val = comps[c].obs.imag();
                    r.re_err = comps[c].re_e;       r.im_err = comps[c].im_e;
                    r.component = c;
                    lookupCalc( calcLookup, r, &nMissingCalc );
                    rows.push_back(r);
                }
            }
        }
    }

    //----------------------------------------------------------------------
    // NMT2 apparent resistivity & phase stations
    //----------------------------------------------------------------------
    {
        const int nSta = pOD->getNumStationsNMT2ApparentResistivityAndPhase();
        for( int i = 0; i < nSta; ++i ){
            const ObservedDataStationNMT2ApparentResistivityAndPhase& sta =
                pOD->getStationNMT2ApparentResistivityAndPhase(i);
            const int sid   = sta.getStationID();
            // NMT2AppRes uses two dipoles; report midpoint of dipole 0 as site location
            const CommonParameters::locationDipole& loc2a = sta.getLocationOfStation(0);
            const double sx = 0.5*(loc2a.startPoint.X + loc2a.endPoint.X);
            const double sy = 0.5*(loc2a.startPoint.Y + loc2a.endPoint.Y);
            const double sz = sta.getZCoordOfPoint(0, 0);
            const int nf    = sta.getTotalNumberOfFrequency();

            for( int ifreq = 0; ifreq < nf; ++ifreq ){
                const double freq = sta.getFrequencyValues(ifreq);
                struct { double obs; double err; } comps[8] = {
                    { sta.getNMT2AppResXXObserved(ifreq), sta.getNMT2AppResXXSD(ifreq) },
                    { sta.getNMT2AppResXYObserved(ifreq), sta.getNMT2AppResXYSD(ifreq) },
                    { sta.getNMT2AppResYXObserved(ifreq), sta.getNMT2AppResYXSD(ifreq) },
                    { sta.getNMT2AppResYYObserved(ifreq), sta.getNMT2AppResYYSD(ifreq) },
                    { sta.getNMT2PhaseXXObserved(ifreq),  sta.getNMT2PhaseXXSD(ifreq)  },
                    { sta.getNMT2PhaseXYObserved(ifreq),  sta.getNMT2PhaseXYSD(ifreq)  },
                    { sta.getNMT2PhaseYXObserved(ifreq),  sta.getNMT2PhaseYXSD(ifreq)  },
                    { sta.getNMT2PhaseYYObserved(ifreq),  sta.getNMT2PhaseYYSD(ifreq)  },
                };
                for( int c = 0; c < 8; ++c ){
                    DataRow r;
                    r.freq = freq; r.datatype = DTYPE_NMT2A;
                    r.site_id = sid; r.site_x = sx; r.site_y = sy; r.site_z = sz;
                    r.re_val = comps[c].obs; r.im_val = 0.0;
                    r.re_err = comps[c].err; r.im_err = 0.0;
                    r.component = c;
                    lookupCalc( calcLookup, r, &nMissingCalc );
                    rows.push_back(r);
                }
            }
        }
    }

    //----------------------------------------------------------------------
    // Write /data group
    //----------------------------------------------------------------------
    const hsize_t nRows = rows.size();

    HdfHandle grpData( H5Gcreate2( fid, "/data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ), H5Gclose );

    // Attribute: total row count, plus how many rows (if any) could not be
    // matched to a gathered calculated value -- see lookupCalc() above.
    HdfHandle grpDataMeta( H5Gcreate2( grpData, "metadata", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ), H5Gclose );
    writeIntAttr( grpDataMeta, "nRows",         (int)nRows     );
    writeIntAttr( grpDataMeta, "iterNum",       iterNum        );
    writeIntAttr( grpDataMeta, "nMissingCalc",  nMissingCalc   );
    grpDataMeta.close();

    if( nRows > 0 ){
        HdfHandle memType = makeDataType();
        HdfHandle sp( H5Screate_simple( 1, &nRows, NULL ), H5Sclose );
        HdfHandle ds( H5Dcreate2( grpData, "data", memType, sp,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ), H5Dclose );
        hdf5check( H5Dwrite( ds, memType, H5S_ALL, H5S_ALL, H5P_DEFAULT, rows.data() ), "H5Dwrite" );
        ds.close(); sp.close(); memType.close();
    }

    grpData.close();

    if( nMissingCalc > 0 ){
        // Not fatal -- results_iterN.h5 is a diagnostic/post-processing
        // artifact layered on top of the inversion itself (same philosophy
        // as the Jacobian assembly's missing-file handling below) -- but
        // worth flagging clearly, since it likely means the MPI_Gatherv in
        // AnalysisControl.cpp and this function's row-building loop have
        // drifted out of sync on ordering/keys.
        OutputFiles::m_logFile << "# WARNING: " << nMissingCalc
                               << " /data row(s) had no matching calculated"
                               << " value (cal_re/cal_im set to NaN)." << std::endl;
    }

    OutputFiles::m_logFile << "# Written HDF5 /data group (iter " << iterNum
                           << ", " << nRows << " rows)" << std::endl;
}

//===========================================================================
// writeDistortionGroup -- writes the /distortion group of results_iterN.h5
//===========================================================================
// Embeds the same numbers ObservedData::outputDistortionParams() writes to
// distortion_iterN.dat (which is still written separately, unchanged --
// this is an addition, not a replacement). PE-0-only, no MPI involved, same
// as outputDistortionParams() itself. A no-op (nothing written, not even an
// empty group) when distortion estimation is disabled for this run.
static void writeDistortionGroup( hid_t fid, const int iterNum )
{
    const AnalysisControl* const pAC = AnalysisControl::getInstance();
    if( !pAC->estimateDistortionMatrix() ) return;

    const ObservedData* const pOD = ObservedData::getInstance();
    std::vector<FemticHDF5DistortionRow> rows;
    pOD->collectDistortionParamsForHDF5( rows );

    const hsize_t nRows = rows.size();

    HdfHandle grpDist( H5Gcreate2( fid, "/distortion", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ), H5Gclose );

    HdfHandle grpDistMeta( H5Gcreate2( grpDist, "metadata", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ), H5Gclose );
    writeIntAttr( grpDistMeta, "iterNum", iterNum );
    writeIntAttr( grpDistMeta, "nRows",   (int)nRows );
    // AnalysisControl::TypeOfDistortion of this run -- see
    // ObservedData::outputDistortionParams()/collectDistortionParamsForHDF5()
    // for what param1..4 mean under each value.
    writeIntAttr( grpDistMeta, "type", pAC->getTypeOfDistortion() );
    grpDistMeta.close();

    if( nRows > 0 ){
        HdfHandle memType = makeDistortionType();
        HdfHandle sp( H5Screate_simple( 1, &nRows, NULL ), H5Sclose );
        HdfHandle ds( H5Dcreate2( grpDist, "params", memType, sp,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ), H5Dclose );
        hdf5check( H5Dwrite( ds, memType, H5S_ALL, H5S_ALL, H5P_DEFAULT, rows.data() ), "H5Dwrite" );
        ds.close(); sp.close(); memType.close();
    }

    grpDist.close();

    OutputFiles::m_logFile << "# Written HDF5 /distortion group (iter " << iterNum
                           << ", " << nRows << " rows)" << std::endl;
}

//===========================================================================
// outputResultsToHDF5 -- combined results_iterN.h5 (added 2026-09-13)
//===========================================================================
// Replaces the former pair outputModelToHDF5()+outputDataToHDF5(), which
// wrote model_iterN.h5 and data_iterN.h5 separately. Now writes ONE file,
// results_iterN.h5, with /model, /data, and (when enabled) /distortion as
// sibling top-level groups -- see the schema comment in OutputHDF5.h.
//
// PE-0-only, like the functions it replaces, and for the same reason it
// must not perform any MPI collective itself: sensitivityScalarValuesReduced
// and calcRowsAll must ALREADY be the fully-reduced/gathered results of
// collective calls made by the caller on EVERY PE (Inversion::
// getSensitivityScalarValuesReduced() and the MPI_Gatherv of
// ObservedData::collectCalculatedValuesForHDF5() in AnalysisControl.cpp,
// respectively) -- see the 2026-09-09 deadlock-fix comments on
// writeModelGroup() above, which apply here unchanged.
void outputResultsToHDF5( const int iterNum,
                           const double* sensitivityScalarValuesReduced,
                           const std::vector<FemticHDF5CalcRow>& calcRowsAll )
{
    const AnalysisControl* const pAC = AnalysisControl::getInstance();
    if( pAC->getMyPE() != 0 ) return;

    std::ostringstream fname;
    fname << "results_iter" << iterNum << ".h5";

    if( !writeHdfFile(fname.str(), iterNum, [&](hid_t fid) {
        writeModelGroup( fid, iterNum, sensitivityScalarValuesReduced );
        writeDataGroup( fid, iterNum, calcRowsAll );
        writeDistortionGroup( fid, iterNum );
    }) ) return;

    OutputFiles::m_logFile << "# Written HDF5 results file: " << fname.str() << std::endl;
}
#endif // _HDF5_OUT

#ifdef _HDF5_JAC
//===========================================================================
// exchange.h5  (added 2026-10-01)
//
// Replaces the former jacobian.h5, rough.h5 and mesh.h5 with ONE file that
// carries everything an external tool needs to work with the Jacobian:
//
//   /metadata                  attrs: iterNum, exchangeVersion
//   /jacobian/metadata         attrs: iterNum, nData, nModel, weighted
//   /jacobian/values           double[nData][nModel]
//   /jacobian/data_errors      double[nData]
//   /rough/metadata            attrs: nRows, nNonZeros, format="CSR"
//   /rough/row_ptr|col_ind|values
//   /mesh/metadata             attrs: meshType, nNodes, nElem, nNodesPerElem,
//                                     neighborFormat[, nNeighborElem]
//   /mesh/node_coords|elem_nodes|neighbor_elements[|neighbor_face_ptr]
//
// It is written by outputJacobianToHDF5(), i.e. exactly when (and only when)
// the Jacobian is written. Mesh and roughening matrix are fixed for the whole
// run, so they are simply re-read from their in-memory owners (MeshData via
// AnalysisControl, RougheningSquareMatrix via ResistivityBlock) each time.
//===========================================================================
namespace {

// Write a scalar string attribute (fixed-length, NUL-terminated).
static void writeStringAttr( hid_t obj, const char* name, const char* val )
{
    HdfHandle sp( H5Screate( H5S_SCALAR ), H5Sclose );
    HdfHandle stype( H5Tcopy( H5T_C_S1 ), H5Tclose );
    hdf5check( H5Tset_size( stype, strlen(val) + 1 ), "H5Tset_size" );
    HdfHandle at( H5Acreate2( obj, name, stype, sp, H5P_DEFAULT, H5P_DEFAULT ), H5Aclose );
    hdf5check( H5Awrite( at, stype, val ), "H5Awrite" );
    at.close(); stype.close(); sp.close();
}

// Create + write one dataset of the given native type and shape.
static void writeDataset( hid_t loc, const char* name, hid_t type,
                          const int rank, const hsize_t* dims, const void* buf )
{
    HdfHandle sp( H5Screate_simple( rank, dims, NULL ), H5Sclose );
    HdfHandle ds( H5Dcreate2( loc, name, type, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ), H5Dclose );
    hdf5check( H5Dwrite( ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf ), "H5Dwrite" );
    ds.close(); sp.close();
}

static void writeDataset1D( hid_t loc, const char* name, hid_t type, const hsize_t n, const void* buf )
{
    writeDataset( loc, name, type, 1, &n, buf );
}

static void writeDataset2D( hid_t loc, const char* name, hid_t type,
                            const hsize_t n0, const hsize_t n1, const void* buf )
{
    const hsize_t dims[2] = { n0, n1 };
    writeDataset( loc, name, type, 2, dims, buf );
}

// Create group <name> with an empty "metadata" subgroup. Returns the
// metadata group handle (caller writes attributes, then closes it); the parent
// group id is returned through grpOut (caller adds datasets, then closes it).
static HdfHandle createGroupWithMetadata( hid_t fid, const char* name, HdfHandle* grpOut )
{
    *grpOut = HdfHandle( H5Gcreate2( fid, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ), H5Gclose );
    HdfHandle meta( H5Gcreate2( *grpOut, "metadata", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ), H5Gclose );
    return meta;
}

//--- /jacobian ------------------------------------------------------------
static void writeJacobianGroup( hid_t fid, const int iterNum,
                                const int numDataTotal, const int numModel,
                                const double* J, const double* dataErrors )
{
    HdfHandle grp;
    HdfHandle meta = createGroupWithMetadata( fid, "/jacobian", &grp );
    writeIntAttr( meta, "iterNum",  iterNum      );
    writeIntAttr( meta, "nData",    numDataTotal );
    writeIntAttr( meta, "nModel",   numModel     );
    writeIntAttr( meta, "weighted", 1            ); // J = Cd^{-1/2} * dF/dm
    meta.close();

    writeDataset2D( grp, "values",      H5T_NATIVE_DOUBLE, (hsize_t)numDataTotal, (hsize_t)numModel, J );
    writeDataset1D( grp, "data_errors", H5T_NATIVE_DOUBLE, (hsize_t)numDataTotal, dataErrors );
    grp.close();
}

//--- /rough ---------------------------------------------------------------
static void writeRoughGroup( hid_t fid, const RougheningSquareMatrix& R )
{
    const int nRows    = R.getNumRows();
    const int nNonZero = R.getNumNonZeros();

    std::vector<int>    rowPtr( nRows + 1 );
    std::vector<int>    colInd( nNonZero );
    std::vector<double> values( nNonZero );
    for( int i = 0; i <= nRows; ++i )   rowPtr[i] = R.getRowIndexCRS(i);
    for( int k = 0; k < nNonZero; ++k ){
        colInd[k] = R.getColumnsCRS(k);
        values[k] = R.getValueCRS(k);
    }

    HdfHandle grp;
    HdfHandle meta = createGroupWithMetadata( fid, "/rough", &grp );
    writeIntAttr( meta, "nRows",     nRows    );
    writeIntAttr( meta, "nNonZeros", nNonZero );
    writeStringAttr( meta, "format", "CSR" );
    meta.close();

    writeDataset1D( grp, "row_ptr", H5T_NATIVE_INT,    (hsize_t)(nRows + 1), rowPtr.data() );
    writeDataset1D( grp, "col_ind", H5T_NATIVE_INT,    (hsize_t)nNonZero,    colInd.data() );
    writeDataset1D( grp, "values",  H5T_NATIVE_DOUBLE, (hsize_t)nNonZero,    values.data() );
    grp.close();
}

//--- /mesh ----------------------------------------------------------------
// Lossless dump of the geometry read from mesh.dat (MeshData::inputMeshData()).
static void writeMeshGroup( hid_t fid, const MeshData* const pMesh )
{
    const int nNodes   = pMesh->getNumNodeTotal();
    const int nElem    = pMesh->getNumElemTotal();
    const int nNPE     = pMesh->getNumNodePerElement();
    const int meshType = pMesh->getMeshType();

    // Non-conforming hexahedral meshes allow a variable number of neighbor
    // elements per element face (hanging nodes at refinement boundaries),
    // so MeshDataNonConformingHexaElement hides MeshData's fixed-degree
    // getIDOfNeighborElement()/getNumNeighborElement() with its own
    // per-(element,face) overloads. Detect that case and switch to a CSR
    // layout; HEXA and TETRA meshes use the generic fixed-degree accessor.
    const MeshDataNonConformingHexaElement* const pMeshNC =
        dynamic_cast<const MeshDataNonConformingHexaElement*>( pMesh );
    const int neighborFormat = ( pMeshNC != NULL ) ? 1 : 0; // 0=dense, 1=CSR

    HdfHandle grp;
    HdfHandle meta = createGroupWithMetadata( fid, "/mesh", &grp );
    writeIntAttr( meta, "meshType",       meshType       );
    writeIntAttr( meta, "nNodes",         nNodes         );
    writeIntAttr( meta, "nElem",          nElem          );
    writeIntAttr( meta, "nNodesPerElem",  nNPE           );
    writeIntAttr( meta, "neighborFormat", neighborFormat );
    if( neighborFormat == 0 ){
        writeIntAttr( meta, "nNeighborElem", pMesh->getNumNeighborElement() );
    }
    meta.close();

    {   // node_coords  double[nNodes][3]
        std::vector<double> coords( (size_t)nNodes * 3 );
        for( int n = 0; n < nNodes; ++n ){
            coords[ n*3 + 0 ] = pMesh->getXCoordinatesOfNodes(n);
            coords[ n*3 + 1 ] = pMesh->getYCoordinatesOfNodes(n);
            coords[ n*3 + 2 ] = pMesh->getZCoordinatesOfNodes(n);
        }
        writeDataset2D( grp, "node_coords", H5T_NATIVE_DOUBLE, (hsize_t)nNodes, 3, coords.data() );
    }

    {   // elem_nodes  int[nElem][nNodesPerElem]
        std::vector<int> enodes( (size_t)nElem * nNPE );
        for( int e = 0; e < nElem; ++e )
            for( int k = 0; k < nNPE; ++k )
                enodes[ (size_t)e*nNPE + k ] = pMesh->getNodesOfElements(e,k);
        writeDataset2D( grp, "elem_nodes", H5T_NATIVE_INT, (hsize_t)nElem, (hsize_t)nNPE, enodes.data() );
    }

    if( neighborFormat == 0 ){
        // Dense: int[nElem][nNeighborElem], fixed degree per element.
        const int nNeighbor = pMesh->getNumNeighborElement();
        std::vector<int> neighbors( (size_t)nElem * nNeighbor );
        for( int e = 0; e < nElem; ++e )
            for( int k = 0; k < nNeighbor; ++k )
                neighbors[ (size_t)e*nNeighbor + k ] = pMesh->getIDOfNeighborElement(e,k);
        writeDataset2D( grp, "neighbor_elements", H5T_NATIVE_INT,
                        (hsize_t)nElem, (hsize_t)nNeighbor, neighbors.data() );
    } else {
        // CSR over (element, face) pairs, 6 faces per hexahedral element:
        // neighbor_face_ptr[nElem*6+1] gives the offset of each
        // (element,face)'s neighbor list within the flat neighbor_elements.
        const int nFaceSlots = nElem * 6;
        std::vector<int> facePtr( nFaceSlots + 1, 0 );
        for( int e = 0; e < nElem; ++e )
            for( int f = 0; f < 6; ++f )
                facePtr[ e*6 + f + 1 ] = facePtr[ e*6 + f ] + pMeshNC->getNumNeighborElement(e,f);

        std::vector<int> neighbors( facePtr[nFaceSlots] );
        for( int e = 0; e < nElem; ++e ){
            for( int f = 0; f < 6; ++f ){
                const int nNeib = pMeshNC->getNumNeighborElement(e,f);
                const int off   = facePtr[ e*6 + f ];
                for( int k = 0; k < nNeib; ++k )
                    neighbors[ off + k ] = pMeshNC->getIDOfNeighborElement(e,f,k);
            }
        }
        writeDataset1D( grp, "neighbor_face_ptr", H5T_NATIVE_INT, (hsize_t)(nFaceSlots + 1), facePtr.data() );
        writeDataset1D( grp, "neighbor_elements", H5T_NATIVE_INT, (hsize_t)neighbors.size(), neighbors.data() );
    }
    grp.close();
}

} // anonymous namespace

//===========================================================================
// outputJacobianToHDF5  -- writes exchange.h5 (jacobian + rough + mesh)
//===========================================================================
void outputJacobianToHDF5( const int iterNum,
                           const int numDataTotal,
                           const int numModel,
                           const double* sensitivityMatrix,
                           const double* dataErrorsGlobal )
{
    // Only PE 0 holds the full arrays and writes.
    const AnalysisControl* const pAC = AnalysisControl::getInstance();
    if( pAC->getMyPE() != 0 ) return;

    // Fixed filename, replaced after a successful export: Jacobians can be very
    // large, so only the most recently written one is kept on disk. The
    // iteration number is recorded in /metadata and /jacobian/metadata.
    const std::string fname = "exchange.h5";

    if( !writeHdfFile(fname, iterNum, [&](hid_t fid) {
        {   // /metadata
            HdfHandle grpMeta( H5Gcreate2( fid, "/metadata", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ), H5Gclose );
            writeIntAttr( grpMeta, "iterNum",         iterNum );
            writeIntAttr( grpMeta, "exchangeVersion", 1       );
            grpMeta.close();
        }
        writeJacobianGroup( fid, iterNum, numDataTotal, numModel, sensitivityMatrix, dataErrorsGlobal );
        writeRoughGroup( fid, ResistivityBlock::getInstance()->getRougheningMatrix() );
        writeMeshGroup( fid, pAC->getPointerOfMeshData() );
    }) ) return;

    OutputFiles::m_logFile << "# Written HDF5 exchange file: " << fname
                           << "  (jacobian " << numDataTotal << " data x " << numModel
                           << " model params, roughening matrix, mesh; iteration "
                           << iterNum << ")" << std::endl;
}
#endif // _HDF5_JAC
