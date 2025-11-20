#ifndef OUTPUTS_ATHENA_OPENPMD_HPP_
#define OUTPUTS_ATHENA_OPENPMD_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file athena_openpmd.hpp
//! \brief provides class for openPMD outputs

// Only proceed if openPMD output enabled
#ifdef OPENPMDOUTPUT

// C++ headers
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "outputs.hpp"

// Forward declarations for openPMD
namespace openPMD {
class Iteration;
class Series;
}  // namespace openPMD

// Forward declarations
class Mesh;
class MeshBlock;
class ParameterInput;

//----------------------------------------------------------------------------------------
//! \class OPENPMDOutput
//! \brief derived OutputType class for OpenPMD files

template <typename opmd_out_t>
class OPENPMDOutput : public OutputType {
 public:
  explicit OPENPMDOutput(OutputParameters oparams);
  void WriteOutputFile(Mesh *pm, ParameterInput *pin, bool flag) override;

 private:
  std::string backend_config_;  // ADIOS2 or HDF5 backend config
  int coarsening_factor_;        // for coarsened outputs (future enhancement)

  // Helper methods
  void WriteAttributes(openPMD::Iteration& it, Mesh *pm, ParameterInput *pin,
                       int num_blocks_global);
  void WriteBlockMetadata(openPMD::Iteration& it, Mesh *pm);
  void WriteVariableData(openPMD::Iteration& it, Mesh *pm, int num_blocks_global);
  std::tuple<std::vector<std::uint64_t>, std::vector<std::uint64_t>>
    GetChunkOffsetAndExtent(Mesh *pm, MeshBlock *pmb, int level);
};

#endif  // OPENPMDOUTPUT
#endif  // OUTPUTS_ATHENA_OPENPMD_HPP_
