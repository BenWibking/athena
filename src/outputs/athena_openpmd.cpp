//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file athena_openpmd.cpp
//! \brief openPMD outputs

// Athena++ headers (must include first to get OPENPMDOUTPUT definition)
#include "../athena.hpp"

// Only proceed if openPMD output enabled
#ifdef OPENPMDOUTPUT

// C++ headers
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// OpenPMD headers
#include <openPMD/openPMD.hpp>

// More Athena++ headers
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "athena_openpmd.hpp"
#include "outputs.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn OPENPMDOutput::OPENPMDOutput(OutputParameters oparams)
//! \brief constructor for OPENPMDOutput, initializes variables

template<typename opmd_out_t>
OPENPMDOutput<opmd_out_t>::OPENPMDOutput(OutputParameters oparams)
    : OutputType(oparams), backend_config_("default"), coarsening_factor_(1) {
}

//----------------------------------------------------------------------------------------
//! \fn void OPENPMDOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin, bool flag)
//! \brief Cycles over all MeshBlocks and writes OutputData in openPMD format

template<typename opmd_out_t>
void OPENPMDOutput<opmd_out_t>::WriteOutputFile(Mesh *pm, ParameterInput *pin,
                                                 bool flag) {
  // Check if ghost zones were requested and warn user
  if (output_params.include_ghost_zones) {
    if (Globals::my_rank == 0) {
      std::cout << "### WARNING in openPMD output" << std::endl
                << "Ghost zones are not supported by openPMD-API." << std::endl
                << "Outputting interior cells only." << std::endl;
    }
  }

  // Determine file name
  std::string filename = output_params.file_basename;
  filename.append(".");
  filename.append(output_params.file_id);
  filename.append(".");
  std::stringstream file_number;
  file_number << std::setw(5) << std::setfill('0') << output_params.file_number;
  filename.append(file_number.str());
  if (flag) {  // final output
    filename.append(".final");
  }
  filename.append(".bp");

  // Prepare backend configuration
  std::string backend_config = backend_config_ == "default" ? "{}" : "@" + backend_config_;

  // Create openPMD Series
  openPMD::Series series = openPMD::Series(
      filename,
      openPMD::Access::CREATE,
#ifdef MPI_PARALLEL
      MPI_COMM_WORLD,
#endif
      backend_config);

  // Set series metadata
  series.setIterationEncoding(openPMD::IterationEncoding::fileBased);
  series.setAuthor("Athena++ User");
  series.setSoftware("Athena++", "unknown");

  // Open iteration
  auto it = series.iterations[output_params.file_number];
  it.open();

  // Set time information
  it.setTime(pm->time);
  it.setDt(pm->dt);
  it.setAttribute("NCycle", pm->ncycle);

  // Count total number of blocks (for slicing support)
  int num_blocks_local = pm->nblocal;
  int num_blocks_global = pm->nbtotal;

  // Write mesh attributes
  WriteAttributes(it, pm, pin, num_blocks_global);

  // Write block metadata (levels and logical locations)
  WriteBlockMetadata(it, pm);

  // Load and write variable data
  WriteVariableData(it, pm, num_blocks_global);

  // Close iteration and series
  it.close();
  series.close();

  // Update output parameters for next output
  output_params.file_number++;
  output_params.next_time += output_params.dt;
  pin->SetInteger(output_params.block_name, "file_number", output_params.file_number);
  pin->SetReal(output_params.block_name, "next_time", output_params.next_time);
}

//----------------------------------------------------------------------------------------
//! \fn void OPENPMDOutput::WriteAttributes()
//! \brief Write mesh attributes to openPMD file

template<typename opmd_out_t>
void OPENPMDOutput<opmd_out_t>::WriteAttributes(openPMD::Iteration& it, Mesh *pm,
                                                 ParameterInput *pin,
                                                 int num_blocks_global) {
  // Write Athena++ mesh metadata
  it.setAttribute("NumDims", pm->ndim);
  it.setAttribute("NumMeshBlocks", num_blocks_global);
  it.setAttribute("MaxLevel", pm->current_level - pm->root_level);

  // Write mesh block size
  MeshBlock *pmb = pm->my_blocks(0);
  std::vector<int> meshblock_size = {pmb->block_size.nx1,
                                     pmb->block_size.nx2,
                                     pmb->block_size.nx3};
  it.setAttribute("MeshBlockSize", meshblock_size);

  // Write root grid domain
  std::vector<Real> root_grid_domain = {
      pm->mesh_size.x1min, pm->mesh_size.x1max, pm->mesh_size.x1rat,
      pm->mesh_size.x2min, pm->mesh_size.x2max, pm->mesh_size.x2rat,
      pm->mesh_size.x3min, pm->mesh_size.x3max, pm->mesh_size.x3rat};
  it.setAttribute("RootGridDomain", root_grid_domain);

  // Write root grid size
  std::vector<int> root_grid_size = {pm->mesh_size.nx1,
                                     pm->mesh_size.nx2,
                                     pm->mesh_size.nx3};
  it.setAttribute("RootGridSize", root_grid_size);

  // Write coordinate system
  it.setAttribute("Coordinates", std::string(COORDINATE_SYSTEM));

  // Write input file as string
  std::ostringstream oss;
  pin->ParameterDump(oss);
  it.setAttribute("InputFile", oss.str());
}

//----------------------------------------------------------------------------------------
//! \fn void OPENPMDOutput::WriteBlockMetadata()
//! \brief Write block levels and logical locations

template<typename opmd_out_t>
void OPENPMDOutput<opmd_out_t>::WriteBlockMetadata(openPMD::Iteration& it, Mesh *pm) {
  // Gather block levels
  std::vector<int> levels_local;
  for (int b = 0; b < pm->nblocal; ++b) {
    MeshBlock *pmb = pm->my_blocks(b);
    levels_local.push_back(pmb->loc.level - pm->root_level);
  }

  // Gather logical locations
  std::vector<std::int64_t> locations_local;
  for (int b = 0; b < pm->nblocal; ++b) {
    MeshBlock *pmb = pm->my_blocks(b);
    locations_local.push_back(pmb->loc.lx1);
    locations_local.push_back(pmb->loc.lx2);
    locations_local.push_back(pmb->loc.lx3);
  }

#ifdef MPI_PARALLEL
  // Gather to all ranks
  std::vector<int> levels_global(pm->nbtotal);
  std::vector<int> recv_counts(Globals::nranks);
  std::vector<int> recv_displs(Globals::nranks);

  for (int n = 0; n < Globals::nranks; ++n) {
    recv_counts[n] = pm->nblist[n];
    recv_displs[n] = pm->nslist[n];
  }

  MPI_Allgatherv(levels_local.data(), pm->nblocal, MPI_INT,
                 levels_global.data(), recv_counts.data(), recv_displs.data(),
                 MPI_INT, MPI_COMM_WORLD);

  std::vector<std::int64_t> locations_global(pm->nbtotal * 3);
  for (int n = 0; n < Globals::nranks; ++n) {
    recv_counts[n] = pm->nblist[n] * 3;
    recv_displs[n] = pm->nslist[n] * 3;
  }

  MPI_Allgatherv(locations_local.data(), pm->nblocal * 3, MPI_LONG,
                 locations_global.data(), recv_counts.data(), recv_displs.data(),
                 MPI_LONG, MPI_COMM_WORLD);

  // Write on rank 0 only
  if (Globals::my_rank == 0) {
    it.setAttribute("Levels", levels_global);
    it.setAttribute("LogicalLocations", locations_global);
  }
#else
  // Serial: write directly
  it.setAttribute("Levels", levels_local);
  it.setAttribute("LogicalLocations", locations_local);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void OPENPMDOutput::WriteVariableData()
//! \brief Write variable data to openPMD file

template<typename opmd_out_t>
void OPENPMDOutput<opmd_out_t>::WriteVariableData(openPMD::Iteration& it, Mesh *pm,
                                                   int num_blocks_global) {
  // Get first block for reference
  MeshBlock *pmb = pm->my_blocks(0);

  // Set output indices (interior cells only)
  out_is = pmb->is;
  out_ie = pmb->ie;
  out_js = pmb->js;
  out_je = pmb->je;
  out_ks = pmb->ks;
  out_ke = pmb->ke;

  // Determine output size per block
  int nx1 = pmb->block_size.nx1;
  int nx2 = pmb->block_size.nx2;
  int nx3 = pmb->block_size.nx3;

  // Load output data for first block to get variable list
  LoadOutputData(pmb);

  // Loop through all variables in the output data list
  OutputData *pod = pfirst_data_;
  while (pod != nullptr) {
    // Determine number of components
    int num_components = 1;
    if (pod->type == "VECTORS") {
      num_components = 3;
    } else if (pod->type == "TENSORS") {
      num_components = 9;
    }

    // Create mesh record for this variable
    std::string record_name = pod->name;
    auto mesh_record = it.meshes[record_name];

    // Set mesh geometry and properties
    mesh_record.setGeometry(openPMD::Mesh::Geometry::cartesian);
    mesh_record.setDataOrder(openPMD::Mesh::DataOrder::C);

    // Calculate grid spacing at finest level
    Real dx1 = (pm->mesh_size.x1max - pm->mesh_size.x1min) /
               (pm->mesh_size.nx1 * (1 << (pm->current_level - pm->root_level)));
    Real dx2 = (pm->mesh_size.x2max - pm->mesh_size.x2min) /
               (pm->mesh_size.nx2 * (1 << (pm->current_level - pm->root_level)));
    Real dx3 = (pm->mesh_size.x3max - pm->mesh_size.x3min) /
               (pm->mesh_size.nx3 * (1 << (pm->current_level - pm->root_level)));

    // Set grid properties based on dimensionality
    if (pm->ndim == 3) {
      mesh_record.setGridSpacing(std::vector<Real>{dx3, dx2, dx1});
      mesh_record.setAxisLabels({"z", "y", "x"});
      mesh_record.setGridGlobalOffset(
          {pm->mesh_size.x3min, pm->mesh_size.x2min, pm->mesh_size.x1min});
    } else if (pm->ndim == 2) {
      mesh_record.setGridSpacing(std::vector<Real>{dx2, dx1});
      mesh_record.setAxisLabels({"y", "x"});
      mesh_record.setGridGlobalOffset({pm->mesh_size.x2min, pm->mesh_size.x1min});
    } else {  // 1D
      mesh_record.setGridSpacing(std::vector<Real>{dx1});
      mesh_record.setAxisLabels({"x"});
      mesh_record.setGridGlobalOffset({pm->mesh_size.x1min});
    }

    // Loop through components
    for (int comp = 0; comp < num_components; ++comp) {
      // Determine component name
      std::string comp_name;
      if (num_components == 1) {
        comp_name = openPMD::MeshRecordComponent::SCALAR;
      } else if (pod->type == "VECTORS") {
        if (comp == 0)
          comp_name = "x";
        else if (comp == 1)
          comp_name = "y";
        else
          comp_name = "z";
      } else {  // TENSORS
        // Map flat index to component name (xx, yy, zz, xy, xz, yx, yz, zx, zy)
        const char *tensor_names[] = {"xx", "yy", "zz", "xy", "xz", "yx", "yz", "zx", "zy"};
        comp_name = tensor_names[comp];
      }

      auto mesh_comp = mesh_record[comp_name];

      // Calculate global extent at finest level
      std::uint64_t global_nx1 = pm->mesh_size.nx1 *
                                 (1 << (pm->current_level - pm->root_level));
      std::uint64_t global_nx2 = pm->mesh_size.nx2 *
                                 (1 << (pm->current_level - pm->root_level));
      std::uint64_t global_nx3 = pm->mesh_size.nx3 *
                                 (1 << (pm->current_level - pm->root_level));

      openPMD::Extent global_extent;
      if (pm->ndim == 3) {
        global_extent = {global_nx3, global_nx2, global_nx1};
      } else if (pm->ndim == 2) {
        global_extent = {global_nx2, global_nx1};
      } else {
        global_extent = {global_nx1};
      }

      // Create dataset
      auto dataset = openPMD::Dataset(openPMD::determineDatatype<opmd_out_t>(),
                                      global_extent);
      mesh_comp.resetDataset(dataset);
      mesh_comp.setPosition(std::vector<Real>(pm->ndim, 0.5));
    }

    pod = pod->pnext;
  }

  ClearOutputData();

  // Now write data for all blocks
  for (int b = 0; b < pm->nblocal; ++b) {
    pmb = pm->my_blocks(b);

    // Reset output indices
    out_is = pmb->is;
    out_ie = pmb->ie;
    out_js = pmb->js;
    out_je = pmb->je;
    out_ks = pmb->ks;
    out_ke = pmb->ke;

    // Load output data for this block
    LoadOutputData(pmb);

    // Get block level
    int level = pmb->loc.level - pm->root_level;

    // Loop through variables again to write data
    pod = pfirst_data_;
    while (pod != nullptr) {
      int num_components = 1;
      if (pod->type == "VECTORS") {
        num_components = 3;
      } else if (pod->type == "TENSORS") {
        num_components = 9;
      }

      std::string record_name = pod->name;
      auto mesh_record = it.meshes[record_name];

      // Get chunk offset and extent
      auto [chunk_offset, chunk_extent] = GetChunkOffsetAndExtent(pm, pmb, level);

      // Calculate buffer size
      std::size_t buffer_size = 1;
      for (auto ext : chunk_extent) {
        buffer_size *= ext;
      }

      // Write each component
      for (int comp = 0; comp < num_components; ++comp) {
        std::string comp_name;
        if (num_components == 1) {
          comp_name = openPMD::MeshRecordComponent::SCALAR;
        } else if (pod->type == "VECTORS") {
          if (comp == 0)
            comp_name = "x";
          else if (comp == 1)
            comp_name = "y";
          else
            comp_name = "z";
        } else {
          const char *tensor_names[] = {"xx", "yy", "zz", "xy", "xz", "yx", "yz", "zx", "zy"};
          comp_name = tensor_names[comp];
        }

        auto mesh_comp = mesh_record[comp_name];

        // Allocate buffer
        std::vector<opmd_out_t> buffer(buffer_size);

        // Copy data from AthenaArray to buffer
        int index = 0;
        for (int k = out_ks; k <= out_ke; ++k) {
          for (int j = out_js; j <= out_je; ++j) {
            for (int i = out_is; i <= out_ie; ++i) {
              buffer[index++] = static_cast<opmd_out_t>(pod->data(comp, k, j, i));
            }
          }
        }

        // Write chunk
        mesh_comp.storeChunk(buffer, chunk_offset, chunk_extent);
      }

      pod = pod->pnext;
    }

    ClearOutputData();
  }

  // Flush all data
  it.seriesFlush();
}

//----------------------------------------------------------------------------------------
//! \fn std::tuple<...> OPENPMDOutput::GetChunkOffsetAndExtent()
//! \brief Calculate chunk offset and extent for a MeshBlock

template<typename opmd_out_t>
std::tuple<std::vector<std::uint64_t>, std::vector<std::uint64_t>>
OPENPMDOutput<opmd_out_t>::GetChunkOffsetAndExtent(Mesh *pm, MeshBlock *pmb,
                                                    int level) {
  std::vector<std::uint64_t> chunk_offset;
  std::vector<std::uint64_t> chunk_extent;

  // Block size
  std::uint64_t nx1 = pmb->block_size.nx1;
  std::uint64_t nx2 = pmb->block_size.nx2;
  std::uint64_t nx3 = pmb->block_size.nx3;

  // Calculate offset based on logical location and level
  std::uint64_t offset_x1 = pmb->loc.lx1 * nx1;
  std::uint64_t offset_x2 = pmb->loc.lx2 * nx2;
  std::uint64_t offset_x3 = pmb->loc.lx3 * nx3;

  if (pm->ndim == 3) {
    chunk_offset = {offset_x3, offset_x2, offset_x1};
    chunk_extent = {nx3, nx2, nx1};
  } else if (pm->ndim == 2) {
    chunk_offset = {offset_x2, offset_x1};
    chunk_extent = {nx2, nx1};
  } else {
    chunk_offset = {offset_x1};
    chunk_extent = {nx1};
  }

  return {chunk_offset, chunk_extent};
}

// Explicit template instantiations
template class OPENPMDOutput<float>;
template class OPENPMDOutput<double>;

#endif  // OPENPMDOUTPUT
