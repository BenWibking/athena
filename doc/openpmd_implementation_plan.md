# Implementation Plan for openPMD Output Support in Athena++

## Overview

This document outlines the plan to add openPMD (Open Standard for Particle-Mesh Data) output support to Athena++. The implementation follows the existing pattern established by Athena++'s HDF5 output (`ATHDF5Output`) while adapting Parthenon's openPMD approach to Athena++'s data structures.

## Key Architectural Decisions

1. **Follow Athena++'s Output Pattern**: Create a new `OPENPMDOutput` class that inherits from `OutputType`, similar to `ATHDF5Output`, `VTKOutput`, etc.

2. **Use openPMD-API Library**: Leverage the same openPMD-API library that Parthenon uses for standard-compliant output

3. **Support Single/Double Precision**: Use a template class like `ATHDF5Output<h5out_t>` to support different output precisions

4. **File Format**: Default to ADIOS2 backend (`.bp` files) as used in Parthenon, with potential support for HDF5 backend

5. **No Ghost Zone Support**: openPMD-API does not support ghost zones, so the `include_ghost_zones` parameter will be ignored/unsupported for openPMD outputs

## Implementation Steps

### 1. Create Header File: `src/outputs/athena_openpmd.hpp`

Define the `OPENPMDOutput` template class:

- Inherit from `OutputType`
- Similar structure to `ATHDF5Output` in `outputs.hpp:182-309`
- Add template parameter for output precision (float, double, etc.)
- Declare `WriteOutputFile()` method override
- Add helper methods for writing mesh records, attributes, and coordinates

**Key class members:**
```cpp
template <typename opmd_out_t>
class OPENPMDOutput : public OutputType {
 public:
  explicit OPENPMDOutput(OutputParameters oparams);
  void WriteOutputFile(Mesh *pm, ParameterInput *pin, bool flag) override;

 private:
  std::string backend_config_;  // ADIOS2 or HDF5 backend config
  int coarsening_factor_;        // for coarsened outputs (future)

  // Helper methods
  void WriteAttributes(openPMD::Iteration& it, Mesh *pm, ParameterInput *pin);
  void WriteVariableData(openPMD::Iteration& it, Mesh *pm);
  void WriteCoordinates(openPMD::Iteration& it, Mesh *pm);
};
```

### 2. Create Implementation File: `src/outputs/athena_openpmd.cpp`

#### Core Components

##### a. File Creation (based on Parthenon lines 302-372)

- Create `openPMD::Series` with file-based iteration encoding
- Support MPI parallel I/O when `MPI_PARALLEL` is defined
- Generate filename: `<basename>.<id>.<number>.bp`
- Set series metadata (author, software, date)

```cpp
std::string backend_config = backend_config_ == "default" ? "{}" : "@" + backend_config_;

auto filename = output_params.file_basename + "." + output_params.file_id;
if (flag) {  // final output
  filename.append(".final");
}
filename.append("." + std::to_string(output_params.file_number));
filename.append(".bp");

openPMD::Series series = openPMD::Series(
  filename,
  openPMD::Access::CREATE,
#ifdef MPI_PARALLEL
  MPI_COMM_WORLD,
#endif
  backend_config
);

series.setIterationEncoding(openPMD::IterationEncoding::fileBased);
auto it = series.iterations[output_params.file_number];
it.open();
```

##### b. Attribute Writing (based on Parthenon lines 386-524)

Write standard openPMD and Athena++ metadata:

**Standard openPMD attributes:**
- `Time` - simulation time
- `Dt` - timestep
- `NCycle` - cycle number

**Athena++ mesh metadata:**
- `NumDims` - number of dimensions
- `NumMeshBlocks` - total number of mesh blocks
- `MaxLevel` - maximum refinement level
- `RootGridDomain` - float[9] array with xyz mins, maxs, rats
- `RootGridSize` - int[3] array with nx1, nx2, nx3
- `MeshBlockSize` - int[3] array with block dimensions
- `Coordinates` - coordinate system name (e.g., "cartesian", "cylindrical", "spherical_polar")
- `BoundaryConditions` - array of boundary condition names

**AMR structure:**
- `loc.lx123` - logical locations (global array)
- `loc.level-gid-lid-cnghost-gflag` - level, global ID, local ID, corner ghost flag
- Block levels array

**Input file:**
- `InputFile` - full input file as string attribute

##### c. Variable Data Writing (based on Parthenon lines 535-823)

Main loop structure:

```cpp
// Get list of variables from OutputData linked list
OutputData* pod = pfirst_data_;

while (pod != nullptr) {
  // Determine variable type and number of components
  int num_components = 1;
  if (pod->type == "VECTORS") num_components = 3;
  else if (pod->type == "TENSORS") num_components = 9;

  // Create openPMD mesh record for this variable
  std::string record_name = pod->name;
  auto mesh_record = it.meshes[record_name];

  // Set mesh geometry
  mesh_record.setGeometry(openPMD::Mesh::Geometry::cartesian);
  mesh_record.setDataOrder(openPMD::Mesh::DataOrder::C);

  // Set grid spacing and labels
  // (depends on coordinate system and block size)

  // For each component
  for (int comp = 0; comp < num_components; comp++) {
    std::string comp_name = (num_components == 1) ?
      openPMD::MeshRecordComponent::SCALAR :
      component_label(comp);  // "x", "y", "z" for vectors

    auto mesh_comp = mesh_record[comp_name];

    // Create dataset with global extent
    auto dataset = openPMD::Dataset(
      openPMD::determineDatatype<opmd_out_t>(),
      global_extent
    );
    mesh_comp.resetDataset(dataset);

    // Write data for each local MeshBlock
    for (int b = 0; b < pm->nblocal; b++) {
      MeshBlock* pmb = pm->my_blocks(b);

      // Calculate chunk offset and extent for this block
      auto [chunk_offset, chunk_extent] = GetChunkOffsetAndExtent(pm, pmb);

      // Copy data to temporary buffer
      std::vector<opmd_out_t> buffer(chunk_extent);
      // ... copy from pod->data to buffer ...

      // Write chunk
      mesh_comp.storeChunk(buffer, chunk_offset, chunk_extent);
    }
  }

  // Flush after each variable
  it.seriesFlush();

  pod = pod->pnext;
}
```

**Variable type mapping:**
- Scalars → single component mesh records
- Vectors → 3-component mesh records with x/y/z components
- Tensors → 9-component mesh records (xx, yy, zz, xy, xz, yx, yz, zx, zy)

##### d. Coordinate Writing

Write mesh coordinates as separate datasets:

```cpp
// For each MeshBlock, write coordinate arrays
// x1f, x2f, x3f (face coordinates)
// x1v, x2v, x3v (volume/cell-center coordinates)

// Set grid spacing
Real dx1 = (pm->mesh_size.x1max - pm->mesh_size.x1min) / pm->mesh_size.nx1;
Real dx2 = (pm->mesh_size.x2max - pm->mesh_size.x2min) / pm->mesh_size.nx2;
Real dx3 = (pm->mesh_size.x3max - pm->mesh_size.x3min) / pm->mesh_size.nx3;

if (pm->ndim == 3) {
  mesh_record.setGridSpacing(std::vector<Real>{dx3, dx2, dx1});
  mesh_record.setAxisLabels({"z", "y", "x"});
  mesh_record.setGridGlobalOffset({
    pm->mesh_size.x3min,
    pm->mesh_size.x2min,
    pm->mesh_size.x1min
  });
}
// Similar for 2D case
```

Handle different coordinate systems:
- **Cartesian**: Direct mapping
- **Cylindrical**: R, phi, z → may need special handling for phi periodicity
- **Spherical polar**: r, theta, phi → may need special handling for angular coordinates

##### e. Slicing Support (based on Parthenon lines 421-428, 774-790)

Support `x1_slice`, `x2_slice`, `x3_slice` output options:

```cpp
// Check if this block intersects with the slice
bool block_in_slice = true;

if (output_params.output_slicex1) {
  if (pmb->block_size.x1min > output_params.x1_slice ||
      pmb->block_size.x1max <= output_params.x1_slice) {
    block_in_slice = false;
  }
}
// Similar for x2_slice and x3_slice

if (!block_in_slice) continue;

// When writing data, reduce dimensions appropriately
// and skip cells outside the slice
```

##### f. Data Buffer Management

Copy data from Athena++ `OutputData` nodes to temporary buffers:

```cpp
// For each block and variable
int index = 0;
std::vector<opmd_out_t> buffer(nx3 * nx2 * nx1);

for (int k = out_ks; k <= out_ke; k++) {
  for (int j = out_js; j <= out_je; j++) {
    for (int i = out_is; i <= out_ie; i++) {
      // Skip if outside slice
      if (is_slice && !in_slice(i, j, k)) continue;

      buffer[index++] = static_cast<opmd_out_t>(pod->data(comp, k, j, i));
    }
  }
}
```

### 3. Modify `src/outputs/outputs.cpp`

In `Outputs::Outputs()` constructor (around line 318-413), add support for openPMD file type:

```cpp
} else if (op.file_type.compare("openpmd") == 0
           || op.file_type.compare("opmd") == 0) {
#ifdef OPENPMDOUTPUT
  // Check if data format is specified
  if (pin->DoesParameterExist(op.block_name, "data_format")) {
    op.data_format = pin->GetString(op.block_name, "data_format");
  } else {
    op.data_format.clear(); // use default
  }

  // Create appropriate output type based on precision
  if (op.data_format.empty()) {
    std::cout << "No data_format specified in output block '"
              << op.block_name << "', using default (double)" << std::endl;
    pnew_type = new OPENPMDOutput<double>(op);
  } else if (type_string_check(base_type::F, 32, op)) {
    std::cout << "Using float32 for openPMD output in block '"
              << op.block_name << "'" << std::endl;
    pnew_type = new OPENPMDOutput<float>(op);
  } else if (type_string_check(base_type::F, 64, op)) {
    std::cout << "Using float64 for openPMD output in block '"
              << op.block_name << "'" << std::endl;
    pnew_type = new OPENPMDOutput<double>(op);
  } else {
    msg << "### FATAL ERROR in Outputs constructor" << std::endl
        << "Unrecognized data_format '" << op.data_format
        << "' in output block '" << op.block_name << "'" << std::endl;
    ATHENA_ERROR(msg);
  }
#else
  msg << "### FATAL ERROR in Outputs constructor" << std::endl
      << "Executable not configured for openPMD outputs, but openPMD file format "
      << "is requested in output block '" << op.block_name << "'" << std::endl;
  ATHENA_ERROR(msg);
#endif
```

### 4. Modify `src/outputs/outputs.hpp`

Add after line 309 (after ATHDF5Output definition):

```cpp
#ifdef OPENPMDOUTPUT
// Forward declaration for openPMD
namespace openPMD {
  class Iteration;
  class Series;
}

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
  void WriteAttributes(openPMD::Iteration& it, Mesh *pm, ParameterInput *pin);
  void WriteVariableData(openPMD::Iteration& it, Mesh *pm);
  std::tuple<std::vector<uint64_t>, std::vector<uint64_t>>
    GetChunkOffsetAndExtent(Mesh *pm, MeshBlock *pmb);
};

// Explicit template instantiations
template class OPENPMDOutput<float>;
template class OPENPMDOutput<double>;
#endif // OPENPMDOUTPUT
```

### 5. Build System Modifications

#### a. CMake Configuration

Add option and dependency detection:

```cmake
# Option to enable openPMD output
option(OPENPMDOUTPUT "Enable openPMD output" OFF)

if(OPENPMDOUTPUT)
  find_package(openPMD REQUIRED)
  target_link_libraries(athena PRIVATE openPMD::openPMD)
  target_compile_definitions(athena PRIVATE OPENPMDOUTPUT)
endif()
```

#### b. Makefile Configuration

Add configuration option:

```make
# OpenPMD output
OPENPMD_OPTION = -DOPENPMDOUTPUT
OPENPMD_LIBS = -lopenPMD
```

### 6. Input File Parameters

Example input block for openPMD output:

```
<output1>
file_type    = openpmd      # or 'opmd'
variable     = prim          # or 'cons', or specific variables
dt           = 0.1           # output cadence
data_format  = float64       # or float32, double (optional, defaults to double)
x2_slice     = 0.0           # optional: output 2D slice at x2=0
</output1>
```

**Supported parameters:**
- `file_type`: `openpmd` or `opmd`
- `variable`: Same as other outputs (prim, cons, individual variables)
- `dt` / `dcycle`: Output cadence
- `data_format`: `float32`, `float64`, `float`, `double` (optional)
- `x1_slice`, `x2_slice`, `x3_slice`: Optional slicing parameters
- `id`: Output identifier (default: `out<N>`)
- `file_number`: Starting file number (default: 0)

**Unsupported parameters:**
- `ghost_zones`: Not supported by openPMD-API (will be ignored with warning)
- `x1_sum`, `x2_sum`, `x3_sum`: Not implemented (can be added as future enhancement)

## Key Adaptations from Parthenon

### 1. No Swarm/Particle Support
Athena++ doesn't have Parthenon's swarm infrastructure. Skip particle output for initial implementation. Future enhancement could add support if Athena++ gains particle capabilities.

### 2. No Params/Package Support
Athena++ doesn't have Parthenon's package system with Params. Skip package parameter output. Input file is written as a string attribute instead.

### 3. Variable Organization
Map Athena++'s OutputData linked list to openPMD mesh records instead of Parthenon's VarInfo system:

- **Athena++**: `OutputData` linked list with `name`, `type`, `data` members
- **Parthenon**: `VarInfo` with component labels and topological elements
- **Mapping**: Use OutputData name as mesh record name, derive component names from type

### 4. AMR Handling
Use Athena++'s `LogicalLocation` structure instead of Parthenon's Forest:

- **Athena++**: `pmb->loc.lx1`, `pmb->loc.lx2`, `pmb->loc.lx3`, `pmb->loc.level`
- **Parthenon**: `pm->Forest().GetLegacyTreeLocation(pmb->loc)`
- Both provide logical location indices and refinement level

### 5. Coordinate Systems
Handle Athena++'s coordinate systems:

- **Cartesian** (`COORDINATE_SYSTEM == "cartesian"`): Direct mapping
- **Cylindrical** (`COORDINATE_SYSTEM == "cylindrical"`): R, φ, z
- **Spherical polar** (`COORDINATE_SYSTEM == "spherical_polar"`): r, θ, φ

Note: Only Cartesian is fully supported by openPMD standard. For curvilinear coordinates, store as rectilinear with coordinate arrays.

### 6. Face-Centered Fields
Special handling for magnetic field components on faces:

- **Cell-centered**: `bcc` (Bcc1, Bcc2, Bcc3) - standard mesh records
- **Face-centered**: `b` (B1, B2, B3) - requires offset in position attribute

Parthenon's `TopologicalElement` system handles this; need to adapt for Athena++'s simpler structure.

### 7. No Ghost Zones
**Critical difference**: openPMD-API does not support ghost zones. The implementation will:
- Ignore the `include_ghost_zones` parameter
- Always output only the interior domain (is:ie, js:je, ks:ke)
- Print a warning if user requests ghost zones

## Testing Strategy

### 1. Unit Tests
- Test basic file creation and series setup
- Test attribute writing (verify all metadata is correct)
- Test single variable output
- Test multiple variables output

### 2. Comparison Tests
Compare openPMD output against HDF5 output for same variables:
- Run simulation with both HDF5 and openPMD outputs
- Load both formats and verify field values match
- Verify coordinate arrays match
- Verify metadata matches

### 3. Visualization Tests
Verify files can be read by visualization tools:
- **ParaView**: Load .bp file with openPMD reader plugin
- **VisIt**: Load with openPMD database plugin
- **Python**: Load with `openpmd-api` Python bindings
- Verify mesh structure and field values display correctly

### 4. MPI Tests
Test parallel I/O with multiple ranks:
- Run on 2, 4, 8 processors
- Verify data is correctly written from all ranks
- Check for race conditions or data corruption
- Verify global extent and local chunks are correct

### 5. AMR Tests
Test with refined meshes at multiple levels:
- Static mesh refinement (SMR)
- Adaptive mesh refinement (AMR) with multiple levels
- Verify logical locations are correct
- Verify data from different levels is correctly positioned

### 6. Slice Tests
Verify x1/x2/x3 slicing works correctly:
- Test each slice direction independently
- Test slices at domain boundaries
- Test slices through refined regions
- Verify output dimensions are reduced correctly

### 7. Coordinate System Tests
Test different coordinate systems:
- Cartesian (standard)
- Cylindrical (R-phi-z)
- Spherical polar (r-theta-phi)
- Verify coordinate arrays are correct
- Verify field values at known positions

### 8. Variable Type Tests
Test different variable types:
- Scalars (density, pressure)
- Vectors (velocity, momentum)
- Tensors (if supported in future)
- Face-centered fields (magnetic field)

## Dependencies

### Required
- **openPMD-API library** (https://github.com/openPMD/openPMD-api)
  - Minimum version: 0.15.0 (recommended: latest)
  - C++14 or later
- **ADIOS2** (recommended backend) or **HDF5** (alternative backend)
  - ADIOS2 recommended for large-scale parallel I/O performance
  - HDF5 for compatibility with existing tools

### Build Instructions

```bash
# With CMake
cmake -DOPENPMDOUTPUT=ON \
      -DopenPMD_DIR=/path/to/openpmd/install \
      ..

# With Makefile (configure.py)
python configure.py \
  --prob=problem_name \
  --coord=cartesian \
  --openpmd \
  --openpmd-dir=/path/to/openpmd/install
```

## File Format Details

### Output Files

**Naming convention:**
```
<basename>.<id>.<number>.bp/
```

Example:
```
turbulence.out1.00000.bp/
turbulence.out1.00001.bp/
turbulence.out1.00002.bp/
```

Note: `.bp` output is a directory containing multiple files (ADIOS2 format)

### File Structure

```
output.00000.bp/
├── Time (attribute: Real)
├── Dt (attribute: Real)
├── NCycle (attribute: int)
├── NumDims (attribute: int)
├── NumMeshBlocks (attribute: int)
├── MaxLevel (attribute: int)
├── RootGridDomain (attribute: Real[9])
├── RootGridSize (attribute: int[3])
├── MeshBlockSize (attribute: int[3])
├── Coordinates (attribute: string)
├── BoundaryConditions (attribute: string[])
├── InputFile (attribute: string)
├── loc.lx123 (attribute: int64[])
├── loc.level-gid-lid-cnghost-gflag (attribute: int[])
├── meshes/
│   ├── rho/
│   │   └── scalar (dataset)
│   ├── press/
│   │   └── scalar (dataset)
│   ├── vel/
│   │   ├── x (dataset)
│   │   ├── y (dataset)
│   │   └── z (dataset)
│   └── Bcc/
│       ├── x (dataset)
│       ├── y (dataset)
│       └── z (dataset)
```

## Future Enhancements

### High Priority
1. **Restart capability**: Read openPMD files for simulation restart
2. **Compression options**: Support ADIOS2/HDF5 compression
3. **Backend configuration**: Allow user-specified backend config files

### Medium Priority
4. **Coarsened outputs**: Support `coarsening_factor` parameter (e.g., output every Nth cell)
5. **Sum outputs**: Support `x1_sum`, `x2_sum`, `x3_sum` parameters
6. **Time series**: Support streaming mode with variable-based iteration encoding

### Low Priority
7. **Particle support**: If Athena++ gains particle capabilities
8. **JSON backend**: For small datasets and debugging
9. **Visualization metadata**: Automatic XDMF generation for legacy tools
10. **Derived quantities**: Support for derived variables computed on-the-fly during output

## References

- **openPMD Standard**: https://github.com/openPMD/openPMD-standard
- **openPMD-API**: https://github.com/openPMD/openPMD-api
- **Parthenon Implementation**: `/Users/benwibking/parthenon_codes/parthenon/src/outputs/parthenon_opmd.cpp`
- **Athena++ HDF5 Output**: `/Users/benwibking/parthenon_codes/athena/src/outputs/athena_hdf5.cpp`
- **ADIOS2**: https://github.com/ornladios/ADIOS2

## Notes

- This implementation is based on analysis of Parthenon's `parthenon_opmd.cpp` and Athena++'s existing output infrastructure
- The design prioritizes compatibility with Athena++'s architecture while maintaining openPMD standard compliance
- Performance should be comparable to or better than HDF5 output when using ADIOS2 backend
- The implementation can coexist with existing output types (HDF5, VTK, restart, etc.)
