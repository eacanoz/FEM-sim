# Graph Report - FEM-sim  (2026-11-02)

## Corpus Check
- 105 files · ~116,307 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 24 file(s) not represented in the graph (top: .ipynb 11, .log 9, (none) 2)

## Summary
- 1813 nodes · 6045 edges · 118 communities (73 shown, 45 thin omitted)
- Extraction: 78% EXTRACTED · 22% INFERRED · 0% AMBIGUOUS · INFERRED: 1319 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Solves the problem on an n x n mesh. Returns the m
- FactorizedSolver / .__init__() / .solve()
- io / .exportVTK() / Writes the solution for ParaVi
- Example_2D_Poisson.py / Example_2D_Transient.py / 
- ff / .addBC_TractionEdges() / .boundaryForce()
- 2DFEM.py / uniform_mesh() / initial_temperature()
- object / Model / .add_material()
- .solve() / .solverConfiguration() / Configures the
- Enum / run() / scipy_sparse_linalg
- test_time_and_scale.py / fourier() / pipe()
- rheometer_mesh() / channel_mesh() / Example_Gmsh_G
- sympy / test_fluids_foundation.py / configure()
- CarreauYasuda / .__call__() / .__init__()
- Polymer modes [(mu, lam, alpha), ...]: viscosity, 
- scipy_integrate / test_coupling.py / adiabatic_fem
- test_slip_ptt.py / channel() / Stage 17: wall slip
- test_stokes.py / configure() / edges()
- dataclasses / Coupled.py / Several physics solved 
- test_nonlinear_transient.py / decay_error() / mode
- Localized artificial diffusion (setDiffusion) / Ba
- Evaluates the boundary conditions with a time func
- exact() / source() / test_nonlinear_diffusion.py
- .__init__() / .__init__() / .__init__()
- .schur_diagonal() / Lumped pressure mass matrix ov
- build() / .boundary_facets() / Outer-boundary face
- test_species.py / configure() / error_diffusion_re
- DofLayout / .dofs() / .element_dofs()
- test_multifield_diffusion.py / configure_steady() 
- test_triangles_gmsh.py / annulus_quarter() / confi
- darcy / .addBC_NormalVelocityEdges() / ._body_flux
- test_non_newtonian.py / channel() / configure()
- ._add_flow_term() / Viscosity law, convection, vis
- Rheology.py / Bingham / .__call__()
- test_modified_newton.py / cavity() / configure()
- test_sst.py / guess() / profiles()
- channel() / couette() / Uniform shear: b is consta
- jax_numpy / Viscoelastic.py / conformation_law()
- FlowTerm / .contribution() / local_derivative()
- Discretization.py / Geometry of a mesh evaluated a
- .n_elements() / test_axisymmetric.py / configure()
- test_viscoelastic.py / mesh2() / Stage 17, priorit
- Commit history table / Proposed next steps / docs/
- .region_field() / A coefficient by region of an im
- CoupledPhysics / .add_terms() / .derived_fields()
- .setElementShapeFunction() / Records the shape-fun
- datetime / logging / matplotlib
- mt / .addBC_ConcentrationEdges() / .addBC_FluxEdge
- .facets() / FacetSet for the given boundary facets
- at_facets() / facet_mass() / facet_source()
- ShapeFunctionType / Convection-diffusion with a fl
- CrossWLF / .__call__() / .__init__()
- Element / .getCoor() / .getLength()
- jax / MassTransfer.py / Turbulence.py
- jit / _calculate_radiative_heat_coef() / .addBC_Co
- .add_terms() / ._nonlinear() / ._pressure_level_se
- Effective viscosity of k-omega SST, mu + rho a1 k 
- Driver-Seegmiller backward step (SST no result) / 
- Wall that slips: R_u += boundary integral of tau(u
- ConformationTerm / .contribution() / The equation 
- test_analytic.py / build_bar() / Analytic checks f
- core.py Model / FactorizedSolver (PARDISO/SuperLU 
- .add_terms() / .add_terms() / Heat conduction with
- _add_hoop() / expand_components() / Scalar element
- bingham_channel() / cavity() / cold_melt()
- conformation() / homogeneous() / The velocity impo
- CoupledPhysics / DofLayout / Extension guide (phys
- math / matplotlib_cm / matplotlib_colors
- Coupled physics Model(psc=[...]) / Darcy flow (dar
- NavierSlip / .__call__() / .__init__()
- _fv1() / Gives the flow physics of the same model 
- A number (every component) or a vector with one va
- test_vectorized.py / make_model() / Regression che
- Walls (no-slip boundaries): nu~ = 0 on them and th
- _line_shape() / shape() / N (npts, nne) and dN (np
- docs/ARQUITECTURA.md / Discretization / Elements r
- Stage closing criteria / docs/PLAN_FLUIDOS.md / St
- Axisymmetric r-z formulation / Navier-Stokes stead
- cavity_mesh() / heated_cavity() / A species with t
- PowerSlip / .__call__() / .__init__()
- .__init__() / element_size() / lsic_tau()
- numdifftools / scipy_optimize / testNumDiffTools.p
- The steady shear viscosity of a spectrum of modes 
- DiffusionLawTerm / .contribution() / .__init__()
- Reaction-like law with two inputs and two outputs.
- pipe() / Pipe of radius 1 driven by a pressure dif
- .derived_fields() / .__init__() / _pointwise_visco
- Arrhenius / .__call__() / .__init__()
- Positive part of k, omega = exp(g), the blending f
- DirectSolver / .__init__() / .PARDISO()
- R_out += sign * integral of N_i f_out(u), with u t
- run_tests.py / main() / Runs every automated test 
- FacetSet / .__init__() / .n_facets()
- test_lagrange.py / f() / lagrange_interpolation()
- error_to_ghia() / Re = 5000 from rest: the Newton 
- Re_tau = 180 and 395: viscous sublayer u+ = y+, la
- Sign and DOF conventions / Conventions (Celsius, o
- measure() / gradient()
- Spalart-Allmaras from a uniform state: with the ro
- Branch history (backup/vmap-wip, test-branch)
- Design decisions table

## God Nodes (most connected - your core abstractions)
1. `Model` - 258 edges
2. `material` - 230 edges
3. `ff` - 161 edges
4. `ProblemType` - 161 edges
5. `Mesh` - 156 edges
6. `ElementType` - 154 edges
7. `StudyType` - 150 edges
8. `ht` - 120 edges
9. `SolverType` - 93 edges
10. `DirectSolvers` - 87 edges

## Surprising Connections (you probably didn't know these)
- `Conventions (Celsius, outward flux)` --semantically_similar_to--> `Sign and DOF conventions`  [INFERRED] [semantically similar]
  README.md → docs/ARQUITECTURA.md
- `source()` --calls--> `exact()`  [INFERRED]
  Test/test_nonlinear_diffusion.py → Example_2D_Poisson.py
- `solve_poisson()` --uses--> `DirectSolvers`  [INFERRED]
  Example_2D_Poisson.py → Source/enums.py
- `solve_poisson()` --uses--> `ProblemType`  [INFERRED]
  Example_2D_Poisson.py → Source/enums.py
- `solve_poisson()` --uses--> `SolverType`  [INFERRED]
  Example_2D_Poisson.py → Source/enums.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **FEM assembly and solve pipeline** — docs_arquitectura_mesh, docs_arquitectura_elements, docs_arquitectura_discretization, docs_arquitectura_operators, docs_arquitectura_system, docs_arquitectura_model_solver [EXTRACTED 1.00]
- **Fluid transport stages 9-17** — docs_plan_fluidos_etapa_9, docs_plan_fluidos_etapa_10, docs_plan_fluidos_etapa_11, docs_plan_fluidos_etapa_12, docs_plan_fluidos_etapa_13, docs_plan_fluidos_etapa_14, docs_plan_fluidos_etapa_15, docs_plan_fluidos_etapa_16, docs_plan_fluidos_etapa_17 [EXTRACTED 1.00]
- **Nonlinear solver robustness stack for SST** — docs_plan_etapa17_field_scaling, docs_plan_etapa17_pseudo_transient, docs_plan_etapa17_ln_omega, docs_plan_etapa17_modified_newton [INFERRED 0.85]

## Communities (118 total, 45 thin omitted)

### Community 0 - "Solves the problem on an n x n mesh. Returns the m"
Cohesion: 0.09
Nodes (27): solve_poisson(), solve_plate(), build_plate(), poisson(), solve_channel(), ElementType, material, ht (+19 more)

### Community 1 - "FactorizedSolver / .__init__() / .solve()"
Cohesion: 0.06
Nodes (7): FactorizedSolver, _same_sparse(), modelSolver, at_old_level(), step_residual(), step_tangent(), tictoc()

### Community 2 - "io / .exportVTK() / Writes the solution for ParaVi"
Cohesion: 0.07
Nodes (28): _array_xml(), export_model(), write_pvd(), write_vtu(), read_vtu(), test_transient_export_writes_a_time_series(), test_vtu_1d_and_errors(), test_vtu_of_a_steady_2d_solution_round_trips() (+20 more)

### Community 3 - "Example_2D_Poisson.py / Example_2D_Transient.py / "
Cohesion: 0.22
Nodes (12): configure(), solve(), configure(), flow_bcs(), inlet_profile(), solve_steady(), solve(), Example scripts (+4 more)

### Community 4 - "ff / .addBC_TractionEdges() / .boundaryForce()"
Cohesion: 0.06
Nodes (16): ff, cavity(), centerline_u(), configure(), kovasznay(), rect_mesh(), solve_kovasznay(), test_description_follows_set_inertia() (+8 more)

### Community 5 - "2DFEM.py / uniform_mesh() / initial_temperature()"
Cohesion: 0.06
Nodes (29): uniform_mesh(), initial_temperature(), axial_gradient(), flow_rate(), permeability(), plot_mesh_and_field(), bulk_temperature(), inlet_pressure() (+21 more)

### Community 6 - "object / Model / .add_material()"
Cohesion: 0.06
Nodes (11): Model, test_axisymmetric_coupled_tangent(), test_tangent_with_buoyancy(), transport(), test_equal_order_meshes_stay_as_they_were(), test_plane_contraction_entrance_loss(), correction(), test_tangent_with_every_term() (+3 more)

### Community 7 - ".solve() / .solverConfiguration() / Configures the"
Cohesion: 0.11
Nodes (21): buoyant_cavity(), cavity(), channel(), distort(), layer(), rect(), species(), test_adaptive_step_assembles_a_linear_system_once() (+13 more)

### Community 8 - "Enum / run() / scipy_sparse_linalg"
Cohesion: 0.13
Nodes (18): run(), IterativeSolvers, Preconditioners, SolverType, build_preconditioner(), IterativeSolver, saddle_point_preconditioner(), poisson() (+10 more)

### Community 9 - "test_time_and_scale.py / fourier() / pipe()"
Cohesion: 0.13
Nodes (21): fourier(), pipe(), rect(), start_up(), stokes_cavity(), test_adaptive_steps_follow_the_tolerance(), test_adaptive_steps_with_a_time_function_and_errors(), test_block_preconditioner() (+13 more)

### Community 10 - "rheometer_mesh() / channel_mesh() / Example_Gmsh_G"
Cohesion: 0.11
Nodes (12): rheometer_mesh(), channel_mesh(), mesh_with_gmsh(), plate_with_hole(), quarter_annulus(), build(), contraction_mesh(), contraction_mesh() (+4 more)

### Community 11 - "sympy / test_fluids_foundation.py / configure()"
Cohesion: 0.15
Nodes (20): configure(), distorted(), line_mesh(), manufactured_viscous(), on_boundary(), power_law_channel(), square_mesh(), system_with() (+12 more)

### Community 12 - "CarreauYasuda / .__call__() / .__init__()"
Cohesion: 0.14
Nodes (16): CarreauYasuda, configure(), edges(), rect_mesh(), startup(), taylor_green(), taylor_green_exact(), test_boundary_forces_from_the_reactions() (+8 more)

### Community 14 - "scipy_integrate / test_coupling.py / adiabatic_fem"
Cohesion: 0.13
Nodes (16): adiabatic_fem(), adiabatic_ode(), configure_steady(), configure_transient(), finite_difference_jacobian(), reactor(), solve_steady_1d(), test_adiabatic_conservation_is_exact() (+8 more)

### Community 15 - "test_slip_ptt.py / channel() / Stage 17: wall slip"
Cohesion: 0.15
Nodes (14): channel(), rect(), reference(), steady(), test_navier_slip_is_exact(), test_no_slip_and_free_slip_limits(), test_power_slip(), test_ptt_homogeneous_flows() (+6 more)

### Community 16 - "test_stokes.py / configure() / edges()"
Cohesion: 0.17
Nodes (15): configure(), edges(), exact_stokes(), poiseuille(), rect_mesh(), startup_model(), test_errors(), test_exact_stokes_flow_converges() (+7 more)

### Community 18 - "test_nonlinear_transient.py / decay_error() / mode"
Cohesion: 0.15
Nodes (17): decay_error(), mode(), model_2d(), radiation_2d_and_1d(), steady(), test_init_field_with_function_and_postprocess_step(), test_jacobian_2d_radiation_matches_finite_differences(), test_newton_2d_from_bad_initial_guess() (+9 more)

### Community 19 - "Localized artificial diffusion (setDiffusion) / Ba"
Cohesion: 0.10
Nodes (24): Localized artificial diffusion (setDiffusion), Bagley entrance loss ratio (Giesekus vs PTT), Saddle-point block preconditioner (GMRES), Cross-WLF pressure (D3) and region materials, Schafer-Turek cylinder benchmark, Die flows: contraction, capillary, slit, Heat and species validation, Modified Newton in transient (+16 more)

### Community 21 - "exact() / source() / test_nonlinear_diffusion.py"
Cohesion: 0.17
Nodes (17): exact(), source(), configure(), kirchhoff(), l2_error_manufactured(), source(), model_1d(), model_2d() (+9 more)

### Community 22 - ".__init__() / .__init__() / .__init__()"
Cohesion: 0.11
Nodes (3): element_type_of(), scalarField, lawTest

### Community 23 - ".schur_diagonal() / Lumped pressure mass matrix ov"
Cohesion: 0.13
Nodes (7): FieldConvectionTerm, at_gauss(), convection(), _coordinates(), supg_tau(), supg_terms(), _velocity()

### Community 24 - "build() / .boundary_facets() / Outer-boundary face"
Cohesion: 0.11
Nodes (3): build(), Discretization, test_quadratic_mesh_and_geometry()

### Community 25 - "test_species.py / configure() / error_diffusion_re"
Cohesion: 0.17
Nodes (13): configure(), error_diffusion_reaction(), l2_error_nonlinear_reaction(), reaction_first_order(), species_model(), test_clear_errors_for_missing_species_or_diffusivity(), test_convection_diffusion_reaction_exponentials(), test_diffusion_first_order_reaction_converges_with_order_two() (+5 more)

### Community 27 - "test_multifield_diffusion.py / configure_steady() "
Cohesion: 0.20
Nodes (15): configure_steady(), coupled_model(), cross_error(), dirichlet_ends(), finite_difference_jacobian(), line_mesh(), test_concentration_dependent_diffusivity_is_exact_by_kirchhoff(), test_conductivity_depending_on_concentration_converges_to_the_exact_profile() (+7 more)

### Community 28 - "test_triangles_gmsh.py / annulus_quarter() / confi"
Cohesion: 0.20
Nodes (15): annulus_quarter(), configure(), nodes_xy(), poisson_error(), solve_annulus(), test_gmsh_annulus_matches_the_logarithmic_profile_and_converges(), test_gmsh_groups_and_file_round_trip(), test_gmsh_quadrilateral_mesh_runs_with_the_q4_element() (+7 more)

### Community 30 - "test_non_newtonian.py / channel() / configure()"
Cohesion: 0.23
Nodes (13): channel(), configure(), edges(), mu_of_T_couette(), rect_mesh(), test_bingham_channel_has_a_plug(), test_coupled_tangent_against_finite_differences(), test_descriptions_and_errors() (+5 more)

### Community 31 - "._add_flow_term() / Viscosity law, convection, vis"
Cohesion: 0.15
Nodes (8): facet_vector_source(), _gauss_stress(), gauss_stress_divergence(), hoop_stress_terms(), recovered_stress(), recovered_stress_divergence(), stress_divergence(), _stress_rows()

### Community 32 - "Rheology.py / Bingham / .__call__()"
Cohesion: 0.13
Nodes (5): Bingham, Cross, _gamma(), HerschelBulkley, ViscosityLaw

### Community 33 - "test_modified_newton.py / cavity() / configure()"
Cohesion: 0.19
Nodes (11): cavity(), configure(), coupled_flow(), rect_mesh(), residual_matches(), test_a_non_finite_residual_stops_the_simulation(), test_modified_newton_gives_the_same_transient(), test_modified_newton_rebuilds_the_tangent_when_it_stops_working() (+3 more)

### Community 34 - "test_sst.py / guess() / profiles()"
Cohesion: 0.18
Nodes (13): guess(), profiles(), reference_1d(), div(), grad(), residual(), reichardt(), solve_channel() (+5 more)

### Community 35 - "channel() / couette() / Uniform shear: b is consta"
Cohesion: 0.16
Nodes (11): channel(), couette(), steady(), test_a_spurious_root_is_rejected(), test_channel_converges_at_second_order(), test_channel_without_solvent(), test_continuation_in_weissenberg(), test_couette_is_exact() (+3 more)

### Community 36 - "jax_numpy / Viscoelastic.py / conformation_law()"
Cohesion: 0.19
Nodes (9): conformation_law(), law(), conformation_rate(), _plane(), PolymerTractionTerm, traction(), _ptt_factor(), test_ptt_rate_of_b_is_the_rate_of_the_conformation() (+1 more)

### Community 38 - "Discretization.py / Geometry of a mesh evaluated a"
Cohesion: 0.17
Nodes (8): l2_error_poisson(), make_model(), nodal_xy(), test_mesh_topology(), test_neumann_edge_patch(), test_newton_edge_patch(), test_patch_dirichlet_linear_field(), test_poisson_manufactured_order_two()

### Community 39 - ".n_elements() / test_axisymmetric.py / configure()"
Cohesion: 0.24
Nodes (10): configure(), edges(), pipe(), rz_mesh(), test_hagen_poiseuille(), test_measures_and_hoop_operators(), test_power_law_pipe_flow(), test_radial_conduction() (+2 more)

### Community 40 - "test_viscoelastic.py / mesh2() / Stage 17, priorit"
Cohesion: 0.19
Nodes (10): mesh2(), tangent_error(), test_axisymmetric_tangent(), test_giesekus_viscometric_functions(), test_inelastic_equivalent_has_the_same_flow_curve(), test_output_of_the_modes(), test_spectrum_in_oscillatory_shear(), test_tangent() (+2 more)

### Community 41 - "Commit history table / Proposed next steps / docs/"
Cohesion: 0.12
Nodes (16): Commit history table, Proposed next steps, Etapa 17 context: SST challenge, Per-field scaling, ln omega variable, Recommended order and verification, Priority 1: nonlinear solver robustness, Priority 2: robust SST with ln omega (+8 more)

### Community 42 - ".region_field() / A coefficient by region of an im"
Cohesion: 0.19
Nodes (8): law(), layered_mesh(), steady(), test_composite_wall_is_exact(), test_law_values(), test_regions_are_element_indices(), test_two_layer_couette_is_exact(), xy_of()

### Community 44 - ".setElementShapeFunction() / Records the shape-fun"
Cohesion: 0.25
Nodes (11): boundary_layer_1d(), configure(), ends(), line_mesh(), test_galerkin_oscillates_and_supg_does_not(), test_shared_velocity_makes_heat_and_species_follow_the_same_flow(), test_supg_does_nothing_when_diffusion_dominates(), test_supg_is_consistent_second_order_with_a_source() (+3 more)

### Community 45 - "datetime / logging / matplotlib"
Cohesion: 0.19
Nodes (3): setup_logger(), H, configure()

### Community 48 - ".facets() / FacetSet for the given boundary facets"
Cohesion: 0.21
Nodes (9): mesh_1d(), mesh_2d(), test_facet_that_is_not_on_the_boundary_is_rejected(), test_facets_1d_are_points_with_outward_normals(), test_facets_2d_length_and_normals(), test_gradient_of_linear_field(), test_operator_coefficients_as_functions_and_tensors(), test_reference_elements_partition_of_unity() (+1 more)

### Community 49 - "at_facets() / facet_mass() / facet_source()"
Cohesion: 0.15
Nodes (4): at_facets(), facet_mass(), facet_source(), FacetLawTerm

### Community 50 - "ShapeFunctionType / Convection-diffusion with a fl"
Cohesion: 0.19
Nodes (8): ShapeFunctionType, test_convection_2d_matches_1d(), decay_model(), radiation_model(), test_newton_line_search_from_bad_guess(), test_newton_radiation_reference(), test_transient_crank_nicolson_decay(), test_transient_implicit_euler_decay()

### Community 51 - "CrossWLF / .__call__() / .__init__()"
Cohesion: 0.16
Nodes (5): CrossWLF, Exponential, Newtonian, test_temperature_dependent_viscosity_in_couette_flow(), test_viscosity_laws()

### Community 52 - "Element / .getCoor() / .getLength()"
Cohesion: 0.14
Nodes (3): Element, Node, test_reaction_errors()

### Community 53 - "jax / MassTransfer.py / Turbulence.py"
Cohesion: 0.22
Nodes (3): sa_law(), wall_distance(), cached_law()

### Community 55 - ".add_terms() / ._nonlinear() / ._pressure_level_se"
Cohesion: 0.14
Nodes (3): divergence(), pspg_mass(), pspg_weight()

### Community 56 - "Effective viscosity of k-omega SST, mu + rho a1 k "
Cohesion: 0.18
Nodes (3): sst, SSTViscosity, test_turbulent_pipe_against_prandtl()

### Community 57 - "Driver-Seegmiller backward step (SST no result) / "
Cohesion: 0.17
Nodes (13): Driver-Seegmiller backward step (SST no result), Nonlinear globalization validation, Multiple steady states (ignition), Known limitations by stage, Removal of per-element path (Etapa 6c), Quadratic gaps (test_quadratic_gaps, uncommitted), Spalart-Allmaras channel validation, Solvers and output limitations (+5 more)

### Community 59 - "Wall that slips: R_u += boundary integral of tau(u"
Cohesion: 0.17
Nodes (3): WallSlipTerm, build_reaction_law(), PointwiseLaw

### Community 60 - "ConformationTerm / .contribution() / The equation "
Cohesion: 0.19
Nodes (3): ConformationTerm, GradientLawTerm, value_gradient_basis()

### Community 61 - "test_analytic.py / build_bar() / Analytic checks f"
Cohesion: 0.22
Nodes (6): build_bar(), test_linear_profile_dirichlet(), test_linear_profile_iterative_cg(), test_newton_convection_exact(), test_radiation_coefficient_uses_kelvin(), test_solve_without_solver_configuration()

### Community 62 - "core.py Model / FactorizedSolver (PARDISO/SuperLU "
Cohesion: 0.17
Nodes (11): core.py Model, FactorizedSolver (PARDISO/SuperLU reuse), modelSolver (linear, Newton, transient), PointwiseLaw (JAX autodiff), Iterative solvers CG/BiCGSTAB/GMRES with AMG, T3 triangles and gmsh meshes, gmsh, jax / jaxlib (+3 more)

### Community 63 - ".add_terms() / .add_terms() / Heat conduction with"
Cohesion: 0.26
Nodes (6): diffusion(), diffusion_tensor(), gradient_source(), mass(), source(), test_operators_match_known_1d_matrices()

### Community 64 - "_add_hoop() / expand_components() / Scalar element"
Cohesion: 0.26
Nodes (7): _add_hoop(), expand_components(), vector_diffusion(), vector_mass(), vector_source(), viscous(), test_vector_operators()

### Community 65 - "bingham_channel() / cavity() / cold_melt()"
Cohesion: 0.23
Nodes (8): bingham_channel(), cavity(), cold_melt(), configure(), rect_mesh(), sa_channel_from_uniform(), test_bingham_without_a_newtonian_start(), test_field_scales_and_relaxation_weights()

### Community 66 - "conformation() / homogeneous() / The velocity impo"
Cohesion: 0.17
Nodes (6): conformation(), homogeneous(), reference(), rate(), test_homogeneous_flows_against_an_independent_integration(), test_start_up_of_couette()

### Community 67 - "CoupledPhysics / DofLayout / Extension guide (phys"
Cohesion: 0.22
Nodes (11): CoupledPhysics, DofLayout, Extension guide (physics, elements, BC, solvers), FieldConvectionTerm, FlowTerm (Navier-Stokes term), Fluid stages 9-16a as built, GradientLawTerm / add_gradient_law, Physics base class (add_terms, build_system) (+3 more)

### Community 68 - "math / matplotlib_cm / matplotlib_colors"
Cohesion: 0.22
Nodes (6): assemble_displacements(), assemble_forces(), assemble_stiffness(), assign_BCs(), element_stiffness(), update_nodes()

### Community 69 - "Coupled physics Model(psc=[...]) / Darcy flow (dar"
Cohesion: 0.20
Nodes (11): Coupled physics Model(psc=[...]), Darcy flow (darcy), Field-dependent coefficients (depends_on), FEM-sim simulator, Flow-heat coupling and viscous heating, Heat transfer physics (ht), Species transport physics (mt), Natural convection (Boussinesq) (+3 more)

### Community 71 - "NavierSlip / .__call__() / .__init__()"
Cohesion: 0.22
Nodes (4): NavierSlip, PowerLaw, test_mooney_analysis(), apparent()

### Community 72 - "_fv1() / Gives the flow physics of the same model "
Cohesion: 0.18
Nodes (5): _fv1(), law(), SAViscosity, sst_law(), test_eddy_viscosity_law()

### Community 74 - "test_vectorized.py / make_model() / Regression che"
Cohesion: 0.29
Nodes (7): make_model(), test_linear_matrices_match_reference(), test_mass_matrix_matches_reference(), test_nonlinear_residual_and_tangent_match_reference(), test_nonlinear_solution_matches_reference_and_is_fast(), test_tangent_matches_finite_differences(), test_transient_matches_reference()

### Community 76 - "_line_shape() / shape() / N (npts, nne) and dN (np"
Cohesion: 0.20
Nodes (3): _line_shape(), shape(), ReferenceElement

### Community 78 - "docs/ARQUITECTURA.md / Discretization / Elements r"
Cohesion: 0.22
Nodes (6): Discretization, Elements registry, Layered architecture (Mesh to Solvers), Mesh, Operators (weak-form, einsum), Source/ project structure

### Community 79 - "Stage closing criteria / docs/PLAN_FLUIDOS.md / St"
Cohesion: 0.25
Nodes (8): Stage closing criteria, Stage closing criteria, Etapa 10: steady Stokes, Etapa 11: steady Navier-Stokes, Etapa 12: non-Newtonian + thermal coupling, Etapa 9: infrastructure + Darcy, Out of scope: compressible, free surface, multiphase, LES, Polymer processing focus

### Community 80 - "Axisymmetric r-z formulation / Navier-Stokes stead"
Cohesion: 0.25
Nodes (9): Axisymmetric r-z formulation, Navier-Stokes steady (setInertia), Non-Newtonian viscosity laws, Stokes flow with PSPG (ff), Taylor-Hood Q2/Q1 P2/P1, Nonlinear transient BDF2 / adaptive step, RANS turbulence: Spalart-Allmaras and k-omega SST, Viscoelastic multimode (ve): Oldroyd-B, Giesekus, PTT (+1 more)

### Community 81 - "cavity_mesh() / heated_cavity() / A species with t"
Cohesion: 0.25
Nodes (5): cavity_mesh(), heated_cavity(), test_de_vahl_davis_cavity(), test_species_follow_the_computed_flow(), xy_of()

### Community 83 - ".__init__() / element_size() / lsic_tau()"
Cohesion: 0.29
Nodes (3): element_size(), lsic_tau(), pspg_tau()

### Community 84 - "numdifftools / scipy_optimize / testNumDiffTools.p"
Cohesion: 0.47
Nodes (3): fun(), fun_der(), fun_hess()

### Community 90 - "Reaction-like law with two inputs and two outputs."
Cohesion: 0.33
Nodes (3): test_pointwise_law_derivatives_match_finite_differences(), law(), test_pointwise_law_is_compiled_once()

### Community 91 - "pipe() / Pipe of radius 1 driven by a pressure dif"
Cohesion: 0.33
Nodes (3): pipe(), test_pipe(), test_two_identical_modes_are_one()

### Community 95 - "Positive part of k, omega = exp(g), the blending f"
Cohesion: 0.40
Nodes (3): _sst_functions(), law(), _sst_nut()

### Community 101 - "error_to_ghia() / Re = 5000 from rest: the Newton "
Cohesion: 0.50
Nodes (3): error_to_ghia(), test_cavity_at_high_reynolds_from_rest(), xy_of()

## Knowledge Gaps
- **44 isolated node(s):** `Field-dependent coefficients (depends_on)`, `T3 triangles and gmsh meshes`, `Iterative solvers CG/BiCGSTAB/GMRES with AMG`, `ParaView VTK export`, `Darcy flow (darcy)` (+39 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 707 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **45 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ff` connect `ff / .addBC_TractionEdges() / .boundaryForce()` to `Solves the problem on an n x n mesh. Returns the m`, `io / .exportVTK() / Writes the solution for ParaVi`, `Example_2D_Poisson.py / Example_2D_Transient.py / `, `object / Model / .add_material()`, `.solve() / .solverConfiguration() / Configures the`, `test_time_and_scale.py / fourier() / pipe()`, `CarreauYasuda / .__call__() / .__init__()`, `Polymer modes [(mu, lam, alpha), ...]: viscosity, `, `test_slip_ptt.py / channel() / Stage 17: wall slip`, `test_stokes.py / configure() / edges()`, `dataclasses / Coupled.py / Several physics solved `, `.__init__() / .__init__() / .__init__()`, `.schur_diagonal() / Lumped pressure mass matrix ov`, `test_non_newtonian.py / channel() / configure()`, `._add_flow_term() / Viscosity law, convection, vis`, `test_modified_newton.py / cavity() / configure()`, `test_sst.py / guess() / profiles()`, `channel() / couette() / Uniform shear: b is consta`, `.n_elements() / test_axisymmetric.py / configure()`, `test_viscoelastic.py / mesh2() / Stage 17, priorit`, `.region_field() / A coefficient by region of an im`, `datetime / logging / matplotlib`, `.addBC_NoSlipEdges() / .addBC_OutletEdges() / .add`, `ShapeFunctionType / Convection-diffusion with a fl`, `.add_terms() / ._nonlinear() / ._pressure_level_se`, `bingham_channel() / cavity() / cold_melt()`, `conformation() / homogeneous() / The velocity impo`, `NavierSlip / .__call__() / .__init__()`, `A number (every component) or a vector with one va`, `cavity_mesh() / heated_cavity() / A species with t`, `pipe() / Pipe of radius 1 driven by a pressure dif`, `.derived_fields() / .__init__() / _pointwise_visco`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `Model` (e.g. with `DirectSolvers` and `Preconditioners`) actually correct?**
  _`Model` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Field-dependent coefficients (depends_on)`, `T3 triangles and gmsh meshes`, `Iterative solvers CG/BiCGSTAB/GMRES with AMG` to the rest of the system?**
  _44 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Solves the problem on an n x n mesh. Returns the m` be split into smaller, more focused modules?**
  _Cohesion score 0.08748114630467571 - nodes in this community are weakly interconnected._
- **Why does `Model` connect `object / Model / .add_material()` to `Solves the problem on an n x n mesh. Returns the m`, `io / .exportVTK() / Writes the solution for ParaVi`, `Example_2D_Poisson.py / Example_2D_Transient.py / `, `ff / .addBC_TractionEdges() / .boundaryForce()`, `.solve() / .solverConfiguration() / Configures the`, `Enum / run() / scipy_sparse_linalg`, `test_time_and_scale.py / fourier() / pipe()`, `rheometer_mesh() / channel_mesh() / Example_Gmsh_G`, `sympy / test_fluids_foundation.py / configure()`, `CarreauYasuda / .__call__() / .__init__()`, `Polymer modes [(mu, lam, alpha), ...]: viscosity, `, `scipy_integrate / test_coupling.py / adiabatic_fem`, `test_slip_ptt.py / channel() / Stage 17: wall slip`, `test_stokes.py / configure() / edges()`, `test_nonlinear_transient.py / decay_error() / mode`, `exact() / source() / test_nonlinear_diffusion.py`, `build() / .boundary_facets() / Outer-boundary face`, `test_species.py / configure() / error_diffusion_re`, `DofLayout / .dofs() / .element_dofs()`, `test_multifield_diffusion.py / configure_steady() `, `test_triangles_gmsh.py / annulus_quarter() / confi`, `test_non_newtonian.py / channel() / configure()`, `test_modified_newton.py / cavity() / configure()`, `test_sst.py / guess() / profiles()`, `channel() / couette() / Uniform shear: b is consta`, `Discretization.py / Geometry of a mesh evaluated a`, `.n_elements() / test_axisymmetric.py / configure()`, `test_viscoelastic.py / mesh2() / Stage 17, priorit`, `.region_field() / A coefficient by region of an im`, `CoupledPhysics / .add_terms() / .derived_fields()`, `.setElementShapeFunction() / Records the shape-fun`, `datetime / logging / matplotlib`, `ShapeFunctionType / Convection-diffusion with a fl`, `Element / .getCoor() / .getLength()`, `test_analytic.py / build_bar() / Analytic checks f`, `bingham_channel() / cavity() / cold_melt()`, `conformation() / homogeneous() / The velocity impo`, `NavierSlip / .__call__() / .__init__()`, `test_vectorized.py / make_model() / Regression che`, `cavity_mesh() / heated_cavity() / A species with t`, `.postProcess2D() / Filled contour of the nodal sol`, `pipe() / Pipe of radius 1 driven by a pressure dif`?**
  _High betweenness centrality (0.105) - this node is a cross-community bridge._
- **Are the 99 inferred relationships involving `ff` (e.g. with `solve()` and `solve_channel()`) actually correct?**
  _`ff` has 99 INFERRED edges - model-reasoned connections that need verification._
- **Should `FactorizedSolver / .__init__() / .solve()` be split into smaller, more focused modules?**
  _Cohesion score 0.0641025641025641 - nodes in this community are weakly interconnected._