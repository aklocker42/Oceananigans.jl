#####
##### TracerReservoir open boundary scheme
#####

"""
    TracerReservoir(; inflow_length_scale = 0, outflow_length_scale = 0)

Open boundary condition for tracers that carries a reservoir value `cʳ` just outside each
boundary point, after the tracer reservoirs of MOM6. Each step the flow advects a distance
`d = |uₙ| Δt` across the boundary face, and the reservoir relaxes toward the upstream value
over a length scale `L`:

    cʳ ← (cʳ + a c★) / (1 + a),    a = d / L

On outflow `c★` is the adjacent interior value and `L = outflow_length_scale`; on inflow `c★`
is the exterior value `cᵉˣᵗ` and `L = inflow_length_scale`. The boundary halo value is set to
`cʳ`, the advecting velocity is taken at the boundary face, and the reservoir starts at `cᵉˣᵗ`.

`L = 0` sets `cʳ = c★` every step, `L = Inf` freezes the reservoir, and a finite `L` relaxes
it. With both length scales `0`, the default, the condition is a memoryless
`ValueBoundaryCondition`: zero-gradient on outflow and `cᵉˣᵗ` on inflow.

`TracerReservoir` is a `Value` scheme for `Center`-located fields.

```jldoctest
using Oceananigans
using Oceananigans.BoundaryConditions: TracerReservoir

TracerReservoir(inflow_length_scale = 20000)

# output
TracerReservoir{Float64}
├── inflow_length_scale: 20000.0
└── outflow_length_scale: 0.0
```
"""
struct TracerReservoir{FT, S, A} <: AbstractRadiationScheme{FT}
    inflow_length_scale  :: FT
    outflow_length_scale :: FT
    cʳ  :: S  # anchor reservoir value (2D array or nothing)
    cʳˡ :: S  # latest reservoir value (2D array or nothing)
    anchors :: A # iteration of the last anchored fill at each boundary point (2D array or nothing)
end

function TracerReservoir(FT = defaults.FloatType;
                         inflow_length_scale = 0,
                         outflow_length_scale = 0)

    inflow_length_scale  = convert(FT, inflow_length_scale)
    outflow_length_scale = convert(FT, outflow_length_scale)

    inflow_length_scale  >= 0 || throw(ArgumentError("inflow_length_scale must be non-negative"))
    outflow_length_scale >= 0 || throw(ArgumentError("outflow_length_scale must be non-negative"))

    return TracerReservoir(inflow_length_scale, outflow_length_scale, nothing, nothing, nothing)
end

Adapt.adapt_structure(to, r::TracerReservoir) =
    TracerReservoir(adapt(to, r.inflow_length_scale),
                    adapt(to, r.outflow_length_scale),
                    adapt(to, r.cʳ),
                    adapt(to, r.cʳˡ),
                    adapt(to, r.anchors))

Base.summary(::TracerReservoir{FT}) where FT = "TracerReservoir{$FT}"

function Base.show(io::IO, r::TracerReservoir)
    print(io, summary(r), '\n')
    print(io, "├── inflow_length_scale: ",  prettysummary(r.inflow_length_scale), '\n')
    print(io, "└── outflow_length_scale: ", prettysummary(r.outflow_length_scale))
end

const TRVBC = BoundaryCondition{<:Value{<:TracerReservoir}}

#####
##### Storage allocation during BC regularization
#####

radiation_buffers(reservoir::TracerReservoir, arch, FT, tangential_size) =
    (ntuple(_ -> zeros(arch, FT, tangential_size...), 2)..., anchor_buffer(arch, tangential_size)) # cʳ, cʳˡ, anchors

radiation_buffers(reservoir::TracerReservoir) = (reservoir.cʳ, reservoir.cʳˡ, reservoir.anchors)

radiation_storage(reservoir::TracerReservoir, (cʳ, cʳˡ, anchors)) =
    TracerReservoir(reservoir.inflow_length_scale, reservoir.outflow_length_scale, cʳ, cʳˡ, anchors)

#####
##### The reservoir update
#####

# Backward-Euler relaxation of the reservoir toward the upstream value c★ over the length scale L.
@inline function reservoir_update(cʳ, cᴵ, cᵉˣᵗ, d, outflow, reservoir)
    L  = ifelse(outflow, reservoir.outflow_length_scale, reservoir.inflow_length_scale)
    c★ = ifelse(outflow, cᴵ, cᵉˣᵗ)
    a  = d / L
    return ifelse(L == 0, c★, (cʳ + a * c★) / (1 + a))
end

# `cʳ` anchors the reservoir at the start of the time step; every stage relaxes from it with
# its own `Δt` into `cʳˡ`, which becomes next step's anchor.
@inline function reservoir_halo!(cᵇ, cᴵ, l, m, grid, c, bc, uₙ, outflow, closed, clock, model_fields)
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    reservoir = bc.classification.scheme
    anchored = anchored_fill(clock, reservoir.anchors, l, m)

    @inbounds begin
        cᵉˣᵗ = getbc(bc, l, m, grid, clock, model_fields)

        cʳᵃ = ifelse(anchored, reservoir.cʳˡ[l, m], reservoir.cʳ[l, m])
        cʳⁿ = ifelse(first_call, cᵉˣᵗ, cʳᵃ)

        d = abs(uₙ) * Δt
        cʳⁿ⁺¹ = reservoir_update(cʳⁿ, c[cᴵ...], cᵉˣᵗ, d, outflow, reservoir)

        c[cᵇ...] = ifelse(closed, zero(grid), cʳⁿ⁺¹)
        reservoir.cʳ[l, m]  = cʳⁿ   # anchor for later stages
        reservoir.cʳˡ[l, m] = cʳⁿ⁺¹ # latest, promoted at the next anchored fill
        record_anchor!(reservoir.anchors, l, m, clock, anchored, first_call)
    end

    return nothing
end

#####
##### Halo filling — Center-located fields on the six boundaries
#####

# A field without a vertical location (eg sea ice on the ocean grid) is closed where the surface cell is.
@inline closure_level(k, grid, ℓz) = k
@inline closure_level(k, grid, ::Nothing) = size(grid, 3)

@inline function _fill_east_halo!(j, k, grid, c, bc::TRVBC, loc::CAA, clock, model_fields)
    ℓx, ℓy, ℓz = loc
    i = grid.Nx + 1
    uₙ = @inbounds model_fields.u[i, j, k]
    closed = immersed_peripheral_node(grid.Nx, j, closure_level(k, grid, ℓz), grid, Center(), ℓy, ℓz)
    return reservoir_halo!((i, j, k), (i-1, j, k), j, k, grid, c, bc, uₙ, uₙ >= 0, closed, clock, model_fields)
end

@inline function _fill_west_halo!(j, k, grid, c, bc::TRVBC, loc::CAA, clock, model_fields)
    ℓx, ℓy, ℓz = loc
    uₙ = @inbounds model_fields.u[1, j, k]
    closed = immersed_peripheral_node(1, j, closure_level(k, grid, ℓz), grid, Center(), ℓy, ℓz)
    return reservoir_halo!((0, j, k), (1, j, k), j, k, grid, c, bc, uₙ, uₙ <= 0, closed, clock, model_fields)
end

@inline function _fill_north_halo!(i, k, grid, c, bc::TRVBC, loc::ACA, clock, model_fields)
    ℓx, ℓy, ℓz = loc
    j = grid.Ny + 1
    uₙ = @inbounds model_fields.v[i, j, k]
    closed = immersed_peripheral_node(i, grid.Ny, closure_level(k, grid, ℓz), grid, ℓx, Center(), ℓz)
    return reservoir_halo!((i, j, k), (i, j-1, k), i, k, grid, c, bc, uₙ, uₙ >= 0, closed, clock, model_fields)
end

@inline function _fill_south_halo!(i, k, grid, c, bc::TRVBC, loc::ACA, clock, model_fields)
    ℓx, ℓy, ℓz = loc
    uₙ = @inbounds model_fields.v[i, 1, k]
    closed = immersed_peripheral_node(i, 1, closure_level(k, grid, ℓz), grid, ℓx, Center(), ℓz)
    return reservoir_halo!((i, 0, k), (i, 1, k), i, k, grid, c, bc, uₙ, uₙ <= 0, closed, clock, model_fields)
end

@inline function _fill_top_halo!(i, j, grid, c, bc::TRVBC, loc::AAC, clock, model_fields)
    ℓx, ℓy, ℓz = loc
    k = grid.Nz + 1
    uₙ = @inbounds model_fields.w[i, j, k]
    closed = immersed_peripheral_node(i, j, grid.Nz, grid, ℓx, ℓy, Center())
    return reservoir_halo!((i, j, k), (i, j, k-1), i, j, grid, c, bc, uₙ, uₙ >= 0, closed, clock, model_fields)
end

@inline function _fill_bottom_halo!(i, j, grid, c, bc::TRVBC, loc::AAC, clock, model_fields)
    ℓx, ℓy, ℓz = loc
    uₙ = @inbounds model_fields.w[i, j, 1]
    closed = immersed_peripheral_node(i, j, 1, grid, ℓx, ℓy, Center())
    return reservoir_halo!((i, j, 0), (i, j, 1), i, j, grid, c, bc, uₙ, uₙ <= 0, closed, clock, model_fields)
end

#####
##### Corner filling (issue #3342: Value BCs never fill the corner two adjacent sides
##### share). Reads the two already-filled edge-halo neighbors one step in from the
##### corner — not the reservoir formula itself — so the corner carries no persistent
##### memory of its own and cannot accumulate error the way the reservoir relaxation
##### does along an edge. Runs every RK3 substage as part of the normal halo fill for
##### this field, right after the west/east/south/north kernels above.
#####

@kernel function _fill_southwest_corner_halo!(c, west_bc::TRVBC, south_bc::TRVBC, loc, grid, args)
    k = @index(Global, Linear)
    ℓx, ℓy, ℓz = loc
    cʷ = @inbounds c[0, 1, k]
    cˢ = @inbounds c[1, 0, k]
    closed = immersed_peripheral_node(1, 1, closure_level(k, grid, ℓz), grid, Center(), Center(), ℓz)
    @inbounds c[0, 0, k] = ifelse(closed, zero(grid), (cʷ + cˢ) / 2)
end

@kernel function _fill_southeast_corner_halo!(c, east_bc::TRVBC, south_bc::TRVBC, loc, grid, args)
    k = @index(Global, Linear)
    ℓx, ℓy, ℓz = loc
    Nx = grid.Nx
    cᵉ = @inbounds c[Nx+1, 1, k]
    cˢ = @inbounds c[Nx,   0, k]
    closed = immersed_peripheral_node(Nx, 1, closure_level(k, grid, ℓz), grid, Center(), Center(), ℓz)
    @inbounds c[Nx+1, 0, k] = ifelse(closed, zero(grid), (cᵉ + cˢ) / 2)
end

@kernel function _fill_northwest_corner_halo!(c, west_bc::TRVBC, north_bc::TRVBC, loc, grid, args)
    k = @index(Global, Linear)
    ℓx, ℓy, ℓz = loc
    Ny = grid.Ny
    cʷ = @inbounds c[0, Ny,   k]
    cⁿ = @inbounds c[1, Ny+1, k]
    closed = immersed_peripheral_node(1, Ny, closure_level(k, grid, ℓz), grid, Center(), Center(), ℓz)
    @inbounds c[0, Ny+1, k] = ifelse(closed, zero(grid), (cʷ + cⁿ) / 2)
end

@kernel function _fill_northeast_corner_halo!(c, east_bc::TRVBC, north_bc::TRVBC, loc, grid, args)
    k = @index(Global, Linear)
    ℓx, ℓy, ℓz = loc
    Nx, Ny = grid.Nx, grid.Ny
    cᵉ = @inbounds c[Nx+1, Ny,   k]
    cⁿ = @inbounds c[Nx,   Ny+1, k]
    closed = immersed_peripheral_node(Nx, Ny, closure_level(k, grid, ℓz), grid, Center(), Center(), ℓz)
    @inbounds c[Nx+1, Ny+1, k] = ifelse(closed, zero(grid), (cᵉ + cⁿ) / 2)
end

@inline function corner_fill_kernels(bcs::FieldBoundaryConditions{W, E, S, N}, data, grid, loc, indices) where {W <: TRVBC, E <: TRVBC, S <: TRVBC, N <: TRVBC}
    arch = architecture(grid)
    _, Sz = fill_halo_size(data, West(), indices, bcs.west, loc, grid)
    size, offset = (Sz,), (0,)
    reduced_dims = reduced_dimensions(loc)

    sw = configure_kernel(arch, grid, kernel_parameters(size, offset), _fill_southwest_corner_halo!; reduced_dimensions=reduced_dims)[1]
    se = configure_kernel(arch, grid, kernel_parameters(size, offset), _fill_southeast_corner_halo!; reduced_dimensions=reduced_dims)[1]
    nw = configure_kernel(arch, grid, kernel_parameters(size, offset), _fill_northwest_corner_halo!; reduced_dimensions=reduced_dims)[1]
    ne = configure_kernel(arch, grid, kernel_parameters(size, offset), _fill_northeast_corner_halo!; reduced_dimensions=reduced_dims)[1]

    kernels = (southwest = sw, southeast = se, northwest = nw, northeast = ne)
    corner_bcs = (southwest = (bcs.west, bcs.south), southeast = (bcs.east, bcs.south),
                  northwest = (bcs.west, bcs.north), northeast = (bcs.east, bcs.north))

    return kernels, corner_bcs
end
