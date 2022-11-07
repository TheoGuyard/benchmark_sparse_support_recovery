# module El0ps

using Dates
using LinearAlgebra
using Printf
using Random

version() = "v0.1"
authors() = "Theo Guyard"
contact() = "theo.guyard@insa-rennes.fr"
license() = "AGPL 3.0"

include("datafits/core.jl")
include("penalties/core.jl")

include("datafits/leastsquares.jl")
include("penalties/bigm.jl")


include("problem.jl")

include("bnb.jl")
include("bounding/core.jl")
include("bounding/accelerations.jl")
include("bounding/cd.jl")

# end