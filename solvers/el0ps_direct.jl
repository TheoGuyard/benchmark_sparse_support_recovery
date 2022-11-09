using El0ps
using SCIP

function solve_el0ps_direct(A, y, M, λ, tolerance, optimizer)

    if optimizer == "scip"
        optimizer = SCIP.Optimizer
        options = Dict(
            "display/verblevel" => 0, 
            "limits/gap" => tolerance,
        )
    else
        error("Unsupported optimizer $optimizer")
    end

    solver = DirectSolver(optimizer, options=options)

    F = LeastSquares()
    G = Bigm(M)
    problem = Problem(F, G, A, y, λ)

    result = optimize(solver, problem)

    return result.x, result.relative_gap
end
