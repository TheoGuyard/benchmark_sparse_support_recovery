using El0ps

function solve_el0ps(A, y, M, λ, tolerance)

    solver = BnbSolver(;
        tolgap      = tolerance,
        maxtime     = Inf,
        dualpruning = true,
        l0screening = true,
        l1screening = true,
        verbosity   = false,
        keeptrace   = false,
    )

    F = LeastSquares()
    G = Bigm(M)
    problem = Problem(F, G, A, y, λ)

    result = optimize(solver, problem)

    return result.x
end
