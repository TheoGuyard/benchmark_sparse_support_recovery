using El0ps

function solve_el0ps_bnb(A, y, M, λ, tolerance, acceleration)

    solver = BnbSolver(;
        tolgap      = tolerance,
        maxtime     = Inf,
        dualpruning = acceleration,
        l0screening = acceleration,
        l1screening = acceleration,
        verbosity   = false,
        keeptrace   = false,
    )

    F = LeastSquares()
    G = Bigm(M)
    problem = Problem(F, G, A, y, λ)

    result = optimize(solver, problem)

    return result.x, result.relative_gap
end