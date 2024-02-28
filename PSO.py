import numpy as np
import random as rn
import time



def PSO(val, objfun, x_min, x_max, itermax):
    N, D = val.shape[0], val.shape[1]
    f = np.zeros((N))

    # Initialization of PSO parameters
    c1 = 2
    c2 = 2
    wmax = 0.9
    wmin = 0.1
    w = np.zeros((itermax, 1))
    for iter in range(itermax):
        w[iter] = wmax - ((wmax - wmin) / itermax) * iter  # Inertia weight update

    m = x_min[0]
    n = x_max[0]
    q = (n - m) / (D * 2)
    Ki = 1

    # Random initialization of position and velocity
    x = val
    v = q * np.random.rand(N, D)

    for i in range(N):
        f[i] = objfun(x[i, :])

    # fgbest = min(f)
    fgbest = np.amin(f)
    igbest = np.where(np.amin(f) == fgbest)
    gbest = x[igbest, :]
    pbest = x
    fpbest = f

    fbst = np.zeros((itermax))
    ct = time.time()

    # Iterate
    for it in range(itermax):
        print(it)

        # Update velocities and position
        v = w[it] * v + c1 * rn.random() * (pbest - x) + c2 * rn.random()
        x = x + v

        for mi in range(N):
            for mj in range(D):
                if x[mi, mj] < x_min[mi, mj]:
                    x[mi, mj] = x_min[mi, mj]
                else:
                    if x[mi, mj] > x_max[mi, mj]:
                        x[mi, mj] = x_max[mi, mj]

        for i in range(N):
            f[i] = objfun(x[i, :])

        # Find global best and Particle best
        minf = np.amin(f)
        iminf = np.where(np.amin(f) == minf)
        a = np.mean(minf)
        b = np.mean(fgbest)
        if a <= b:
            fgbest = minf
            gbest = x[iminf, :]
            best_sub = x[iminf, :]
            fbst[it] = minf
        else:
            fbst[it] = fgbest
            best_sub = gbest

        '''inewpb = np.where(f <= fpbest)
        pbest[inewpb, :] = x[inewpb[1], :]
        fpbest[inewpb] = f[inewpb]'''
    ct = time.time() - ct
    best_fit = fbst[itermax - 1]

    return best_fit, fbst, best_sub.ravel(), ct
