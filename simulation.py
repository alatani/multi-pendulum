import numpy
import numpy.linalg
import scipy
from sympy import symbols
from sympy import Dummy, lambdify
from sympy.physics.mechanics import *

from scipy.integrate import odeint
import matplotlib.pyplot as plt

def createEq():
    q1, q2 = dynamicsymbols('q1 q2')
    q1d, q2d = dynamicsymbols('q1 q2', 1)
    u1, u2 = dynamicsymbols('u1 u2')
    u1d, u2d = dynamicsymbols('u1 u2', 1)
    l, m, g = symbols('l m g')

    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [q1, N.z])
    B = N.orientnew('B', 'Axis', [q2, N.z])

    A.set_ang_vel(N, u1 * N.z)
    B.set_ang_vel(N, u2 * N.z)

    O = Point('O')
    P = O.locatenew('P', l * A.x)
    R = P.locatenew('R', l * B.x)

    O.set_vel(N, 0)
    P.v2pt_theory(O, N, A)
    R.v2pt_theory(P, N, B)

    ParP = Particle('ParP', P, m)
    ParR = Particle('ParR', R, m)

    kd = [q1d - u1, q2d - u2] #位置と速度を紐付ける
    FL = [(P, m * g * N.x), (R, m * g * N.x)]
    BL = [ParP, ParR]

    KM = KanesMethod(N, q_ind=[q1, q2], u_ind=[u1, u2], kd_eqs=kd)

    (fr, frstar) = KM.kanes_equations(FL, BL)


    ## plot
    parameters = [g]                                             # Parameter Definitions
    parameter_vals = [9.81]                                      # First we define gravity

    parameters += [l,m]
    parameter_vals += [1.0, 1.0]  #長さ1m  重さ1kg
    
    dummy_symbols = [Dummy() for i in [q1,q2,u1,u2]]             # Necessary to translate
    dummy_dict = dict(zip([q1,q2,u1,u2], dummy_symbols))                 # out of functions of time

    kds = KM.kindiffdict()                                       # Need to eliminate qdots
    print(kds)
    MM = KM.mass_matrix_full.subs(kds).subs(dummy_dict)          # Substituting away qdots
    Fo = KM.forcing_full.subs(kds).subs(dummy_dict)              # and in dummy symbols
    mm = lambdify(dummy_symbols + parameters, MM)                # The actual call that gets
    fo = lambdify(dummy_symbols + parameters, Fo)                # us to a NumPy function



    def rhs(y, t, args):                                         # Creating the rhs function
        into = numpy.hstack((y, args))                                 # States and parameters
        return numpy.array(numpy.linalg.solve(mm(*into), fo(*into))).T[0]    # Solving for the udots

    init_q = [0,0.2]
    init_u = [-0.1, 0.1]
    y0 = numpy.hstack((init_q, init_u))               # Initial conditions, q and u
    t = numpy.linspace(0, 10, 1000)                                    # Time vector
    y = odeint(rhs, y0, t, args=(parameter_vals,))               # Actual integration



    
    f, ax = plt.subplots(2, sharex=True, sharey=False)
    f.set_size_inches(6.5, 6.5)
    
    # PLOTTING
    ax[0].plot(t, y[:, 0], label='$q_' + str(0) + '$')
    ax[1].plot(t, y[:, 0 + 2], label='$u_' + str(0) + '$')
    ax[0].plot(t, y[:, 1], label='$q_' + str(1) + '$')
    ax[1].plot(t, y[:, 1 + 2], label='$u_' + str(1) + '$')
    
    #figure(1)
    ax[0].legend(loc=0)
    ax[1].legend(loc=0)
    ax[1].set_xlabel('Time [s]')
    ax[0].set_ylabel('Angle [rad]')
    ax[1].set_ylabel('Angular rate [rad/s]')

    f.subplots_adjust(hspace=0)
    plt.setp(ax[0].get_xticklabels(), visible=False)
    plt.tight_layout()
    plt.savefig('four_link_pendulum_time_series.eps') 

    return KM

if __name__ == "__main__":
    #from sympy import *
    createEq()
    pass
