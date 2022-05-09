def ackley(x1, x2):
    return f"- 20* exp(- 0.2* sqrt(1 / 2 * (x1 ** 2 + x2 ** 2))) - exp(1 / 2 * (cos( 2 * pi* x1) + cos( 2 * pi* x2))) +  20+ exp(1)"


ackley_restr_eq = ['x1+x2=0', '2x1-3x2=0']
ackley_restr_ineq = ['x1+x2<=2', '2x1-3x2<1']
ackley_restr_eq_ineq = ['x1+x2=0', '2x1-3x2<1']


def bukin(x1, x2):
    return f"100 * sqrt(abs(x2 - 0.01 * x1 ** 2)) + 0.01 * abs(x1 + 10)"


bukin_restr_eq = ['x1+x2=-9', '2x1-3x2=-23']
bukin_restr_ineq = ['x1+x2>=-9', '2x1-3x2>=-23']
bukin_restr_eq_ineq = ['x1+x2=-9', '2x1-3x2>-24']


def cross_in_tray(x1, x2):
    return f"-0.0001 * (abs(sin(x1) * sin(x2) * exp(abs(100 - sqrt(x1 ** 2 + x2 ** 2) / pi))) + 1) ** 0.1"


cross_in_tray_restr_eq = ['x1-x2=0 ', '2x1+3x2=6.7455']
cross_in_tray_restr_ineq = ['x1-x2>=0 ', '2x1+3x2<7.7455']
cross_in_tray_restr_eq_ineq = ['x1-x2=0 ', '2x1+3x2>0']


def drop_wave(x1, x2):
    return f"- (1 + cos(12 * sqrt(x1 ** 2 + x2 ** 2))) / (0.5 * (x1 ** 2 + x2 ** 2) + 2)"


drop_wave_restr_eq = ['x1+x2=0', '2x1-3x2=0']
drop_wave_restr_ineq = ['x1+x2<=2', '2x1-3x2<1']
drop_wave_restr_eq_ineq = ['x1+x2=0', '2x1-3x2<1']


def eggholder(x1, x2):
    return f"-(x2 + 47) * sin(sqrt(abs(x2 + x1 / 2 + 47))) - x1 * sin(sqrt(abs(x1 - (x2 + 47))))"


eggholder_restr_eq = ['x1+x2=916.2319', '2x1-3x2=-188.6957']
eggholder_restr_ineq = ['x1+x2<=1000', '2x1-3x2<-160']
eggholder_restr_eq_ineq = ['x1+x2=916.2319', '2x1-3x2<-160']


def griewank(x1, x2):
    return f"(x1 ** 2 + x2 ** 2) / 4000 - cos(x1 / sqrt(1)) * cos(x2 / sqrt(2)) + 1"


griewank_restr_eq = ['x1+x2=0', '2x1-3x2=0']
griewank_restr_ineq = ['x1+x2<=2', '2x1-3x2<1']
griewank_restr_eq_ineq = ['x1+x2=0', '2x1-3x2<1']


def holder_table(x1, x2):
    return f"-abs(sin(x1) * cos(x2) * exp(abs(1 - sqrt(x1 ** 2 + x2 ** 2) / pi)))"


holder_restr_eq = ['x1+x2=17.71961', '2x1-3x2=-12.88373']
holder_restr_ineq = ['x1+x2<=18', '2x1-3x2<-12']
holder_restr_eq_ineq = ['x1+x2=17.71961', '2x1-3x2<-10']


def levy(x1, x2):
    return f"sin(pi *  1 + (x1 - 1) / 4) ** 2 + ( 1 + (x1 - 1) / 4- 1) ** 2 * (1 + 10 * sin(pi *  1 + (x1 - 1) / 4+ 1) ** 2) + ( 1 + (x2 - 1) / 4- 1) ** 2 * (1 + 10 * sin(2 * pi *  1 + (x2 - 1) / 4) ** 2)"


levy_restr_eq = ['x1+x2=2', '2x1-3x2=-1']
levy_restr_ineq = ['x1+x2<=4', '2x1-3x2<1']
levy_restr_eq_ineq = ['x1+x2=2', '2x1-3x2<1']


def levy13(x1, x2):
    return f"sin(3 * pi * x1) ** 2 + (x1 - 1) ** 2 * (1 + sin(3 * pi * x1) ** 2) + (x2 - 1) ** 2 * (1 + sin(2 * pi * x2) ** 2)"


levy13_restr_eq = ['x1+x2=2', '2x1-3x2=-1']
levy13_restr_ineq = ['x1+x2<=4', '2x1-3x2<1']
levy13_restr_eq_ineq = ['x1+x2=2', '2x1-3x2<1']


def rastrigin(x1, x2):
    return f"10 * 2 + (x1 ** 2 - 10 * cos(2 * pi * x1)) + (x2 ** 2 - 10 * cos(2 * pi * x2))"


rastrigin_restr_eq = ['x1+x2=0', '2x1-3x2=0']
rastrigin_restr_ineq = ['x1+x2<=2', '2x1-3x2<1']
rastrigin_restr_eq_ineq = ['x1+x2=0', '2x1-3x2<1']


def schaffer(x1, x2):
    return f"0.5 + (sin(x1 ** 2 - x2 ** 2) ** 2 - 0.5) / (1 + 0.001 * (x1 ** 2 + x2 ** 2)) ** 2"


schaffer_restr_eq = ['x1+x2=0', '2x1-3x2=0']
schaffer_restr_ineq = ['x1+x2<=2', '2x1-3x2<1']
schaffer_restr_eq_ineq = ['x1+x2=0', '2x1-3x2<1']


def schwefel(x1, x2):
    return f"418.9829 * 2 - x1 * sin(sqrt(abs(x1))) - x2 * sin(sqrt(abs(x2)))"


schwefel_restr_eq = ['x1+x2=841.9474', '2x1-3x2=-420.9687']
schwefel_restr_ineq = ['x1+x2<=1000', '2x1-3x2<-100']
schwefel_restr_eq_ineq = ['x1+x2=841.9474', '2x1-3x2<-100']


def bocharevsky(x1, x2):
    return f"x1 ** 2 + 2 * x2 ** 2 - 0.3 * cos(3 * pi * x1) - 0.4 * cos(4 * pi * x2) + 0.7"


bocharevsky_restr_eq = ['x1+x2=0', '2x1-3x2=0']
bocharevsky_restr_ineq = ['x1+x2<=2', '2x1-3x2<1']
bocharevsky_restr_eq_ineq = ['x1+x2=0', '2x1-3x2<1']


def perm(x1, x2):
    return f"((1 +  0.5) * (x1 ** 1 - 1 / (1) ** 1) + (2 +  0.5) * (x2 ** 1 - 1 / (2) ** 1)) ** 2 + ((1 +  0.5) * (x1 ** 2 - 1 / (1) ** 2) + (2 +  0.5) * (x2 ** 2 - 1 / (2) ** 2)) ** 2"


perm_restr_eq = ['x1+x2=0.5', '2x1-3x2=-1.5']
perm_restr_ineq = ['x1+x2<=1', '2x1-3x2<0']
perm_restr_eq_ineq = ['x1+x2=0.5', '2x1-3x2<0']


def diff_power(x1, x2):
    return f"abs(x1) ** 2 + abs(x2) ** 3"


diff_power_restr_eq = ['x1+x2=0', '2x1-3x2=0']
diff_power_restr_ineq = ['x1+x2<=2', '2x1-3x2<1']
diff_power_restr_eq_ineq = ['x1+x2=0', '2x1-3x2<1']


def trid(x1, x2):
    return f"(x1 - 1) ** 2 + (x2 - 1) ** 2 - x1 * x2"


trid_restr_eq = ['x1+x2=8', '2x1-3x2=-14']
trid_restr_ineq = ['x1+x2<=10', '2x1-3x2<0']
trid_restr_eq_ineq = ['x1+x2=8', '2x1-3x2<0']


def booth(x1, x2):
    return f"(x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2"


booth_restr_eq = ['x1+x2=4', '2x1-3x2=-7']
booth_restr_ineq = ['x1+x2<=10', '2x1-3x2<0']
booth_restr_eq_ineq = ['x1+x2=4', '2x1-3x2<0']


def matyas(x1, x2):
    return f"0.26 * (x1 ** 2 + x2 ** 2) - 0.48 * x1 * x2"


matyas_restr_eq = ['x1+x2=0', '2x1-3x2=0']
matyas_restr_ineq = ['x1+x2<=2', '2x1-3x2<1']
matyas_restr_eq_ineq = ['x1+x2=0', '2x1-3x2<1']


def mccormick(x1, x2):
    return f"sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1"


mccormick_restr_eq = ['x1+x2=-2.09438', '2x1+3x2=-5.73595']
mccormick_restr_ineq = ['x1+x2<=2', '2x1-3x2<0']
mccormick_restr_eq_ineq = ['x1+x2=-2.09438', '2x1-3x2<0']


def zakharov(x1, x2):
    return f"x1 ** 2 + x2 ** 2 + (0.5 * 1 * x1 + 0.5 * 2 * x2) ** 2 + (0.5 * 1 * x1 + 0.5 * 2 * x2) ** 4"


zakharov_restr_eq = ['x1+x2=0', '2x1-3x2=0']
zakharov_restr_ineq = ['x1+x2<=2', '2x1-3x2<1']
zakharov_restr_eq_ineq = ['x1+x2=0', '2x1-3x2<1']


def three_hump(x1, x2):
    return f"2 * x1 ** 2 - 1.05 * x1 ** 4 + x1 ** 6 / 6 + x1 * x2 + x2 ** 2"


three_hump_restr_eq = ['x1+x2=0', '2x1-3x2=0']
three_hump_restr_ineq = ['x1+x2<=2', '2x1-3x2<1']
three_hump_restr_eq_ineq = ['x1+x2=0', '2x1-3x2<1']


def six_hump(x1, x2):
    return f"(4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2"


six_hump_restr_eq = ['x1+x2=-0.6228', '2x1-3x2=-1.9582']
six_hump_restr_ineq = ['x1+x2<=2', '2x1-3x2<1']
six_hump_restr_eq_ineq = ['x1+x2=-0.6228', '2x1-3x2<1']


def dixon_price(x1, x2):
    return f"(x1 - 1) ** 2 + 2 * (2 * x2 ** 2 - x1) ** 2"


dixon_price_restr_eq = ['x1+x2=1.7071', '2x1-3x2=-0.12132']
dixon_price_restr_ineq = ['x1+x2<=2', '2x1-3x2<1']
dixon_price_restr_eq_ineq = ['x1+x2=1.7071', '2x1-3x2<1']


def rosenbrock(x1, x2):
    return f"100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2"


rosenbrock_restr_eq = ['x1+x2=2', '2x1-3x2=-1']
rosenbrock_restr_ineq = ['x1+x2<=4', '2x1-3x2<1']
rosenbrock_restr_eq_ineq = ['x1+x2=2', '2x1-3x2<1']


def easom(x1, x2):
    return f"-cos(x1) * cos(x2) * exp(-(x1 - pi) ** 2 - (x2 - pi) ** 2)"


easom_restr_eq = ['x1+x2=2*np.pi', '2x1-3x2=-np.pi']
easom_restr_ineq = ['x1+x2<=10', '2x1-3x2<1']
easom_restr_eq_ineq = ['x1+x2=2*np.pi', '2x1-3x2<1']


def michalewicz(x1, x2):
    return f"-(sin(x1)*sin(x1**2/pi)**(2*10))-(sin(x2)*sin(2*x2**2/pi)**(2*10))"


michalewicz_restr_eq = ['x1+x2=3.77', '2x1-3x2=-0.31']
michalewicz_restr_ineq = ['x1+x2<=4', '2x1-3x2<1']
michalewicz_restr_eq_ineq = ['x1+x2=3.77', '2x1-3x2<1']

ackley_point_min = "0; 0"
ackley_point_start = "-2; 2"

bukin_point_min = "-10; 1"
bukin_point_start = "-5; -4"

cross_in_tray_point_min = "1.3491; 1.3491"
cross_in_tray_point_start = "-1; 1"

drop_wave_point_min = "0; 0"
drop_wave_point_start = "-2; 2"

eggholder_point_min = "512; 404.2319"
eggholder_point_start = "300; -616.2319"

griewank_point_min = "0; 0"
griewank_point_start = "-2; 2"

holder_table_point_min = "8.05502; 9.66459"
holder_table_point_start = "1.71961; 16"

levy_point_min = "1; 1"
levy_point_start = "-1; 3"

levy13_point_min = "1; 1"
levy13_point_start = "-1; 3"

rastrigin_point_min = "0; 0"
rastrigin_point_start = "-2; 2"

schaffer_point_min = "0; 0"
schaffer_point_start = "-2; 2"

schwefel_point_min = "420.9687; 420.9687"
schwefel_point_start = "1; 840.9474"

bocharevsky_point_min = "0; 0"
bocharevsky_point_start = "-2; 2"

perm_point_min = "0; 0.5"
perm_point_start = "100; 741.9474"

diff_power_point_min = "0; 0"
diff_power_point_start = "-2; 2"

trid_point_min = "2; 6"
trid_point_start = "1; 7"

booth_point_min = "1; 3"
booth_point_start = "-1; 5"

matyas_point_min = "0; 0"
matyas_point_start = "-2; 2"

mccormick_point_min = "-0.54719; -1.54719"
mccormick_point_start = "-2.09438; 0"

# powersum_point_min = "None; None"
# powersum_point_start = "0; 0"

zakharov_point_min = "0; 0"
zakharov_point_start = "-2; 2"

three_hump_point_min = "0; 0"
three_hump_point_start = "-2; 2"

six_hump_point_min = "0.0898; -0.7126"
six_hump_point_start = "-0.6228; 0"

dixon_price_point_min = "1; 2 ** (-0.5)"
dixon_price_point_start = "0; 1.7071"

rosenbrock_point_min = "1; 1"
rosenbrock_point_start = "-2; 4"

easom_point_min = "pi; pi"
easom_point_start = "0; 6.28"

michalewicz_point_min = "2.20; 1.57"
michalewicz_point_start = "-0.33; 4"

funcs_str = {
    'Ackley function': [ackley, {'eq': ackley_restr_eq, 'ineq': ackley_restr_ineq, 'eq&ineq': ackley_restr_eq_ineq},
                        ackley_point_min, ackley_point_start],
    'Bukin function №6': [bukin, {'eq': bukin_restr_eq, 'ineq': bukin_restr_ineq, 'eq&ineq': bukin_restr_eq_ineq},
                          bukin_point_min, bukin_point_start],
    'Cross-in-tray function': [cross_in_tray,
                               {'eq': cross_in_tray_restr_eq, 'ineq': cross_in_tray_restr_ineq, 'eq&ineq': cross_in_tray_restr_eq_ineq},
                               cross_in_tray_point_min, cross_in_tray_point_start],
    'Drop-wave function': [drop_wave,
                           {'eq': drop_wave_restr_eq, 'ineq': drop_wave_restr_ineq, 'eq&ineq': drop_wave_restr_eq_ineq},
                           drop_wave_point_min, drop_wave_point_start],
    'Eggholder function': [eggholder,
                           {'eq': eggholder_restr_eq, 'ineq': eggholder_restr_ineq, 'eq&ineq': eggholder_restr_eq_ineq},
                           eggholder_point_min, eggholder_point_start],
    'Griewank function': [griewank, {'eq': griewank_restr_eq, 'ineq': griewank_restr_ineq, 'eq&ineq': griewank_restr_eq_ineq},
                          griewank_point_min, griewank_point_start],
    # 'Holder table function': [holder_table, holder_table_point_min, holder_table_point_start],
    'Levy function': [levy, {'eq': levy_restr_eq, 'ineq': levy_restr_ineq, 'eq&ineq': levy_restr_eq_ineq},
                      levy_point_min, levy_point_start],
    'Levy №13 function': [levy13, {'eq': levy13_restr_eq, 'ineq': levy13_restr_ineq, 'eq&ineq': levy13_restr_eq_ineq},
                          levy13_point_min, levy13_point_start],
    'Rastrigin function': [rastrigin,
                           {'eq': rastrigin_restr_eq, 'ineq': rastrigin_restr_ineq, 'eq&ineq': rastrigin_restr_eq_ineq},
                           rastrigin_point_min, rastrigin_point_start],
    'Schaffer function': [schaffer, {'eq': schaffer_restr_eq, 'ineq': schaffer_restr_ineq, 'eq&ineq': schaffer_restr_eq_ineq},
                          schaffer_point_min, schaffer_point_start],
    'Schwefel function': [schwefel, {'eq': schwefel_restr_eq, 'ineq': schwefel_restr_ineq, 'eq&ineq': schwefel_restr_eq_ineq},
                          schwefel_point_min, schwefel_point_start],
    'Bocharevsky function': [bocharevsky,
                             {'eq': bocharevsky_restr_eq, 'ineq': bocharevsky_restr_ineq, 'eq&ineq': bocharevsky_restr_eq_ineq},
                             bocharevsky_point_min, bocharevsky_point_start],
    'Perm 0,d,beta function': [perm,
                               {'eq': perm_restr_eq, 'ineq': perm_restr_ineq, 'eq&ineq': perm_restr_eq_ineq},
                               perm_point_min, perm_point_start],
    'Sum of different powers function': [diff_power, {'eq': diff_power_restr_eq, 'ineq': diff_power_restr_ineq,
                                                      'eq&ineq': diff_power_restr_eq_ineq}, diff_power_point_min,
                                         diff_power_point_start],
    'Trid function': [trid, {'eq': trid_restr_eq, 'ineq': trid_restr_ineq, 'eq&ineq': trid_restr_eq_ineq},
                      trid_point_min, trid_point_start],
    'Booth function': [booth, {'eq': booth_restr_eq, 'ineq': booth_restr_ineq, 'eq&ineq': booth_restr_eq_ineq},
                       booth_point_min, booth_point_start],
    'Matyas function': [matyas, {'eq': matyas_restr_eq, 'ineq': matyas_restr_ineq, 'eq&ineq': matyas_restr_eq_ineq},
                        matyas_point_min, matyas_point_start],
    'Mccormick function': [mccormick,
                           {'eq': mccormick_restr_eq, 'ineq': mccormick_restr_ineq, 'eq&ineq': mccormick_restr_eq_ineq},
                           mccormick_point_min, mccormick_point_start],
    # 'Power sum function': [powersum, powersum_point_min, powersum_point_start],
    'Zakharov function': [zakharov, {'eq': zakharov_restr_eq, 'ineq': zakharov_restr_ineq, 'eq&ineq': zakharov_restr_eq_ineq},
                          zakharov_point_min, zakharov_point_start],
    'Three-hump function': [three_hump,
                            {'eq': three_hump_restr_eq, 'ineq': three_hump_restr_ineq, 'eq&ineq': three_hump_restr_eq_ineq},
                            three_hump_point_min, three_hump_point_start],
    'Six-hump function': [six_hump, {'eq': six_hump_restr_eq, 'ineq': six_hump_restr_ineq, 'eq&ineq': six_hump_restr_eq_ineq},
                          six_hump_point_min, six_hump_point_start],
    'Dixon-price function': [dixon_price,
                             {'eq': dixon_price_restr_eq, 'ineq': dixon_price_restr_ineq, 'eq&ineq': dixon_price_restr_eq_ineq},
                             dixon_price_point_min, dixon_price_point_start],
    'Rosenbrock function': [rosenbrock,
                            {'eq': rosenbrock_restr_eq, 'ineq': rosenbrock_restr_ineq, 'eq&ineq': rosenbrock_restr_eq_ineq},
                            rosenbrock_point_min, rosenbrock_point_start],
    'Easom function': [easom, {'eq': easom_restr_eq, 'ineq': easom_restr_ineq, 'eq&ineq': easom_restr_eq_ineq},
                       easom_point_min, easom_point_start],
    'Michalewicz function': [michalewicz,
                             {'eq': michalewicz_restr_eq, 'ineq': michalewicz_restr_ineq, 'eq&ineq': michalewicz_restr_eq_ineq},
                             michalewicz_point_min, michalewicz_point_start]}
