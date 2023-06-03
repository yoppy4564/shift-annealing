import neal
from pyqubo import Array, Constraint

N = 8
limit_weight = 3000
num_reads = 10

def check(result):
    for r in result.data(['sample']):
        spin = r.sample
        if spin["x[5]"] == 1 and spin["x[6]"] == 1:
            print(spin)
            return True
    return False

def create_model():
    x = Array.create('x', shape=N, vartype='BINARY')
    H1 = - (12000*x[0] + 2500*x[1] + 100*x[2] + 200*x[3] + 3000*x[4] + 5000*x[5] + 8500*x[6] + 700*x[7])
    H2 = Constraint((limit_weight - (2500*x[0] + 800*x[1] + 30*x[2] + 50*x[3] + 600*x[4] + 1300*x[5] + 1700*x[6] + 250*x[7]))**2, "const")
    H = H1 + H2

    model = H.compile()
    qubo, offset = model.to_qubo(feed_dict={"const": 1.0})
    return qubo, offset

def exe(qubo):
    sampler = neal.SimulatedAnnealingSampler()
    result = sampler.sample_qubo(qubo, num_reads=num_reads)
    return result

if __name__ == "__main__":
    qubo, offset = create_model()
    result = exe(qubo)
    if check(result):
        print("Success")
    else:
        print("Failed")

