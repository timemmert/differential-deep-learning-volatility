import os, sys

project_dir = f"{os.path.dirname(os.path.abspath(__file__))}/.."
print(os.listdir(project_dir))
sys.path.append(project_dir)

from src import Pricer, UniformSampler, PolynomialAndTrigonometricModel

def main():
    model = PolynomialAndTrigonometricModel()
    sampler = UniformSampler([-1, -2, -3, -4], [1, 2, 3, 4])
    pricer = Pricer(model=model, parameters_sampler=sampler, option=None)

    data = pricer.generate_training_data(5, 5)
    print(data[0])


if __name__ == "__main__":
    main()