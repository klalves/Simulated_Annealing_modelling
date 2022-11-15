import csv
import argparse
from math import sqrt, exp
import random
import matplotlib.pyplot as plt
import Simulated_Annealing

delay_range  = [0, 30]
tau1_range   = [0.1, 200]
tau2_range   = [0.1, 200]
dynamic_skew_range = [0, 1]
tbase_range  = [-30, -10]
gain_range   = [-0.005, 0.000]
temp_1_range   = [-1, 1]
temp_2_range   = [-1, 1]
parameters_range = [delay_range,
                    tau1_range,
                    tau2_range,
                    dynamic_skew_range,
                    tbase_range,
                    gain_range,
                    temp_1_range,
                    temp_2_range]

reference_data = []

class data_point:
    def __init__(self, time, tsens, tevap, speed):
        self.time = time
        self.tsens = tsens
        self.tevap = tevap
        self.speed = speed

class system_model:
    def __init__(self, delay, tau1, tau2, dynamic_skew, tbase, gain, temp_1, temp_2):
        self.delay = delay
        self.tau1 = tau1
        self.tau2 = tau2
        self.dynamic_skew = dynamic_skew
        self.tbase = tbase
        self.gain = gain
        self.temp_1 = temp_1
        self.temp_2 = temp_2

    def parameter_count_get(self):
        return 8

    def parameter_list_get(self):
        param_list = []
        param_list.append(self.delay)
        param_list.append(self.tau1)
        param_list.append(self.tau2)
        param_list.append(self.dynamic_skew)
        param_list.append(self.tbase)
        param_list.append(self.gain)
        param_list.append(self.temp_1)
        param_list.append(self.temp_2)

        return param_list


def t_sens_target_from_speed(model, speed):
    return model.tbase + model.gain*speed

def index_delta_get(time_delta):
    index_delta = 0

    for point in reference_data:
        if point.time - reference_data[0].time >= time_delta:
            return index_delta
        index_delta += 1

    return index_delta

def sampling_time_get():
    time_start = reference_data[0].time
    time_end = reference_data[len(reference_data)-1].time
    return (time_end - time_start)/(len(reference_data)-1)

def simulate_system_model(model, debug=0):
    output_data = []

    index_t_sens_delay = index_delta_get(model.delay)

    s_pole_1 = -1/model.tau1
    s_pole_2 = -1/model.tau2
    s_zero_1 = -1/(model.dynamic_skew*(model.tau2 - model.tau1) + model.tau1)

    if(debug):
        print("s_pole_1: " + str(s_pole_1))
        print("s_pole_2: " + str(s_pole_2))
        print("s_zero_1: " + str(s_zero_1))

    Ts = sampling_time_get()

    if(debug):
        print("Ts: " + str(Ts))

    z_pole_1 = exp(Ts*s_pole_1)
    z_pole_2 = exp(Ts*s_pole_2)
    z_zero_1 = exp(Ts*s_zero_1)

    if(debug):
        print("z_pole_1: " + str(z_pole_1))
        print("z_pole_2: " + str(z_pole_2))
        print("z_zero_1: " + str(z_zero_1))

    coeff_x_k_1 = 1
    coeff_x_k_2 = -z_zero_1
    coeff_y_k_1 = z_pole_1+z_pole_2
    coeff_y_k_2 = -z_pole_1*z_pole_2
    coeff_gain = -(coeff_y_k_1 + coeff_y_k_2 - 1)/(coeff_x_k_1+coeff_x_k_2)

    if(debug):
        print("coeff_x_k_1: " + str(coeff_x_k_1))
        print("coeff_x_k_2: " + str(coeff_x_k_2))
        print("coeff_y_k_1: " + str(coeff_y_k_1))
        print("coeff_y_k_2: " + str(coeff_y_k_2))
        print("coeff_gain: " + str(coeff_gain))

    #Initialize simulated temperatures with same values as reference_data
    tsens_simulated = reference_data[0].tsens
    tevap_simulated = reference_data[0].tevap

    tsens_simulated_1 = reference_data[0].tsens + model.temp_1
    tsens_simulated_2 = reference_data[0].tsens + model.temp_2

    tsens_speed_1 = reference_data[0].speed
    tsens_speed_2 = reference_data[0].speed

    for data_index in range(len(reference_data)):

        if data_index > index_t_sens_delay:
            index_t_sens = data_index - index_t_sens_delay
        else:
            index_t_sens = 0

        tsens_speed = reference_data[index_t_sens].speed
        tsens_target_1 = t_sens_target_from_speed(model, tsens_speed_1)
        tsens_target_2 = t_sens_target_from_speed(model, tsens_speed_2)
        tsens_speed_2 = tsens_speed_1
        tsens_speed_1 = tsens_speed

        tsens_simulated = coeff_gain*(coeff_x_k_1*tsens_target_1 + coeff_x_k_2*tsens_target_2) + coeff_y_k_1*tsens_simulated_1 + coeff_y_k_2*tsens_simulated_2
        tsens_simulated_2 = tsens_simulated_1
        tsens_simulated_1 = tsens_simulated

        datum = data_point(reference_data[data_index].time,
                           tsens_simulated,
                           tevap_simulated,
                           reference_data[data_index].speed)
        
        output_data.append(datum)
    
    return output_data

def plot_system_model_simulation(model, axis):
    simulate_data = simulate_system_model(model)

    t = []
    tsens_sim = []
    tsens_ref = []
    for idx in range(len(simulate_data)):
        t.append(simulate_data[idx].time)
        tsens_sim.append(simulate_data[idx].tsens)
        tsens_ref.append(reference_data[idx].tsens)

    axis.plot(t, tsens_ref, 'r') 
    axis.plot(t, tsens_sim, 'b')

def fitness_function(model):
    
    fitness = 0

    simulated_data = simulate_system_model(model)

    for simulated_point, reference_point in zip(simulated_data, reference_data):
        error = simulated_point.tsens - reference_point.tsens
        fitness += error*error
    
    return sqrt(fitness)

def neighbour_generator_function(model=None, max_relative_step_size=0.01):
    tmp_param = []

    for idx in range(len(parameters_range)):
        if(model == None):
            value = random.uniform(parameters_range[idx][0], parameters_range[idx][1])
        else:
            model_parameters = model.parameter_list_get()
            step_max = max_relative_step_size*abs(parameters_range[idx][1] - parameters_range[idx][0])
            gain = random.uniform(-1,1)
            value = model_parameters[idx] + gain*step_max
            if(value < parameters_range[idx][0]):
                value = parameters_range[idx][0]
            elif(value > parameters_range[idx][1]):
                value = parameters_range[idx][1]
        
        tmp_param.append(value)

    neighbour = system_model(tmp_param[0],
                             tmp_param[1],
                             tmp_param[2],
                             tmp_param[3],
                             tmp_param[4],
                             tmp_param[5],
                             tmp_param[6],
                             tmp_param[7])

    return neighbour


def plot_parameter_log(log_sample, log_best):
    
    parameter_count = log_sample[0].parameter_count_get()

    fig = plt.figure()
    for param_idx in range(parameter_count):
        t = []
        param_sample = []
        param_best = []
        for log_idx in range(len(log_sample)):
            t.append(log_idx)
            sample_param_list = log_sample[log_idx].parameter_list_get()
            best_param_list   = log_best[log_idx].parameter_list_get()
            param_sample.append(sample_param_list[param_idx])
            param_best.append(best_param_list[param_idx])
            
        ax = fig.add_subplot(parameter_count+1,1,param_idx+1)
        ax.plot(t, param_sample, 'r')
        ax.plot(t, param_best, 'b')
    plt.show()


def main():
    parser = argparse.ArgumentParser(
                    prog = 'System_Model_Calculator',
                    description = 'Calculate a refrigeration system model based on functional data',
                    epilog = 'Text at the bottom of help')
    parser.add_argument('-i', '--input_file', default = 'system_data.csv' )
    parser.add_argument('-o', '--output_file', default = 'system_model.csv')
    parser.add_argument('-d', '--debug', action = 'store_true')
    args = parser.parse_args()

    with open(args.input_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            datum = data_point(float(row['Time']),
                               float(row['Tsens']),
                               float(row['Tevap']),
                               float(row['Speed']))

            reference_data.append(datum)

        print("len(reference_data): " + str(len(reference_data)))

    print("reference_data loaded")
    
    if args.debug:
        #system_model(delay, tau1, tau2, dynamic_skew, tbase, gain)
        delay = 3
        tau1 = 10
        tau2 = 1
        dynamic_skew = 0.5
        tbase = -19
        gain = -6/3000
        temp_1 = -23
        temp_2 = -23
        model = system_model(delay, tau1, tau2, dynamic_skew, tbase, gain, temp_1, temp_2)
        simulate_data = simulate_system_model(model)

        t = []
        tsens_sim = []
        tsens_ref = []
        for idx in range(len(simulate_data)):
            t.append(simulate_data[idx].time)
            tsens_sim.append(simulate_data[idx].tsens)
            tsens_ref.append(reference_data[idx].tsens)

        #plt.plot(t, tsens_ref, 'r') 
        #plt.plot(t, tsens_sim, 'b')
        #plt.show()

    sa_process = Simulated_Annealing.Simulated_Annealing(fitness_function,
                                                         neighbour_generator_function)

    for idx in range(40000):
        sa_process.iterate()

        if(idx==5000):
            sa_process.temperature = 20
        
        if(idx==10000):
            sa_process.temperature = 10

        if(idx==15000):
            sa_process.temperature = 5

    if args.debug:
        best_sample_parameters = sa_process.best_sample.parameter_list_get()
        print("best_sample_parameters: " + str(best_sample_parameters))
        print("best_fitness: " + str(sa_process.best_fitness))
        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1)
        ax2 = fig.add_subplot(2,2,3)
        ax3 = fig.add_subplot(2,2,(2,4))
        ax1.set_yscale('log')
        ax1.plot(sa_process.log_iteration_count, sa_process.log_fitness, 'r')
        ax1.plot(sa_process.log_iteration_count, sa_process.log_best_fitness, 'b')
        ax2.plot(sa_process.log_iteration_count, sa_process.log_temperature, 'r')
        plot_system_model_simulation(sa_process.best_sample ,ax3)
        plt.show()

        plot_parameter_log(sa_process.log_sample, sa_process.log_best_sample)

if __name__ == "__main__":
    main()