import random
import numpy as np
import pandas as pd

# Read the baseline traffic data
baseline_df = pd.read_csv("Data/Traffic_data_orig.csv")
baseline_df = baseline_df.sort_values('Time')
times = baseline_df['Time'].tolist()

ipd_list = []
for i in range(1, len(times)):
    delay = times[i] - times[i - 1]
    ipd_list.append(delay)

min_val = min(ipd_list)
median_val = pd.Series(ipd_list).median()
max_val = max(ipd_list)

def simulateBuffer(distribution_type, m, initial_buffer, B=20, N=500, sigma=0.05):

    # Read secret message bits from file, if possible, else generate random bits.
    try:
        with open("secret_message_bits", "r") as f:
            s = "".join(line.strip() for line in f)
        if len(s) != m:
            raise ValueError("Secret bit length mismatch.")
    except:
        s = "".join(random.choice("01") for _ in range(m))
    
    count_underflow = 0
    count_overflow = 0
    count_success = 0

    for _ in range(N):

        #If bit = '0':  ipd_k ~ Uniform(min_val, median_val)
        #If  bit  = '1':  ipd_k ~ Uniform(median_val, max_val)
        ipd_list = []
        for bit in s:
            if bit == '0':
                ipd = random.uniform(min_val, median_val)
            else:
                ipd = random.uniform(median_val, max_val)
            ipd_list.append(ipd)
        ipd = np.array(ipd_list)
        
        # Add noise: ipd_noised
        ipd_noised = ipd + np.random.normal(0, sigma, ipd.shape)
        
        if distribution_type == "exponential":
            X = np.random.exponential(scale=1, size=1000)
            X = np.minimum(X, 5) 
        elif distribution_type == "uniform":
            X = np.random.uniform(0, 1, size=1000)
        else:
            raise ValueError("Invalid distribution type")
        a = np.cumsum(X)  # Arrival times
        
        CB = initial_buffer  
        T_current = 0.0      #T[0] = 0
        previous_arrival_count = 0    # Index to track arrivals times
        trial_failed = False
        
        for ipd_k in ipd_noised:
            T_next = T_current + ipd_k 
            
            current_arrival_count = np.searchsorted(a, T_next, side='right')
            arrivals_during_transmission = current_arrival_count - previous_arrival_count  #number of arrivals during this interval
            previous_arrival_count = current_arrival_count
            
            #Previous buffer + arrivals_during_transmission - 1 transmitted packet = CB(k-1) + arrivals_during_transmission - 1. 
            #Arrivals =                    
            CB = CB + arrivals_during_transmission - 1
            
            #buffer underflowed 
            if CB < 0:
                count_underflow += 1
                trial_failed = True
                break
            if CB > B: #overflowed
                count_overflow += 1
                trial_failed = True
                break
                
            T_current = T_next
        
        if not trial_failed:
            count_success += 1

    return count_underflow / N, count_overflow / N, count_success / N

if __name__ == "__main__":
    
    distribution_type = input("Enter IPD distribution (exponential or uniform): ").strip().lower()
    m = int(input("Enter size of the secret message (m) ( 16, 32): "))
    initial_buffer = int(input("Enter the initial buffer value (i): "))

    pu, po, ps = simulateBuffer(distribution_type, m, initial_buffer)
    
    print("\nSimulation Results:")
    print("Underflow Probability: {:.3f}".format(pu))
    print("Overflow Probability:  {:.3f}".format(po))
    print("Success Probability:   {:.3f}".format(ps))
