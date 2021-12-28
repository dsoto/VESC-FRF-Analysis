import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def getFFT(signal, dt):
    Signal = np.fft.fft(signal, axis = 0)
    f = np.fft.fftfreq(len(signal), dt)
    Signal = Signal[f > 0]
    Signal = 2 * Signal
    f = f[f > 0]
    return Signal, f

def read_response_dist(data_filename, voltage_disturbance):
    df = pd.read_csv(data_filename, sep=';')
    # limit to 2^11 - 1 samples
    df = df[:2047]
    response = df.FRF_d.values / 1e3
    dist = (df['PRBS'] * 2 - 1).values * voltage_disturbance
    dt = df['T'][1] # infer time step from first non-zero timestamp
    return response, dist, dt

def plot_mag_ax(f, mag, ax, label, alpha=1.0):
    ax.semilogx(f, 20 * np.log10(mag), label=label, alpha=alpha)
    ax.grid(1, 'both', 'both')
    ax.set_ylabel('Magnitude [dB]')
    ax.set_xlabel('Frequency [Hz]')
    ax.legend()

def plot_phase_ax(f, phase, ax, label):
    ax.semilogx(f, phase * 180 / np.pi, label=label)
    ax.grid( 1 , 'both')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Phase [deg]')
    ax.set_ylim( -182, 182)
    ax.legend()
    #plt.show()

def read_current_data(data_filename):
    df = pd.read_csv(data_filename, sep=';')
    df = df[:2047]
    dt = df['T'][1]
    return df.I1.values, df.I2.values, df.I3.values, df.Phase.values / 360 * 2 * np.pi, dt

# get list of data files from data_directory
def get_file_list(data_directory):
    import glob
    data_filenames = glob.glob(data_directory + '/*.csv')
    data_filenames.sort()
    return data_filenames

def data_report(data_filenames):
    for i, data_filename in enumerate(data_filenames):
        print(data_filename, end=' | ')
        df = pd.read_csv(data_filename, sep=';')
        # trim file to 2^11 - 1 samples
        df = df[:2047]
        response = df.FRF_d.values / 1e3
        dist = (df['PRBS'] * 2 - 1).values
        print(f"{1/df['T'].diff().max():.2e} Hz", end=' | ')
        print(f"len {len(df)}", end=' | ')
        if i > 0:
            trace_period = int(data_filenames[i][-17:-4]) - int(data_filenames[i-1][-17:-4])
            print(f"gap {trace_period}")
        else:
            print(f"gap NA")
        if np.abs(dist.mean()) > 0.0005:
            print('PRBS non-zero mean error', dist.mean())
        if len(df) < 2047:
            print('truncated file error', len(df))
        if dist.std() < 0.99:
            print('low std dev for disturbance', dist.std())

def plot_all_data(data_filename, title):
    df = pd.read_csv(data_filename, sep=';')
    # limit to 2^11 - 1 samples
    df = df[:2047]
    # check for truncated files
    if len(df) < 2047:
        print('truncated file', len(df))
    response = df.FRF_d.values / 1e3
    dist = (df['PRBS'] * 2 - 1).values
    dt = df['T'][1] # infer time step from first non-zero timestamp
    # check for PRBS with large DC component
    if np.abs(dist.mean()) > 0.0005:
        print('PRBS non-zero mean error', dist.mean(), data_filename)
    fig, ax = plt.subplots(4, 1)
    fig.suptitle(title)
    ax[0].plot(dist)
    ax[0].set_ylabel('Dist')
    ax[1].plot(response)
    ax[1].set_ylabel('Response')
    ax[2].plot(df.I1)
    ax[2].plot(df.I2)
    ax[2].plot(df.I3)
    ax[2].set_ylabel('Currents')
    ax[3].plot(df.V1)
    ax[3].plot(df.V2)
    ax[3].plot(df.V3)
    ax[3].set_ylabel('Voltages')
    plt.show()

def plot_plant(data_directory, R=100E-3, L=100E-6,
               plot_mag=True, plot_phase=True,
               f_min=1E0, f_max=25E3, plot_show=True, ax=None,
               plot_sim=True, title=None, mag_label=None, alpha=1.0):
    data_filenames = get_file_list(data_directory)
    num_files = len(data_filenames)
    avg_ID_by_R = np.zeros(1023, dtype='complex')
    avg_IQ_by_R = np.zeros(1023, dtype='complex')

    for data_file in data_filenames:

        I1, I2, I3, phase, dt = read_current_data(data_file)

        I_alpha = (I1 - (I2 + I3) * np.sin(np.pi / 6)) * np.sqrt(2/3)
        I_beta = ((I2 - I3) * np.cos(np.pi / 6)) * np.sqrt(2/3)

        Id = I_alpha * np.cos(phase) + I_beta * np.sin(phase)
        Iq = I_beta * np.cos(phase) - I_alpha * np.sin(phase)

        ID, f = getFFT(Id, dt)
        IQ, f = getFFT(Iq, dt)

        response, dist, dt = read_response_dist(data_file, 1)
        RESPONSE, f = getFFT(response, dt)

        avg_IQ_by_R += IQ / RESPONSE / num_files

    f_sim = np.logspace(np.log(f_min), np.log(f_max), 100)
    omega = f_sim * 2 * np.pi
    simulated_plant = 1 / (R + 1j * omega * L)

    if plot_mag:
        if plot_show:
            fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlim([1, f_max])
        ax.set_ylim([-50, 30])
        plot_mag_ax(f, np.abs(avg_IQ_by_R), ax, label=mag_label, alpha=alpha)
        if plot_sim:
            plot_mag_ax(f_sim, np.abs(simulated_plant), ax, label='simulation')
        if plot_show == True:
            plt.show()

    if plot_phase:
        if plot_show:
            fig, ax = plt.subplots()
        ax.set_title('Phase')
        plot_phase_ax(f, np.angle(avg_IQ_by_R), ax, label='measured')
        if plot_sim:
            plot_phase_ax(f_sim, np.angle(simulated_plant), ax, label='simulation')
        if plot_show == True:
            plt.show()