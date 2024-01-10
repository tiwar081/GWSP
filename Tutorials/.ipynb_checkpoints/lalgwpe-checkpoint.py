from os import fsync
import numpy as np
import scipy as sp
import lal
import lalsimulation as lalsim
import gwpy
from gwpy.timeseries import TimeSeries

#Generate and plot time domain waveforms of different models (perhaps IMRPhenomD and TaylorF2) using same parameters.
#You'll first notice that they don't quite line up, but if you use the match_phase functions, you should be able to get them to line up better. 

class data():
    """
    A class that gets information from strain data from LIGO/Virgo data releases.
    """
    def __init__(self, T, Fs, T_long, tgps, tukey_alpha = 0.05, detstrings = ['H1', 'L1', 'V1']):
        """
        Initializes the data object, which contains a time series, a frequency series, and a psd for each of the LIGO/Virgo detectors specified in detstrings.
        T: length of time series (s)
        Fs: sampling rate (Hz)
        T_long: length of longer time series used to compute the psd using Welch's method (s)
        tgps: the approximate gps time of the event
        tukey_alpha: the alpha parameter in the tukey window used in the fourier transform to get the frequency series
        detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
        """
        self.ndet = len(detstrings)
        self.detstrings = detstrings
        
        self.strain_td = np.zeros([self.ndet, T*Fs])
        self.strain_fd = np.zeros([self.ndet, T*Fs//2+1], dtype = complex)
        self.psd = np.zeros([self.ndet, T*Fs//2+1])
        self.ts = np.linspace(-T/2, T/2-1/Fs, T*Fs)
        
        for detstring, i in zip(detstrings, range(self.ndet)):
            data_long = TimeSeries.fetch_open_data(detstring, tgps-T_long/2, tgps+T_long/2).resample(Fs)
            data_short = TimeSeries.fetch_open_data(detstring, tgps-T/2, tgps+T/2).resample(Fs)

            strain_td_long = np.array(data_long)
            self.strain_td[i] = np.array(data_short)
            # a tukey window is applied to the data before the fft to remove edge effects
            self.strain_fd[i] = 1/Fs*np.fft.rfft(self.strain_td[i]*sp.signal.tukey(len(self.strain_td[i]), alpha=tukey_alpha))

            self.fs, self.psd[i] = sp.signal.welch(strain_td_long, fs=Fs, nperseg = Fs*T)
        self.T = T
        self.Fs = Fs
        self.tgps = tgps

def hpc(params, f_seq, modelstring): 
    """
    Computes the plus and cross polarizations for the dict params and model, at the frequencies in f_seq by calling lal for the polarizations directly.
    params: parameter dictionary
    f_seq: REAL8Sequence LAL object that contains a frequency array
    modelstring: string specifying which model to use, some possible values include "TaylorF2", IMRPhenomD", "IMRPhenomXPHM"
    """
    pdict = {'DL': params['distance'],
                 'phiref': params['phi_ref'],
                 'f_ref': params['f_ref'],
                 'inclination': params['inclination'],
                 'm1': params['mass_1'] * lal.MSUN_SI, 'm2': params['mass_2'] * lal.MSUN_SI,
                 's1x': params['chi_1x'], 's1y': params['chi_1y'], 's1z': params['chi_1z'],
                 's2x': params['chi_2x'], 's2y': params['chi_2y'], 's2z': params['chi_2z']}

    PARAMNAMES = ['phiref', 'm1', 'm2', 's1x', 's1y', 's1z',
                          's2x', 's2y', 's2z', 'f_ref', 'DL', 'inclination']

    # Tidal deformabilities are zero for black holes, 
    lal_pars = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertTidalLambda1(lal_pars, params['Lambda1'])
    lalsim.SimInspiralWaveformParamsInsertTidalLambda2(lal_pars, params['Lambda2'])

    # Parameters in the order that LAL takes, give approximant for the given model
    wfparams = [pdict[p] for p in PARAMNAMES] \
        + [lal_pars, lalsim.GetApproximantFromString(modelstring), f_seq]

    # Generate hplus, hcross
    hplus, hcross = lalsim.SimInspiralChooseFDWaveformSequence(
        *wfparams)

    return hplus.data.data, hcross.data.data

def compute_response_coeffs(params, detstrings = ['H1', 'L1', 'V1']):
    """
    Computes the detector response coefficients Fplus and Fcross for the parameters in the dict params.
    params: parameter dictionary
    detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
    """
    ra = params['ra']
    dec = params['dec']
    psi = params['psi']
    tgps = params['tgps']
    
    ndet = len(detstrings)
    
    # get detector response matrices from lal
    DET_CODE = {'H1': lal.LHO_4K_DETECTOR,
                'L1': lal.LLO_4K_DETECTOR,
                'V1': lal.VIRGO_DETECTOR}
    det_response = [lal.CachedDetectors[DET_CODE[det_name]].response
                                 for det_name in detstrings] #detector orientation 

    # get Greenwich Mean Sidereal Time (time relative to stars) from the gps time
    gmst = lal.GreenwichMeanSiderealTime(tgps)
    # get azimuthal orientation from Greenwich Mean Sidereal Time and right inclination (azimuthal angle relative to Earth)
    gha = gmst - ra

    X = np.array([-np.cos(psi)*np.sin(gha)-np.sin(psi)*np.cos(gha)*np.sin(dec),
                 -np.cos(psi)*np.cos(gha)+np.sin(psi)*np.sin(gha)*np.sin(dec),
                 np.sin(psi)*np.cos(dec)])
    Y = np.array([np.sin(psi)*np.sin(gha)-np.cos(psi)*np.cos(gha)*np.sin(dec),
                 np.sin(psi)*np.cos(gha)+np.cos(psi)*np.sin(gha)*np.sin(dec),
                 np.cos(psi)*np.cos(dec)])

    Fplus = np.zeros(ndet)
    Fcross = np.zeros(ndet)
    for i_det in range(ndet):
        for i in range(3):
            for j in range(3):
                Fplus[i_det] += X[j]*det_response[i_det][j][i]*X[i] - Y[j]*det_response[i_det][j][i]*Y[i]
                Fcross[i_det] += X[j]*det_response[i_det][j][i]*Y[i] + Y[j]*det_response[i_det][j][i]*X[i]
                
    return Fplus, Fcross

def compute_response_coeffs_time_dep(params, T, Fs, detstrings = ['H1', 'L1', 'V1']):
    """
    Computes the detector response coefficients Fplus and Fcross for the parameters in the dict params, as functions of time.
    params: parameter dictionary
    T: the length of the corresponding time series (s)
    Fs: the sampling rate of the corresponding time series (Hz)
    detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
    """
    ra = params['ra']
    dec = params['dec']
    psi = params['psi']
    tc = params['tc']
    tcoarse = params['tcoarse']
    tgps_0 = params['tgps']
    tgps_min = tgps_0 - tc - tcoarse
    tgps = np.linspace(tgps_min, tgps_min + T - 1/Fs, T*Fs) #GPS time at merger, i.e. at peak strain value (see Jupyter notebook). How to define this?
    ndet = len(detstrings)
    
    # get detector response matrices from lal
    DET_CODE = {'H1': lal.LHO_4K_DETECTOR,
                'L1': lal.LLO_4K_DETECTOR,
                'V1': lal.VIRGO_DETECTOR}
    det_response = [lal.CachedDetectors[DET_CODE[det_name]].response
                                 for det_name in detstrings] #detector orientation 

    # get Greenwich Mean Sidereal Time (time relative to stars) from the gps time
    gmst = np.array([lal.GreenwichMeanSiderealTime(t) for t in tgps]) #calculate current GMST angle, given tgps
    # get azimuthal orientation from Greenwich Mean Sidereal Time and right inclination (azimuthal angle relative to Earth)
    gha = gmst - ra

    X = np.array([-np.cos(psi)*np.sin(gha)-np.sin(psi)*np.cos(gha)*np.sin(dec),
                 -np.cos(psi)*np.cos(gha)+np.sin(psi)*np.sin(gha)*np.sin(dec),
                 np.full_like(gha, np.sin(psi)*np.cos(dec))])
    Y = np.array([np.sin(psi)*np.sin(gha)-np.cos(psi)*np.cos(gha)*np.sin(dec),
                 np.sin(psi)*np.cos(gha)+np.cos(psi)*np.sin(gha)*np.sin(dec),
                 np.full_like(gha, np.cos(psi)*np.cos(dec))])

    Fplus = np.zeros((ndet, len(gha)))
    Fcross = np.zeros((ndet, len(gha)))
    for i_det in range(ndet):
        for i in range(3):
            for j in range(3):
                Fplus[i_det] += X[j]*det_response[i_det][j][i]*X[i] - Y[j]*det_response[i_det][j][i]*Y[i]
                Fcross[i_det] += X[j]*det_response[i_det][j][i]*Y[i] + Y[j]*det_response[i_det][j][i]*X[i]
                
    return Fplus, Fcross

def compute_time_delay(params, T, Fs, detstrings = ['H1', 'L1', 'V1']):
    """
    Computes the vectorized time delay for each of the detectors in detstrings for the parameters in the dict params.
    params: parameter dictionary
    T: the length of the corresponding time series (s)
    Fs: the sampling rate of the corresponding time series (Hz)
    detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
    """       
    ndet = len(detstrings)
    
    ra = params['ra']
    dec = params['dec']
    tc = params['tc']
    tcoarse = params['tcoarse']
    tgps_0 = params['tgps']
    tgps_min = tgps_0 - tc - tcoarse
    tgps = np.linspace(tgps_min, tgps_min + T - 1/Fs, T*Fs) #GPS time at merger, i.e. at peak strain value (see Jupyter notebook). How to define this?

    DET_CODE = {'H1': lal.LHO_4K_DETECTOR,
                'L1': lal.LLO_4K_DETECTOR,
                'V1': lal.VIRGO_DETECTOR} #dictionary assigning numeric codes to each detector. inputted into lal.CachedDetectors
    
    ndet = len(detstrings) #no. of detectors

    det_location = [lal.CachedDetectors[DET_CODE[det_name]].location #get location of each detector you're working with in (x, y, z) in meters
                                 for det_name in detstrings] #xyz system rotates with earth here - so this is time-independent

    gmst = np.array([lal.GreenwichMeanSiderealTime(t) for t in tgps]) #calculate current GMST angle, given tgps
    gha = gmst - ra
    esrc = np.array([np.cos(dec)*np.cos(gha), -np.cos(dec)*np.sin(gha), np.full_like(gha, np.sin(dec))]) #vector from earth to source (minus k in notes) in (x, y, z), same coords as yours in toy problem and det_location

    time_delay = np.array([[ -np.dot(esrc[:,t], det_location[i_det])/ lal.C_SI for t in range(T*Fs)] for i_det in range(ndet)]) #calculate time_delay_offset for each detector you're working with (see notes)

    return time_delay

def compute_strain_fd(params, T, Fs, modelstring, detstrings = ['H1', 'L1', 'V1']):
    """
    Computes the frequency domain strain in the detectors in detstrings for the parameters in the dict params.
    params: parameter dictionary
    T: the length of the corresponding time series (s)
    Fs: the sampling rate of the corresponding time series (Hz)
    modelstring: string specifying which model to use, some possible values include "TaylorF2", IMRPhenomD", "IMRPhenomXPHM"
    detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
    """       
    f_min = params['f_min']
    f_max = params['f_max']
    ndet = len(detstrings)
    fs = np.linspace(f_min, f_max, int((f_max-f_min)*T)+1) #array going from fmin to fmax inclusive, incrementing f by 1/T
    f_seq = lal.CreateREAL8Sequence(len(fs)) #creates an object with two properties: length, data (np array)
    f_seq.data = fs
    hplus, hcross = hpc(params, f_seq, modelstring) #1D arrays that depend only on frequency. here, we get h+/hx for exactly the frequencies in fs.
    
    ra = params['ra']
    dec = params['dec']
    tgps = params['tgps'] #GPS time at merger, i.e. at peak strain value (see Jupyter notebook). How to define this?
    tc = params['tc']
    tcoarse = params['tcoarse']

    DET_CODE = {'H1': lal.LHO_4K_DETECTOR,
                'L1': lal.LLO_4K_DETECTOR,
                'V1': lal.VIRGO_DETECTOR} #dictionary assigning numeric codes to each detector. inputted into lal.CachedDetectors
    
    ndet = len(detstrings) #no. of detectors

    #assign an index to the reference detector from the inputted detector array
    try:
        i_refdet = detstrings.index('L1') # Livingston is the (default) reference detector
    except:
        try:
            i_refdet = detstrings.index('H1') # If no Livingston, Hanford is the reference detector
        except:
            try:
                i_refdet = detstrings.index('V1') # If no Livingston or Hanford, Virgo is the reference detector
            except:
                raise ValueError('detstring must contain H1, L1, or V1')

    det_location = [lal.CachedDetectors[DET_CODE[det_name]].location #get location of each detector you're working with in (x, y, z) in meters
                                 for det_name in detstrings] #xyz system rotates with earth here - so this is time-independent

    gmst = lal.GreenwichMeanSiderealTime(tgps) #calculate current GMST angle, given tgps
    gha = gmst - ra
    esrc = np.array([np.cos(dec)*np.cos(gha), -np.cos(dec)*np.sin(gha), np.sin(dec)]) #vector from earth to source (minus k in notes) in (x, y, z), same coords as yours in toy problem and det_location

    time_delay_offset = np.array([-np.dot(esrc, det_location[i_det]) / lal.C_SI for i_det in range(ndet)]) #calculate time_delay_offset for each detector you're working with (see notes)
    time_delay_offset_refdet = time_delay_offset[i_refdet] #get offset for current ref detector

    time_delay = time_delay_offset - time_delay_offset_refdet #time delays between detectors and reference detector. t_delay[det] in project motivation notes.
    #no need to make any changes other than vectorizing the time for this.
    
    Fplus, Fcross = compute_response_coeffs(params, detstrings)
    
    # Detector strain; see project motivation notes. structure? and how does this only get nonzero values?
    strain_nonzero = ((Fplus[:, np.newaxis] * hplus
             + Fcross[:, np.newaxis] * hcross)
            * np.exp(-2j * np.pi * f_seq.data
                     * (tcoarse + tc + time_delay[:, np.newaxis])))

    strain = np.zeros([ndet,T*Fs//2+1], dtype=complex) #initialize entire strain array
    strain[:,int(f_min*T):int(f_max*T)+1] = strain_nonzero #fill in the nonzero strains
    return strain

def fd_to_td(strain_fd, T, Fs): 
    """
    Transforms a frequency domain strain to the time domain.
    strain_fd: frequency series array
    T: the length of the corresponding time series (s)
    Fs: the sampling rate of the corresponding time series (Hz)
    """
    ndet = strain_fd.shape[0]
    strain_td = np.zeros((ndet, T*Fs))
    for i in range(ndet):
        strain_td[i] = Fs*np.fft.irfft(strain_fd[i]) #irfft takes in a frequency series up to Nyquist freq. and outputs your full-length strain_td
    return strain_td

def td_to_fd(strain_td, T, Fs): 
    """
    Transforms a time domain strain to the frequency domain.
    strain_td: time series array
    T: the length of the corresponding time series (s)
    Fs: the sampling rate of the corresponding time series (Hz)
    """
    ndet = strain_td.shape[0]
    strain_fd = np.zeros((ndet, T*Fs//2+1), dtype=complex)
    for i in range(ndet):
        strain_fd[i] = 1/Fs*np.fft.rfft(strain_td[i]) #rfft takes in a time series and outputs your strain_fd which is up to Nyq freq. (length floor(nfft/2)+1)
        #also have np.fft, which stores EVERYTHING for strain_fd, not just up to Nyquist
    return strain_fd

def compute_strain_td(params, T, Fs, modelstring, detstrings = ['H1', 'L1', 'V1']): 
    """
    Computes the time domain strain in the detectors in detstrings for the parameters in the dict params.
    params: parameter dictionary
    T: the length of the corresponding time series (s)
    Fs: the sampling rate of the corresponding time series (Hz)
    modelstring: string specifying which model to use, some possible values include "TaylorF2", IMRPhenomD", "IMRPhenomXPHM"
    detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
    """
    ndet = len(detstrings)
    strain_fd = compute_strain_fd(params, T, Fs, modelstring, detstrings=detstrings)
    strain_td = np.zeros((ndet, T*Fs)) #can replace code hereon with strain_td = fd_to_td(strain_fd, T, Fs)
    for i in range(ndet):
        strain_td[i] = Fs*np.fft.irfft(strain_fd[i])
    return strain_td

def compute_strain_td_rot(params, T, Fs, modelstring, detstrings = ['H1', 'L1', 'V1'], Fs_large=4096000): 
    """
    Computes the time domain strain for a rotating Earth in the detectors in detstrings for the parameters in the dict params.
    params: parameter dictionary
    T: the length of the corresponding time series (s)
    Fs: the sampling rate of the corresponding time series (Hz)
    modelstring: string specifying which model to use, some possible values include "TaylorF2", IMRPhenomD", "IMRPhenomXPHM"
    detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
    """
    f_min = params['f_min']
    f_max = params['f_max']
    ndet = len(detstrings)
    fs = np.linspace(f_min, f_max, int((f_max-f_min)*T)+1) #array going from fmin to fmax inclusive, incrementing f by 1/T
    f_seq = lal.CreateREAL8Sequence(len(fs)) #creates an object with two properties: length, data (np array)
    f_seq.data = fs
    
    #create padded fd arrays of hplus and hcross up to Fs_large, and then add nonzero values
    hplus_fd_nonzero, hcross_fd_nonzero = hpc(params, f_seq, modelstring) #1D arrays that depend only on frequency. here, we get h+/hx for exactly the frequencies in fs.
    print(hplus_fd_nonzero.shape)
    hplus_fd = np.zeros(T*Fs_large//2+1, dtype=complex)
    hplus_fd[int(f_min*T):int(f_max*T)+1] = hplus_fd_nonzero
    hcross_fd = np.zeros(T*Fs_large//2+1, dtype=complex)
    hcross_fd[int(f_min*T):int(f_max*T)+1] = hcross_fd_nonzero    

#     hplus_fd = np.zeros([ndet,T*Fs_large//2+1], dtype=complex)
#     hplus_fd[:,int(f_min*T):int(f_max*T)+1] = hplus_fd_nonzero
#     hcross_fd = np.zeros([ndet,T*Fs_large//2+1], dtype=complex)
#     hcross_fd[:,int(f_min*T):int(f_max*T)+1] = hcross_fd_nonzero
    
    #fourier transform hplus and hcross to get fine td versions
    hplus_td_fine, hcross_td_fine = Fs*np.fft.irfft(hplus_fd), Fs*np.fft.irfft(hcross_fd)
    
    #get time delay at each time 
    t_delay = compute_time_delay(params, T, Fs, detstrings = detstrings)
    
    #compute time shift = t + t_delay
    t_signal = np.zeros([ndet, T*Fs])
    for i in range(ndet):
        t_signal[i:] = np.linspace(0, T-1/Fs, T*Fs)
    t_shift = t_signal + t_delay
    print(t_shift)
    
    #compute fine indices and query hplus_td and hcross_td at them
    fine_index = ((t_shift*Fs_large) % (T*Fs_large)).astype(np.int32)
    print(fine_index)
    hplus_td, hcross_td = hplus_td_fine[fine_index], hcross_td_fine[fine_index]
    
    #get F+, Fx coefficients
    Fplus, Fcross = compute_response_coeffs_time_dep(params, T, Fs, detstrings = detstrings)
    
    #return strain in time domain
    strain_td = Fplus*hplus_td + Fcross*hcross_td
    return strain_td

def compute_strain_fd_rot(params, T, Fs, modelstring, detstrings = ['H1', 'L1', 'V1'], Fs_large=4096000):
    """
    Computes the frequency domain strain for a rotating Earth in the detectors in detstrings for the parameters in the dict params.
    params: parameter dictionary
    T: the length of the corresponding time series (s)
    Fs: the sampling rate of the corresponding time series (Hz)
    modelstring: string specifying which model to use, some possible values include "TaylorF2", IMRPhenomD", "IMRPhenomXPHM"
    detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
    """
    strain_td = compute_strain_td_rot(params, T, Fs, modelstring, detstrings=detstrings, Fs_large=Fs_large)
    strain_fd = td_to_fd(strain_td)
    return strain_fd   
    
def inn_prod(x, y, fs, psd, T, fmin, fmax):
    """
    Computes the inner product between two frequency series for the same detector.
    x, y: frequency series arrays
    psd: the psd array for the source of x and y
    T: the length of the corresponding time series (s)
    """
    return np.sum((x*np.conj(y)/(psd*T/4))*np.heaviside(fs-fmin, 0)*np.heaviside(fmax-fs, 0))

def dh(d, h, fs, psd, T, fmin, fmax):
    """
    Computes the inner product between d and h (= sum of inner product of each detector in d/h).
    d, h: frequency series arrays for multiple detectors
    fs: the full array of frequencies
    psd: the psd array for the source of x and y
    T: the length of the corresponding time series (s)
    fmin: the minimum frequency used in computing the inner product (Hz)
    fmax: the maximum frequency used in computing the inner product (Hz)
    """
    ndet = d.shape[0]
    return np.sum([inn_prod(d[i], h[i], fs, psd[i], T, fmin, fmax) for i in range(ndet)])

def hh(h, fs, psd, T, fmin, fmax):
    """
    Computes the inner product between h and itself.
    h: frequency series array
    fs: the full array of frequencies
    psd: the psd array for the source of x and y
    T: the length of the corresponding time series (s)
    fmin: the minimum frequency used in computing the inner product (Hz)
    fmax: the maximum frequency used in computing the inner product (Hz)
    """
    ndet = h.shape[0]
    return np.sum([inn_prod(h[i], h[i], fs, psd[i], T, fmin, fmax) for i in range(ndet)])

def overlap_cosine(h1, h2, fs, psd, T, fmin, fmax):
    """
    Computes the overlap cosine between h1 and h2 as is.
    h1, h2: frequency series arrays
    fs: the full array of frequencies
    psd: the psd array for the source of x and y
    T: the length of the corresponding time series (s)
    Fs: the sampling rate of the corresponding time series (Hz)
    fmin: the minimum frequency used in computing the overlap cosine (Hz)
    fmax: the maximum frequency used in computing the overlap cosine (Hz)
    """
    h1h2 = np.abs(dh(h1, h2, fs, psd, T, fmin, fmax))
    h1h1 = np.real(hh(h1, fs, psd, T, fmin, fmax))
    h2h2 = np.real(hh(h2, fs, psd, T, fmin, fmax))

    return (h1h2/np.sqrt(h1h1*h2h2))

def compute_match_phase(h1, h2, fs, psd, T, Fs, fmin, fmax):
    """
    Computes the phase and time offset added to h1 to optimally match with h2.
    h1, h2: frequency series arrays
    psd: the psd array for the source of x and y
    T: the length of the corresponding time series (s)
    Fs: the sampling rate of the corresponding time series (Hz)
    """
    ndet = h1.shape[0]
    fft_arg = h1*np.conj(h2)/(psd*T/4)
    overlaps = np.sum([np.fft.irfft(fft_arg[i]) for i in range(ndet)], axis=0)
    max_ind = np.argmax(np.abs(overlaps))
    nfft = T*Fs
    ts = np.linspace(0, (nfft-1)/Fs, nfft)
    time = ts[max_ind]
    time_corrected_overlap = inn_prod(h1*np.exp(2*np.pi*1j*(time*fs)), h2, fs, psd, T, fmin, fmax)
    phase = -np.angle(time_corrected_overlap)
    return phase, time

def match_phase_td(h1, h2, fs, psd, T, Fs, fmin, fmax):
    """
    Add the optimal phase and time offset to h1 to optimally match with h2.
    h1, h2: time series arrays
    fs: the full array of frequencies
    psd: the psd array for the source of x and y
    T: the length of the corresponding time series (s)
    Fs: the sampling rate of the corresponding time series (Hz)
    """
    h1f = td_to_fd(h1, T, Fs)
    h2f = td_to_fd(h2, T, Fs)
    phi, t = compute_match_phase(h1f, h2f, fs, psd, T, Fs, fmin, fmax)
    h1f_opt = add_phase(h1f, fs, phi, t)
    return fd_to_td(h1f_opt, T, Fs)

def match_phase_fd(h1, h2, fs, psd, T, Fs, fmin, fmax):
    """
    Add the optimal phase and time offset to h1 to optimally match with h2.
    h1, h2: frequency series arrays
    fs: the full array of frequencies
    psd: the psd array for the source of x and y
    T: the length of the corresponding time series (s)
    Fs: the sampling rate of the corresponding time series (Hz)
    """
    phi, t = compute_match_phase(h1, h2, fs, psd, T, Fs, fmin, fmax)
    return add_phase(h1, fs, phi, t)

def add_phase(h1, fs, phase, time):
    """
    Add the given phase and time offset to h1.
    h1 frequency series array
    fs: the full array of frequencies
    phase: a phase offset to add to h1 (radians)
    time: a time offset to add to h1 (s)
    """
    return h1*np.exp(1j*((2*np.pi*time*fs)+phase))

def faithfulness(h1, h2, fs, psd, T, Fs, fmin, fmax):
    """
    Computes the optimal overlap cosine between h1 and h2 optimizing for time and phase offsets.
    h1, h2: frequency series arrays
    fs: the full array of frequencies
    psd: the psd array for the source of x and y
    T: the length of the corresponding time series (s)
    Fs: the sampling rate of the corresponding time series (Hz)
    fmin: the minimum frequency used in computing the overlap cosine (Hz)
    fmax: the maximum frequency used in computing the overlap cosine (Hz)
    """
    phi, t = compute_match_phase(h1, h2, fs, psd, T, Fs)
    h1opt = add_phase(h1, fs, phi, t)
    return overlap_cosine(h1opt, h2, fs, psd, T, fmin, fmax)