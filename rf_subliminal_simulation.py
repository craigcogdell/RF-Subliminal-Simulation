# RF Subliminal Direct Transmission for SDR Hardware
# Purpose: Creates and transmits a pulsed RF signal mimicking the microwave auditory effect (Frey effect)
# to deliver subliminal messages perceived as internal thoughts, inspired by US Patent 4,777,529.
# This generates waveforms and directly transmits them using SDR hardware.
# WARNING: Use only with legal/safe hardware setup (e.g., FCC-compliant SDR, low power, proper licensing).
# Comments explain each section's role in the subliminal process and RF context.

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy import signal  # For envelope shaping
import argparse
import os
import sys
import time
import threading
import queue

# SDR libraries - uncomment the one you're using
try:
    # For HackRF
    import hackrf
    SDR_TYPE = "hackrf"
except ImportError:
    try:
        # For USRP
        import uhd
        SDR_TYPE = "usrp"
    except ImportError:
        try:
            # For RTL-SDR (transmit not supported, but kept for completeness)
            import rtlsdr
            SDR_TYPE = "rtlsdr"
        except ImportError:
            try:
                # For LimeSDR
                import limesdr
                SDR_TYPE = "limesdr"
            except ImportError:
                # For SoapySDR (generic SDR interface)
                try:
                    import SoapySDR
                    from SoapySDR import SOAPY_SDR_TX, SOAPY_SDR_CF32
                    SDR_TYPE = "soapysdr"
                except ImportError:
                    print("WARNING: No SDR library found. Running in simulation mode only.")
                    SDR_TYPE = "none"

def create_rf_signal(fs=2e6, duration=5.0, pulse_rate=50, pulse_width=0.001, 
                    audio_freq=100, rf_freq=2.45e9, output_dir=None, visualize=True):
    """
    Create RF subliminal signal with configurable parameters for direct transmission
    
    Parameters:
    -----------
    fs : float
        Sampling frequency in Hz (default 2 MHz for SDR)
    duration : float
        Duration of signal in seconds
    pulse_rate : int
        Pulses per second (Hz)
    pulse_width : float
        Pulse duration in seconds
    audio_freq : int
        Frequency of subliminal audio message in Hz
    rf_freq : float
        RF carrier frequency in Hz (default 2.45 GHz for microwave auditory effect)
    output_dir : str
        Directory to save output files (default: current directory)
    visualize : bool
        Whether to generate and save visualization plots
    
    Returns:
    --------
    dict
        Dictionary containing generated signals and parameters
    """
    try:
        # Set output directory
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        
        # --- Section 1: Define Signal Parameters ---
        print(f"Generating signal with {fs/1e6:.1f} MHz sample rate...")
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)  # Time array
        samples_per_pulse = int(pulse_width * fs)  # Number of samples per pulse

        # --- Section 2: Generate Subliminal Audio Message ---
        print("Creating subliminal audio modulation...")
        modulating_signal = np.sin(2 * np.pi * audio_freq * t)  # Generate sine wave
        # Apply a Hann window to create a smooth envelope, simulating spoken words' rise/fall
        envelope = signal.windows.hann(int(fs * duration))
        modulating_signal *= envelope
        # Scale amplitude low to ensure subliminal effect (-30 to -40 dB equivalent)
        modulating_signal *= 0.01  # Arbitrary scaling for subliminal effect

        # --- Section 3: Generate RF Carrier Wave ---
        # Note: In actual transmission, the SDR hardware handles the carrier frequency
        # This is just for visualization and simulation purposes
        print("Creating carrier wave representation...")
        # Use a lower frequency for visualization only
        vis_freq = 10000  # 10 kHz for visualization
        carrier = np.sin(2 * np.pi * vis_freq * t[:int(fs*0.001)])  # Only generate a small portion for viz

        # --- Section 4: Create Pulsed RF Signal (Frey Effect) ---
        print("Generating pulsed RF signal...")
        rf_pulses = np.zeros_like(t, dtype=np.complex64)  # Initialize as complex for IQ
        pulse_times = np.arange(0, duration, 1 / pulse_rate)  # Times for pulse starts
        for pt in pulse_times:
            pulse_start_idx = int(pt * fs)  # Start index of pulse
            if pulse_start_idx >= len(t):
                break
            pulse_end_idx = min(pulse_start_idx + samples_per_pulse, len(t))  # End index
            # Modulate pulse amplitude by the subliminal signal's absolute value at this time
            audio_amp = abs(modulating_signal[pulse_start_idx])
            # Create complex IQ pulse (real component only for AM modulation)
            rf_pulses[pulse_start_idx:pulse_end_idx] = audio_amp + 0j
        
        # --- Section 5: Prepare IQ Data for SDR Transmission ---
        print("Preparing IQ data for transmission...")
        # Normalize to range -1 to 1 for SDR transmission
        max_val = max(np.max(np.abs(rf_pulses.real)), np.max(np.abs(rf_pulses.imag)))
        if max_val > 0:
            rf_pulses /= max_val
        
        # Save as binary file for direct SDR use
        bin_file_path = os.path.join(output_dir, 'rf_subliminal_iq.bin')
        rf_pulses.tofile(bin_file_path)
        print(f"IQ samples saved as binary file '{bin_file_path}' for direct SDR use.")

        # --- Section 6: Optional - Simulate Perceived Audio ---
        print("Generating simulated perceived audio...")
        perceived_audio = np.abs(rf_pulses.real) * 32767  # Scale to 16-bit audio
        perceived_audio = perceived_audio.astype(np.int16)  # Convert to int16 for .wav
        from scipy.io import wavfile
        wav_file_path = os.path.join(output_dir, 'simulated_perceived_clicks.wav')
        wavfile.write(wav_file_path, int(fs // 100), perceived_audio[::100])  # Downsample for audible demo
        print(f"Perceived audio saved as '{wav_file_path}'.")

        # --- Section 7: Visualize the Signals ---
        if visualize:
            print("Generating visualizations...")
            fig, axs = plt.subplots(3, 1, figsize=(12, 10))

            # Plot subliminal audio (first 0.2s for clarity)
            viz_samples = int(fs * 0.2)
            axs[0].plot(t[:viz_samples], modulating_signal[:viz_samples])
            axs[0].set_title('Subliminal Audio Message (Modulating Signal)')
            axs[0].set_ylabel('Amplitude')
            axs[0].grid(True)

            # Plot RF carrier (small portion to show frequency)
            small_t = t[:int(fs*0.001)]
            axs[1].plot(small_t, carrier)
            axs[1].set_title(f'RF Carrier Wave Representation (actual: {rf_freq/1e6:.1f} MHz)')
            axs[1].set_ylabel('Amplitude')
            axs[1].grid(True)

            # Plot pulsed RF signal (first 0.1s to show pulses)
            viz_samples = int(fs * 0.1)
            axs[2].plot(t[:viz_samples], rf_pulses[:viz_samples].real)
            axs[2].set_title('Pulsed RF Signal (Frey Effect)')
            axs[2].set_xlabel('Time (s)')
            axs[2].set_ylabel('Amplitude')
            axs[2].grid(True)

            plt.tight_layout()
            plot_file_path = os.path.join(output_dir, 'rf_subliminal_visualization.png')
            plt.savefig(plot_file_path)  # Save plot for reference
            print(f"Visualization saved as '{plot_file_path}'.")
            plt.close()  # Close the plot to free memory
        
        # Create a metadata file with transmission parameters
        metadata_path = os.path.join(output_dir, 'rf_transmission_parameters.txt')
        with open(metadata_path, 'w') as f:
            f.write(f"Sample Rate: {fs} Hz\n")
            f.write(f"Center Frequency: {rf_freq} Hz\n")
            f.write(f"IQ Format: complex64\n")
            f.write(f"Duration: {duration} seconds\n")
            f.write(f"Pulse Rate: {pulse_rate} Hz\n")
            f.write(f"Pulse Width: {pulse_width} seconds\n")
            f.write(f"Audio Frequency: {audio_freq} Hz\n")
        print(f"Transmission parameters saved as '{metadata_path}'.")
        
        return {
            'iq_signal': rf_pulses,
            'modulating_signal': modulating_signal,
            'fs': fs,
            'duration': duration,
            't': t,
            'rf_freq': rf_freq
        }
    
    except Exception as e:
        print(f"Error in RF signal generation: {str(e)}")
        return None

def transmit_signal_soapysdr(iq_data, sample_rate, center_freq, gain=20, repeat=False):
    """
    Transmit IQ data using SoapySDR (works with many SDR devices)
    """
    try:
        # Find available devices
        results = SoapySDR.Device.enumerate()
        if len(results) == 0:
            print("No SDR devices found")
            return False
        
        # Create device instance
        print(f"Using device: {results[0]['driver']}")
        sdr = SoapySDR.Device(results[0])
        
        # Setup the device for transmission
        sdr.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
        sdr.setFrequency(SOAPY_SDR_TX, 0, center_freq)
        sdr.setGain(SOAPY_SDR_TX, 0, gain)
        
        # Create a transmit stream
        tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32)
        sdr.activateStream(tx_stream)
        
        # Transmit the data
        print(f"Transmitting signal at {center_freq/1e6:.2f} MHz...")
        
        # Buffer size for transmission
        buffer_size = 1024
        
        if repeat:
            # Continuous transmission mode
            print("Starting continuous transmission (Ctrl+C to stop)...")
            try:
                while True:
                    for i in range(0, len(iq_data), buffer_size):
                        buf = iq_data[i:i+buffer_size]
                        if len(buf) < buffer_size:
                            buf = np.pad(buf, (0, buffer_size - len(buf)), 'constant')
                        sdr.writeStream(tx_stream, [buf], len(buf))
            except KeyboardInterrupt:
                print("\nTransmission stopped by user")
        else:
            # Single transmission
            for i in range(0, len(iq_data), buffer_size):
                buf = iq_data[i:i+buffer_size]
                if len(buf) < buffer_size:
                    buf = np.pad(buf, (0, buffer_size - len(buf)), 'constant')
                sdr.writeStream(tx_stream, [buf], len(buf))
            print("Transmission complete")
        
        # Cleanup
        sdr.deactivateStream(tx_stream)
        sdr.closeStream(tx_stream)
        return True
        
    except Exception as e:
        print(f"Error during transmission: {str(e)}")
        return False

def transmit_signal_hackrf(iq_data, sample_rate, center_freq, gain=20, repeat=False):
    """
    Transmit IQ data using HackRF
    """
    try:
        # Initialize HackRF device
        device = hackrf.HackRF()
        
        # Setup the device for transmission
        device.sample_rate = sample_rate
        device.center_freq = center_freq
        device.vga_gain = min(gain, 47)  # VGA gain (0-47 dB)
        device.enable_amp = (gain > 20)  # Enable amplifier for higher gain
        
        # Convert to correct format for HackRF
        iq_bytes = (iq_data * 127).astype(np.int8)
        
        # Transmit the data
        print(f"Transmitting signal at {center_freq/1e6:.2f} MHz...")
        
        if repeat:
            # Continuous transmission mode
            print("Starting continuous transmission (Ctrl+C to stop)...")
            try:
                while True:
                    device.transmit(iq_bytes)
            except KeyboardInterrupt:
                print("\nTransmission stopped by user")
        else:
            # Single transmission
            device.transmit(iq_bytes)
            print("Transmission complete")
        
        # Cleanup
        device.close()
        return True
        
    except Exception as e:
        print(f"Error during HackRF transmission: {str(e)}")
        return False

def transmit_from_file(file_path, sample_rate, center_freq, gain=20, repeat=False):
    """
    Transmit IQ data from a binary file
    """
    try:
        # Load IQ data from file
        print(f"Loading IQ data from {file_path}...")
        iq_data = np.fromfile(file_path, dtype=np.complex64)
        
        # Transmit based on available SDR type
        if SDR_TYPE == "soapysdr":
            return transmit_signal_soapysdr(iq_data, sample_rate, center_freq, gain, repeat)
        elif SDR_TYPE == "hackrf":
            return transmit_signal_hackrf(iq_data, sample_rate, center_freq, gain, repeat)
        else:
            print(f"Transmission not supported for {SDR_TYPE}")
            return False
            
    except Exception as e:
        print(f"Error loading or transmitting file: {str(e)}")
        return False

def main():
    """Main function to parse command line arguments and run the RF transmission"""
    parser = argparse.ArgumentParser(description='RF Subliminal Direct Transmission for SDR Hardware')
    parser.add_argument('--fs', type=float, default=2e6, help='Sampling frequency in Hz (default: 2 MHz)')
    parser.add_argument('--duration', type=float, default=5.0, help='Duration of signal in seconds')
    parser.add_argument('--pulse-rate', type=int, default=50, help='Pulses per second (Hz)')
    parser.add_argument('--pulse-width', type=float, default=0.001, help='Pulse duration in seconds')
    parser.add_argument('--audio-freq', type=int, default=100, help='Frequency of subliminal audio message in Hz')
    parser.add_argument('--rf-freq', type=float, default=2.45e9, help='RF carrier frequency in Hz (default: 2.45 GHz)')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save output files')
    parser.add_argument('--no-visualize', action='store_false', dest='visualize', 
                        help='Disable visualization generation')
    parser.add_argument('--transmit', action='store_true', help='Transmit the signal using SDR hardware')
    parser.add_argument('--gain', type=float, default=20, help='Transmission gain (dB)')
    parser.add_argument('--repeat', action='store_true', help='Continuously repeat transmission')
    parser.add_argument('--file', type=str, help='Use existing IQ data file instead of generating new signal')
    
    args = parser.parse_args()
    
    # Print SDR availability
    print(f"SDR Support: {SDR_TYPE}")
    
    if args.file:
        # Transmit from existing file
        if args.transmit:
            if SDR_TYPE == "none":
                print("ERROR: No SDR library found. Cannot transmit.")
                return
            
            print(f"Transmitting from file: {args.file}")
            transmit_from_file(args.file, args.fs, args.rf_freq, args.gain, args.repeat)
        else:
            print("File specified but --transmit flag not set. Nothing to do.")
    else:
        # Generate new signal
        print("Starting RF Subliminal Signal Generation with the following parameters:")
        print(f"  Sampling Frequency: {args.fs/1e6:.2f} MHz")
        print(f"  Duration: {args.duration} seconds")
        print(f"  Pulse Rate: {args.pulse_rate} Hz")
        print(f"  Pulse Width: {args.pulse_width} seconds")
        print(f"  Audio Frequency: {args.audio_freq} Hz")
        print(f"  RF Frequency: {args.rf_freq/1e6:.2f} MHz")
        print(f"  Output Directory: {args.output_dir or 'Current directory'}")
        
        # Generate the signal
        result = create_rf_signal(
            fs=args.fs,
            duration=args.duration,
            pulse_rate=args.pulse_rate,
            pulse_width=args.pulse_width,
            audio_freq=args.audio_freq,
            rf_freq=args.rf_freq,
            output_dir=args.output_dir,
            visualize=args.visualize
        )
        
        if result and args.transmit:
            if SDR_TYPE == "none":
                print("ERROR: No SDR library found. Cannot transmit.")
                return
                
            print("\nPreparing for transmission...")
            if SDR_TYPE == "soapysdr":
                transmit_signal_soapysdr(result['iq_signal'], args.fs, args.rf_freq, args.gain, args.repeat)
            elif SDR_TYPE == "hackrf":
                transmit_signal_hackrf(result['iq_signal'], args.fs, args.rf_freq, args.gain, args.repeat)
            else:
                print(f"Direct transmission not supported for {SDR_TYPE}")
                print("You can use the generated binary file with GNU Radio or other SDR software.")
        
    print("\nOperation completed!")
    if not args.transmit:
        print("\nTo transmit the signal with SDR hardware:")
        print("1. Run this script with the --transmit flag")
        print("   Example: python rf_subliminal_simulation.py --transmit --rf-freq 915e6 --gain 20")
        print("2. For continuous transmission, add the --repeat flag")
        print("3. To use an existing IQ file, use the --file option")
        print("   Example: python rf_subliminal_simulation.py --transmit --file rf_subliminal_iq.bin --rf-freq 915e6")

if __name__ == "__main__":
    main()