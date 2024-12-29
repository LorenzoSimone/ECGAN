import torch
import torch.fft

def compute_spectral_coefficients(time_data):
    """
    Compute the real Fourier coefficients (spectral coefficients) from the time-domain data.
    
    Args:
        time_data (torch.Tensor): The input time-domain data (shape: [batch_size, time_length]).
        
    Returns:
        torch.Tensor: The spectral coefficients (shape: [batch_size, time_length//2 + 1]).
    """
    # Perform FFT on time-domain data and retain the real and imaginary parts
    spectral_coeffs = torch.fft.rfft(time_data, dim=-1)
    
    return spectral_coeffs

def convert_to_time_domain(spectral_coeffs):
    """
    Convert spectral coefficients (real Fourier coefficients) back to time-domain representation.
    
    Args:
        spectral_coeffs (torch.Tensor): The spectral coefficients (shape: [batch_size, time_length//2 + 1]).
        
    Returns:
        torch.Tensor: The time-domain data (shape: [batch_size, time_length]).
    """
    # Perform inverse FFT to get back to time domain
    time_data = torch.fft.irfft(spectral_coeffs, dim=-1)
    
    return time_data