import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
from scipy import signal
from scipy.io import wavfile
import math
import time

output_device = 4
input_device = 1
sd.default.device = (input_device, output_device)

SAMPLE_RATE = 44100  # Taxa de amostragem do audio
BIT_DURATION = 1.0   # 1 segundo por bit
FREQ_LOW = 440       # bit '0' (Lá)
FREQ_HIGH = 880      # bit '1' (Lá oitava)


def generate_tone(frequency, duration, sample_rate=SAMPLE_RATE):
    """
    Gera um tom senoidal
    
    Args:
        frequency: Frequência em Hz
        duration: Duração em segundos
        sample_rate: Taxa de amostragem
    
    Returns:
        array: Sinal de áudio
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Aplica janela para suavizar transições
    tone = np.sin(2 * np.pi * frequency * t)
    # Janela de Hanning para evitar cliques
    window = np.hanning(len(tone))
    return tone * window

def show(data:str,debug):
    if debug==True:
        print(data)

    
def encode_nrz(data_bits,debug=False):
    """
    Codifica dados usando NRZ
    
    Args:
        data_bits: string de bits (ex: "10110")
    
    Returns:
        array: Sinal de áudio codificado
    """
    audio_signal = np.array([])
    
    show(f"Codificando NRZ: {data_bits}",debug)
    
    for i, bit in enumerate(data_bits):
        if bit == '1':
            freq = FREQ_HIGH
            show(f"Bit {i}: '1' -> {freq} Hz",debug)
        else:
            freq = FREQ_LOW
            show(f"Bit {i}: '0' -> {freq} Hz",debug)
        
        tone = generate_tone(freq, BIT_DURATION)
        audio_signal = np.concatenate([audio_signal, tone])
    
    return audio_signal

def detect_frequency(audio_segment, sample_rate=SAMPLE_RATE):
    """
    Detecta a frequência dominante em um segmento de áudio
    
    Args:
        audio_segment: Segmento de áudio
        sample_rate: Taxa de amostragem
    
    Returns:
        float: Frequência detectada
    """
    # FFT para análise espectral
    fft = np.fft.fft(audio_segment)
    freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
    
    # Considera apenas frequências positivas
    magnitude = np.abs(fft[:len(fft)//2])
    freqs_positive = freqs[:len(freqs)//2]
    
    # Encontra o pico de frequência
    peak_idx = np.argmax(magnitude)
    detected_freq = abs(freqs_positive[peak_idx])
    
    return detected_freq

def frequency_to_bit(frequency, threshold=660):
    """
    Converte frequência detectada em bit
    
    Args:
        frequency: Frequência detectada
        threshold: Limiar para decisão (média entre FREQ_LOW e FREQ_HIGH)
    
    Returns:
        str: '0' ou '1'
    """
    return '1' if frequency > threshold else '0'


def decode_nrz(audio_signal, num_bits, sample_rate=SAMPLE_RATE,debug=False):
    """
    Decodifica sinal NRZ
    
    Args:
        audio_signal: Sinal de áudio
        num_bits: Número esperado de bits
        sample_rate: Taxa de amostragem
    
    Returns:
        str: Bits decodificados
    """
    samples_per_bit = int(sample_rate * BIT_DURATION)
    decoded_bits = ""
    
    show("Decodificando NRZ:",debug)
    
    for i in range(num_bits):
        start_idx = i * samples_per_bit
        end_idx = start_idx + samples_per_bit
        
        if end_idx > len(audio_signal):
            show(f"Aviso: Áudio muito curto para {num_bits} bits",debug)
            break
        
        # Analisa o meio do bit para evitar transições
        mid_start = start_idx + samples_per_bit // 4
        mid_end = end_idx - samples_per_bit // 4
        segment = audio_signal[mid_start:mid_end]
        
        freq = detect_frequency(segment, sample_rate)
        bit = frequency_to_bit(freq)
        decoded_bits += bit
        
        show(f"Bit {i}: freq={freq:.1f}Hz -> '{bit}'",debug)
    
    return decoded_bits


def adicionar_ruido(audio_signal, snr_db=-12):
    """
    Adiciona ruído gaussiano ao sinal
    
    Args:
        audio_signal: Sinal original
        snr_db: Relação sinal-ruído em dB
    
    Returns:
        array: Sinal com ruído
    """
    # Calcula potência do sinal
    signal_power = np.mean(audio_signal ** 2)
    
    # Calcula potência do ruído baseada no SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear


def qtd_bits_com_erro(mensagem):
    """
    Conta a quantidade de bits com erro
    
    Args:
        mensagem: mensagem binaria 
    
    Returns:
        int: Número de bits com erro
    """
    errors = 0
    for bit in mensagem:
        if bit == '?':
            errors += 1

    return errors


"""parte para o escutar o audio a partir do microfone"""

'''
def capturar_do_microfone(duracao_segundos):
    """
    Captura áudio do microfone
    
    Args:
        duracao_segundos: Duração da captura
    
    Returns:
        array: Áudio capturado
    """
    print(f"Iniciando captura por {duracao_segundos} segundos...")
    print("Reproduza o áudio no seu celular AGORA!")
    
    # Captura áudio
    audio_capturado = sd.rec(
        int(duracao_segundos * SAMPLE_RATE), 
        samplerate=SAMPLE_RATE, 
        channels=1
    )
    sd.wait()  # Aguarda terminar a captura
    
    print("Captura concluída!")
    return audio_capturado.flatten()

duracao = 28 * BIT_DURATION + 1  # +1 segundo de margem
audio_capturado = capturar_do_microfone(duracao)

# Salva captura para análise
sf.write('captura_microfone.wav', audio_capturado, SAMPLE_RATE)

# Tenta decodificar
print("\nTentando decodificar...")
decoded = decode_nrz(audio_capturado, 28)

print(f"Original: ?????")
print(f"Capturado: {decoded}")
'''


def decodificar_do_arquivo(nome_arquivo, sample_rate=SAMPLE_RATE, bit_duration=BIT_DURATION, debug=False):
    """
    Decodifica o sinal Manchester de um arquivo de áudio WAV, 
    calculando o número de bits automaticamente.
    
    Args:
        nome_arquivo (str): Nome do arquivo WAV a ser lido.
        sample_rate (int): Taxa de amostragem (padrão SAMPLE_RATE global).
        bit_duration (float): Duração de um bit em segundos (padrão BIT_DURATION global).
        debug (bool): Flag para mostrar mensagens de debug.
    
    Returns:
        str: Sequência de bits decodificada.
    """
    try:
        
        print(f"Lendo o arquivo de áudio: {nome_arquivo}")
        audio_signal, read_sample_rate = sf.read(nome_arquivo)
        
        if read_sample_rate != sample_rate:
            print(f" Aviso: Taxa de amostragem do arquivo ({read_sample_rate} Hz) é diferente da global ({sample_rate} Hz).")
            
        if audio_signal.ndim > 1:
            print(" Aviso: O arquivo tem múltiplos canais. Usando apenas o primeiro canal.")
            audio_signal = audio_signal[:, 0]
            
        total_samples = len(audio_signal)
        duracao_total = total_samples / sample_rate
        
        num_bits_estimado = math.floor(duracao_total / bit_duration)
        
        print(f"Duração total do áudio: {duracao_total:.4f} segundos.")
        print(f"Duração por bit: {bit_duration} segundos.")
        print(f"Estimativa de bits a decodificar: {num_bits_estimado}")
        
        print("Iniciando decodificação...")
        decoded_bits = decode_nrz(audio_signal, num_bits_estimado, sample_rate, debug)
        
        print("Decodificação concluída!")
        return decoded_bits

    except FileNotFoundError:
        print(f" ERRO: Arquivo '{nome_arquivo}' não encontrado.")
        return ""
    except Exception as e:
        print(f" Ocorreu um erro durante a leitura ou decodificação: {e}")
        return ""


print(f"\n--- Resultado da Decodificação Automática ---")
nome_do_arquivo = r"lab01\modulacao\dados_codificados\dados_119211122_44100hz.wav"
print(f"Original: ?????... (Desconhecido)") 
print(f"Decodificado do Arquivo: '{nome_do_arquivo}'")
print(f"Capturado: {decodificar_do_arquivo(nome_do_arquivo, debug=True)}")  

