#pragma once

#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fftw3.h>
#include <mutex>
#include <string>
#include <vector>

struct OFDMConfig
{
    int FFT_SIZE = 128;
    int RS = 6;
    float C = 0.25;
    int CP_ratio = 8;
};

struct ChannelConfig
{
    int N_b = 6;
    int B = 7e6;
    float carrier_freq = 1700e6;
    float N_0 = 0.0f;
};

struct UserData
{
    std::string tx_buf;
    std::string rx_buf;

    std::vector<uint8_t> bytes;
    std::vector<uint32_t> encoded;
    std::vector<uint32_t> interleaved;
    std::vector<std::complex<float>> qpsk;

    float ber = 0.f;
};

struct SharedData
{
    OFDMConfig OfdmCfg;
    ChannelConfig ChannelCfg;

    std::mutex mtx;

    UserData users[4];

    std::vector<std::complex<float>> symbols_rx;
    std::vector<std::complex<float>> ofdm_symbols;
    std::vector<std::complex<float>> signal;
    std::vector<float> rx_spectrum;
    std::vector<std::vector<std::complex<float>>> qpsk_gui;

    float BER = 0.f;
    std::vector<float> BER_vec;
    size_t ber_vec_offset = 0;
    size_t ber_vec_size = 0;
    std::atomic<size_t> ber_frames_processed{ 0 };

    std::vector<float> ber_curve;

    std::atomic<bool> back_running{ true };
    std::atomic<bool> input_flag{ false };
    std::atomic<bool> experiment{ false };
    std::atomic<bool> is_realtime{ false };

    fftwf_plan plan_ifft;
    fftwf_complex *in_ifft;
    fftwf_complex *out_ifft;

    fftwf_plan plan_fft;
    fftwf_complex *in_fft;
    fftwf_complex *out_fft;

    fftwf_plan plan_spectrum;
    fftwf_complex *in_spectrum;
    fftwf_complex *out_spectrum;

    int fft_size_alloc = 0;
};

constexpr size_t PACKET_SIZE = 100;

void back(SharedData &sd);

void ensure_fft_plans(SharedData &sd);

std::vector<uint8_t> encoder(std::string text);
std::string decoder(std::vector<uint8_t> &bytes);

std::vector<uint32_t> hamming_encoder(std::vector<uint8_t> &bytes);
std::vector<uint8_t> hamming_decoder(std::vector<uint32_t> &encoded_bytes);

std::vector<uint32_t> interleaving(std::vector<uint32_t> &hamming_encoded);
std::vector<uint32_t> deinterleaving(std::vector<uint32_t> interleaving_block);

std::vector<std::complex<float>> QPSK_modulator(const std::vector<uint32_t> &bits);
std::vector<uint32_t> QPSK_demodulator(std::vector<std::complex<float>> &symbols);

int compute_OFDM_params(const OFDMConfig &cfg, int &out_NSYM, int &out_RS, int &out_CP);
std::vector<std::complex<float>> ofdm_modulator(const std::vector<std::complex<float>> &symbols, SharedData &sd);
std::vector<std::complex<float>> ofdm_demodulator(const std::vector<std::complex<float>> &rx_signal, SharedData &sd);

std::vector<std::complex<float>> ofdma_modulator(SharedData &sd);
void ofdma_demodulator(const std::vector<std::complex<float>> &rx, SharedData &sd);

std::vector<std::complex<float>> chanel_model(const std::vector<std::complex<float>> ofdm_symbols, SharedData &sd);
std::vector<float> spectrum_calculate(const std::vector<std::complex<float>> &signal, SharedData &sd);

float BER(const std::vector<uint8_t> &tx, const std::vector<uint8_t> &rx);