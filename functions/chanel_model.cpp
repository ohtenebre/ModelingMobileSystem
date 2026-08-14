#include "header.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

std::vector<std::complex<float>> chanel_model(const std::vector<std::complex<float>> ofdm_symbols, SharedData &sd)
{
    int N_b = sd.ChannelCfg.N_b;
    std::vector<int> D(N_b);
    std::vector<int> L(N_b);
    std::vector<float> G(N_b);

    std::random_device rd;
    static std::mt19937 gen(42);
    // std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(10, 500);

    for (int i = 0; i < N_b; ++i)
        D[i] = dist(gen);

    std::sort(D.begin(), D.end());

    float B = sd.ChannelCfg.B;
    float Ts = 1.0f / B;
    float c = 3e8;

    for (int i = 0; i < N_b; ++i)
    {
        float tau = (static_cast<float>(D[i] - D[0]) / (c * Ts));
        L[i] = static_cast<int>(std::round(tau));
    }

    int carrier_f = sd.ChannelCfg.carrier_freq;
    for (int i = 0; i < N_b; ++i)
    {
        if (carrier_f > 0)
            G[i] = static_cast<float>(c / (4 * M_PI * D[i] * carrier_f));
        else
            G[i] = 1.0f;
    }

    float maxG = *std::max_element(G.begin(), G.end());
    for (auto &g : G)
        g /= maxG;

    std::vector<std::complex<float>> rx_signal(ofdm_symbols.size(), { 0.f, 0.f });

    for (int i = 0; i < N_b; ++i)
    {
        int shift = L[i];
        float gain = G[i];

        if (shift >= ofdm_symbols.size())
            continue;

        for (size_t k = 0; k < ofdm_symbols.size() - shift; ++k)
        {
            rx_signal[k + shift] += ofdm_symbols[k] * gain;
        }
    }

    float noise_pwr_linear = std::pow(10.0f, sd.ChannelCfg.N_0 / 10.0f);
    float sigma = std::sqrt(noise_pwr_linear / 2.0f);

    std::normal_distribution<float> noise_dist(0.0f, sigma);
    for (auto &sample : rx_signal)
        sample += std::complex<float>(noise_dist(gen), noise_dist(gen));

    return rx_signal;
}