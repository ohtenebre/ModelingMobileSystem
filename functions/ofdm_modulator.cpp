#include "header.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <fftw3.h>
#include <vector>

int compute_OFDM_params(const OFDMConfig &cfg, int &out_NSYM, int &out_RS, int &out_CP)
{
    const int guard = static_cast<int>(std::floor(cfg.C * cfg.FFT_SIZE / (1 + 2 * cfg.C)));
    const int N_active = cfg.FFT_SIZE - 2 * guard;

    out_RS = static_cast<int>(std::floor(static_cast<double>(N_active) / cfg.RS));

    out_NSYM = N_active - out_RS;
    out_CP = cfg.FFT_SIZE / cfg.CP_ratio;

    return cfg.FFT_SIZE;
}

void ensure_fft_plans(SharedData &sd)
{
    const int size = sd.OfdmCfg.FFT_SIZE;
    if (sd.fft_size_alloc == size)
        return;

    if (sd.fft_size_alloc > 0)
    {
        fftwf_destroy_plan(sd.plan_ifft);
        fftwf_destroy_plan(sd.plan_fft);
        fftwf_free(sd.in_ifft);
        fftwf_free(sd.out_ifft);
        fftwf_free(sd.in_fft);
        fftwf_free(sd.out_fft);
    }

    sd.in_ifft = fftwf_alloc_complex(size);
    sd.out_ifft = fftwf_alloc_complex(size);
    sd.in_fft = fftwf_alloc_complex(size);
    sd.out_fft = fftwf_alloc_complex(size);

    sd.plan_ifft = fftwf_plan_dft_1d(size, sd.in_ifft, sd.out_ifft, FFTW_BACKWARD, FFTW_ESTIMATE);
    sd.plan_fft = fftwf_plan_dft_1d(size, sd.in_fft, sd.out_fft, FFTW_FORWARD, FFTW_ESTIMATE);

    sd.fft_size_alloc = size;
}

std::vector<std::complex<float>> ofdm_modulator(const std::vector<std::complex<float>> &symbols, SharedData &sd)
{
    int NSYM, RS, CP;
    const int N_fft = compute_OFDM_params(sd.OfdmCfg, NSYM, RS, CP);
    const int N_pl = NSYM + RS;
    const int NZ = static_cast<int>(std::floor(sd.OfdmCfg.C * N_pl));

    std::vector<std::complex<float>> output;
    size_t num_symbols = (symbols.size() + NSYM - 1) / NSYM;
    output.reserve(num_symbols * (N_fft + CP));

    const std::complex<float> pilot(0.707f, 0.707f);

    size_t pos = 0;
    while (pos < symbols.size())
    {
        size_t chunk_size = std::min(NSYM, static_cast<int>(symbols.size() - pos));
        std::vector<std::complex<float>> chunk(symbols.begin() + pos, symbols.begin() + pos + chunk_size);
        if (chunk_size < NSYM)
            chunk.resize(NSYM, { 0.f, 0.f });

        std::vector<bool> is_pilot(N_fft, false);
        std::vector<std::complex<float>> freq(N_fft, { 0.f, 0.f });
        for (int i = 0; i < RS; ++i)
        {
            int j = NZ + i * sd.OfdmCfg.RS;
            if (j < N_fft)
            {
                freq[j] = pilot;
                is_pilot[j] = true;
            }
        }

        int data_ptr = 0;
        for (int i = NZ; i < NZ + N_pl && data_ptr < NSYM; ++i)
        {
            if (!is_pilot[i])
            {
                freq[i] = chunk[data_ptr++];
            }
        }

        std::vector<std::complex<float>> time_sym(N_fft);

        std::copy_n(freq.begin(), N_fft, reinterpret_cast<std::complex<float> *>(sd.in_ifft));
        fftwf_execute(sd.plan_ifft);
        std::copy_n(reinterpret_cast<std::complex<float> *>(sd.out_ifft), N_fft, time_sym.begin());

        for (auto &s : time_sym)
            s /= static_cast<float>(N_fft);

        for (int i = sd.OfdmCfg.FFT_SIZE - CP; i < sd.OfdmCfg.FFT_SIZE; ++i)
            output.push_back(time_sym[i]);

        output.insert(output.end(), time_sym.begin(), time_sym.end());

        pos += chunk_size;
    }

    return output;
}

std::vector<std::complex<float>> ofdm_demodulator(const std::vector<std::complex<float>> &rx_signal, SharedData &sd)
{
    int NSYM, RS, CP;
    const int N_fft = compute_OFDM_params(sd.OfdmCfg, NSYM, RS, CP);
    const int N_pl = NSYM + RS;
    const int NZ = static_cast<int>(std::floor(sd.OfdmCfg.C * N_pl));

    std::vector<std::complex<float>> output;
    size_t symbol_len = N_fft + CP;
    size_t num_symbols = rx_signal.size() / symbol_len;
    output.reserve(num_symbols * NSYM);

    std::vector<int> pilot_idx;
    pilot_idx.reserve(RS);
    for (int i = 0; i < RS; ++i)
    {
        int j = NZ + i * sd.OfdmCfg.RS;
        if (j < N_fft)
            pilot_idx.push_back(j);
    }

    std::vector<bool> is_pilot(N_fft, false);
    for (int p : pilot_idx)
        is_pilot[p] = true;

    std::vector<std::complex<float>> freq(N_fft);
    std::vector<std::complex<float>> H(N_fft);
    std::vector<std::complex<float>> freq_eq(N_fft);

    const std::complex<float> pilot(0.707f, 0.707f);

    for (size_t sym = 0; sym < num_symbols; ++sym)
    {
        size_t pos = sym * symbol_len;
        std::copy_n(rx_signal.begin() + pos + CP, N_fft, reinterpret_cast<std::complex<float> *>(sd.in_fft));

        fftwf_execute(sd.plan_fft);

        std::copy_n(reinterpret_cast<std::complex<float> *>(sd.out_fft), N_fft, freq.begin());

        for (int p : pilot_idx)
            H[p] = freq[p] / pilot;

        for (size_t p = 0; p + 1 < pilot_idx.size(); ++p)
        {
            int k1 = pilot_idx[p];
            int k2 = pilot_idx[p + 1];
            float mag1 = std::abs(H[k1]), mag2 = std::abs(H[k2]);
            float phi1 = std::arg(H[k1]), phi2 = std::arg(H[k2]);

            float dphi = phi2 - phi1;

            for (int k = k1 + 1; k < k2; ++k)
            {
                float a = float(k - k1) / float(k2 - k1);
                H[k] = std::polar(mag1 + a * (mag2 - mag1), phi1 + a * dphi);
            }
        }

        if (!pilot_idx.empty())
        {
            int first = pilot_idx.front();
            int last = pilot_idx.back();

            for (int k = 0; k < first; ++k)
                H[k] = H[first];

            for (int k = last + 1; k < N_fft; ++k)
                H[k] = H[last];
        }

        for (int i = 0; i < N_fft; ++i)
        {
            if (std::abs(H[i]) > 1e-6f)
                freq_eq[i] = freq[i] / H[i];
            else
                freq_eq[i] = freq[i];
        }

        int data_pos = 0;
        for (int i = NZ; i < NZ + N_pl && data_pos < NSYM; ++i)
        {
            if (!is_pilot[i])
            {
                output.push_back(freq_eq[i]);
                data_pos++;
            }
        }
    }

    return output;
}

std::vector<float> spectrum_calculate(const std::vector<std::complex<float>> &signal, SharedData &sd)
{
    std::vector<std::complex<float>> local_signal = signal;
    local_signal.resize(1024, { 0.f, 0.f });

    std::copy_n(local_signal.data(), 1024, reinterpret_cast<std::complex<float> *>(sd.in_spectrum));

    fftwf_execute(sd.plan_spectrum);

    std::vector<float> spectrum_db(1024);
    std::complex<float> *out_ptr = reinterpret_cast<std::complex<float> *>(sd.out_spectrum);

    for (int i = 0; i < 1024; ++i)
    {
        float power = std::norm(out_ptr[i]);
        spectrum_db[i] = 10.0f * std::log10(power + 1e-12f);
    }

    return spectrum_db;
}

static void build_user_subcarriers(SharedData &sd, std::vector<std::vector<int>> &user_sc, int &guard_out, int &usable_out)
{
    int NSYM, RS, CP;
    compute_OFDM_params(sd.OfdmCfg, NSYM, RS, CP);

    const int N_fft = sd.OfdmCfg.FFT_SIZE;
    const int guard = (int)std::floor(sd.OfdmCfg.C * (NSYM + RS));
    const int usable = N_fft - 2 * guard;
    const int step = usable / 4;

    guard_out = guard;
    usable_out = usable;

    user_sc.assign(4, {});
    for (int u = 0; u < 4; u++)
    {
        int start = guard + u * step;
        int end = guard + (u == 3 ? usable : (u + 1) * step);

        for (int k = start; k < end; k++)
        {
            bool is_pilot = ((k - guard) % sd.OfdmCfg.RS == 0);
            if (!is_pilot)
                user_sc[u].push_back(k);
        }
    }
}

std::vector<std::complex<float>> ofdma_modulator(SharedData &sd)
{
    ensure_fft_plans(sd);

    int NSYM, RS, CP;
    const int N_fft = compute_OFDM_params(sd.OfdmCfg, NSYM, RS, CP);

    int guard, usable;
    std::vector<std::vector<int>> user_sc;
    build_user_subcarriers(sd, user_sc, guard, usable);

    size_t num_ofdm = 0;
    for (int u = 0; u < 4; u++)
    {
        if (user_sc[u].empty())
            continue;
        size_t n = (sd.users[u].qpsk.size() + user_sc[u].size() - 1) / user_sc[u].size();
        num_ofdm = std::max(num_ofdm, n);
    }

    if (num_ofdm == 0)
        return {};

    const std::complex<float> pilot(1.f, 0.f);

    std::vector<std::complex<float>> output;
    output.reserve(num_ofdm * (N_fft + CP));

    for (size_t sym = 0; sym < num_ofdm; sym++)
    {
        std::vector<std::complex<float>> freq(N_fft, { 0.f, 0.f });

        for (int k = guard; k < N_fft - guard; k += sd.OfdmCfg.RS)
            freq[k] = pilot;

        for (int u = 0; u < 4; u++)
        {
            int sc_u = (int)user_sc[u].size();
            for (int j = 0; j < sc_u; j++)
            {
                size_t idx = sym * sc_u + j;
                if (idx < sd.users[u].qpsk.size())
                    freq[user_sc[u][j]] = sd.users[u].qpsk[idx];
            }
        }

        std::copy_n(freq.begin(), N_fft, reinterpret_cast<std::complex<float> *>(sd.in_ifft));
        fftwf_execute(sd.plan_ifft);

        std::vector<std::complex<float>> time_sym(N_fft);
        std::copy_n(reinterpret_cast<std::complex<float> *>(sd.out_ifft), N_fft, time_sym.begin());

        for (auto &v : time_sym)
            v /= (float)N_fft;

        output.insert(output.end(), time_sym.end() - CP, time_sym.end());
        output.insert(output.end(), time_sym.begin(), time_sym.end());
    }

    return output;
}

void ofdma_demodulator(const std::vector<std::complex<float>> &rx, SharedData &sd)
{
    ensure_fft_plans(sd);

    int NSYM, RS, CP;
    const int N_fft = compute_OFDM_params(sd.OfdmCfg, NSYM, RS, CP);

    int guard, usable;
    std::vector<std::vector<int>> user_sc;
    build_user_subcarriers(sd, user_sc, guard, usable);

    for (int u = 0; u < 4; u++)
        sd.users[u].qpsk.clear();

    size_t symbol_len = N_fft + CP;
    size_t num_symbols = rx.size() / symbol_len;

    std::vector<int> pilot_pos;
    for (int k = guard; k < N_fft - guard; k += sd.OfdmCfg.RS)
        pilot_pos.push_back(k);

    const std::complex<float> pilot(1.f, 0.f);

    std::vector<std::complex<float>> freq(N_fft);
    std::vector<std::complex<float>> H(N_fft, { 1.f, 0.f });
    std::vector<std::complex<float>> freq_eq(N_fft);

    for (size_t sym = 0; sym < num_symbols; sym++)
    {
        size_t pos = sym * symbol_len;

        std::copy_n(rx.begin() + pos + CP, N_fft, reinterpret_cast<std::complex<float> *>(sd.in_fft));
        fftwf_execute(sd.plan_fft);
        std::copy_n(reinterpret_cast<std::complex<float> *>(sd.out_fft), N_fft, freq.begin());

        for (int k : pilot_pos)
            H[k] = freq[k] / pilot;

        for (int p = 0; p + 1 < (int)pilot_pos.size(); p++)
        {
            int k1 = pilot_pos[p];
            int k2 = pilot_pos[p + 1];

            float m1 = std::abs(H[k1]), m2 = std::abs(H[k2]);
            float a1 = std::arg(H[k1]), a2 = std::arg(H[k2]);

            for (int k = k1 + 1; k < k2; k++)
            {
                float t = (float)(k - k1) / (float)(k2 - k1);
                H[k] = std::polar(m1 + t * (m2 - m1), a1 + t * (a2 - a1));
            }
        }

        if (!pilot_pos.empty())
        {
            for (int k = 0; k < pilot_pos.front(); k++)
                H[k] = H[pilot_pos.front()];
            for (int k = pilot_pos.back() + 1; k < N_fft; k++)
                H[k] = H[pilot_pos.back()];
        }

        for (int k = 0; k < N_fft; k++)
            freq_eq[k] = (std::abs(H[k]) > 1e-6f) ? freq[k] / H[k] : freq[k];

        for (int u = 0; u < 4; u++)
            for (int k : user_sc[u])
                sd.users[u].qpsk.push_back(freq_eq[k]);
    }
}