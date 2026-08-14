#include "header.hpp"

#include <chrono>
#include <random>

std::vector<uint8_t> generate_random_data(size_t size)
{
    static std::mt19937 gen(std::chrono::steady_clock::now().time_since_epoch().count());

    static const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                   "abcdefghijklmnopqrstuvwxyz";

    static std::uniform_int_distribution<size_t> dis(0, sizeof(alphabet) - 2);

    std::vector<uint8_t> data(size);

    for (auto &byte : data)
        byte = alphabet[dis(gen)];

    return data;
}

void back(SharedData &sd)
{
    while (sd.back_running.load())
    {
        bool process_frame = false;

        if (sd.is_realtime.load())
        {
            process_frame = true;

            auto data = generate_random_data(70);
            for (int u = 0; u < 4; ++u)
                sd.users[u].tx_buf.assign(data.begin(), data.end());
        }
        else
        {
            if (sd.input_flag.load())
            {
                if (sd.experiment.load())
                    sd.experiment.store(false);

                process_frame = true;
            }
        }

        if (process_frame)
        {
            sd.input_flag.store(false);

            for (int u = 0; u < 4; ++u)
            {
                sd.users[u].bytes = encoder(sd.users[u].tx_buf);
                sd.users[u].encoded = hamming_encoder(sd.users[u].bytes);
                sd.users[u].interleaved = interleaving(sd.users[u].encoded);
                sd.users[u].qpsk = QPSK_modulator(sd.users[u].interleaved);
            }

            for (int u = 0; u < 4; ++u)
            {
                size_t space = sd.users[u].qpsk.size();
                if (space % 15 != 0)
                    sd.users[u].qpsk.resize(((space + 14) / 15) * 15, { 0.707f, 0.707f });
            }

            sd.ofdm_symbols = ofdma_modulator(sd);
            sd.signal = chanel_model(sd.ofdm_symbols, sd);
            sd.rx_spectrum = spectrum_calculate(sd.signal, sd);

            ofdma_demodulator(sd.signal, sd);

            float avg_ber = 0.0f;
            for (int u = 0; u < 4; ++u)
            {
                sd.users[u].interleaved = QPSK_demodulator(sd.users[u].qpsk);
                sd.users[u].encoded = deinterleaving(sd.users[u].interleaved);
                sd.users[u].bytes = hamming_decoder(sd.users[u].encoded);
                sd.users[u].rx_buf = decoder(sd.users[u].bytes);

                sd.users[u].rx_buf.resize(sd.users[u].tx_buf.size());

                std::vector<uint8_t> tx(sd.users[u].tx_buf.begin(), sd.users[u].tx_buf.end());
                std::vector<uint8_t> rx(sd.users[u].rx_buf.begin(), sd.users[u].rx_buf.end());

                sd.users[u].ber = BER(tx, rx);
                avg_ber += sd.users[u].ber;
            }

            std::lock_guard<std::mutex> lock(sd.mtx);

            for (int i = 0; i < 4; ++i){
                sd.qpsk_gui[i].assign(sd.users[i].qpsk.begin(), sd.users[i].qpsk.end());
            }

            sd.symbols_rx.clear();
            for (int u = 0; u < 4; ++u)
            {
                auto &v = sd.users[u].qpsk;
                sd.symbols_rx.insert(sd.symbols_rx.end(), v.begin(), v.end());
            }

            sd.BER = avg_ber / 4.0f;
            sd.BER_vec[sd.ber_vec_offset] = sd.BER;
            sd.ber_vec_offset = (sd.ber_vec_offset + 1) % sd.ber_vec_size;
            sd.ber_frames_processed++;
        }
    }
}