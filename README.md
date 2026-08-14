# Mobile Communication System Model (OFDMA)

An interactive C++ simulation of an OFDMA downlink, visualized with a Dear ImGui / ImPlot GUI. Four users transmit QPSK-modulated, Hamming-coded text through a multipath channel with AWGN; the receiver estimates the channel from scattered pilots and equalizes the subcarriers.

## Features

- **OFDMA** — 4 users share subcarriers with guard bands and scattered pilot signals
- **QPSK modulation** — 30-bit Hamming words mapped to 15 QPSK symbols
- **Hamming (30, 24)** error-correcting code with syndrome-based single-bit correction
- **Bit interleaving / deinterleaving** across codewords to scatter burst errors
- **Channel model** — random multipath with path gains from the free-space loss formula, plus configurable AWGN (PSD in dB)
- **Channel estimation** — LS estimation at pilot positions with linear interpolation of magnitude and phase
- **Live GUI** — subcarrier map, per-user QPSK constellations, RX spectrum, BER history plot
- **Real-time mode** — continuous random-data frames; **Experiment mode** — one-shot frames

## Tech Stack

- C++17
- CMake
- FFTW3 (float FFT)
- SDL2 + OpenGL + GLEW
- Dear ImGui + ImPlot (docking branch, fetched automatically)

## Project Structure

```
.
├── CMakeLists.txt
├── include/
│   └── header.hpp          # Shared data structures and function declarations
├── src/
│   ├── main.cpp            # GUI thread (window, controls, plots)
│   └── back.cpp            # Processing thread (TX/RX pipeline)
└── functions/
    ├── bits_operation.cpp  # Text <-> bytes conversion
    ├── error_detector.cpp  # Hamming (30,24) encoder/decoder
    ├── interleaving.cpp    # Bit interleaving
    ├── modulator.cpp       # QPSK modulator/demodulator
    ├── chanel_model.cpp    # Multipath + AWGN channel
    ├── ofdm_modulator.cpp  # OFDMA mapping, IFFT/FFT, pilot-based equalization
    └── ber.cpp             # Bit error rate calculation
```

## Getting Started

### Prerequisites

- CMake 3.24+
- A C++17 compiler (gcc/clang)
- pkg-config, FFTW3 (`libfftw3-dev` on Debian/Ubuntu: `sudo apt install libfftw3-dev`)

SDL2, GLEW, ImGui and ImPlot are fetched and built automatically if not found on the system.

### Build

```bash
cmake -B build
cmake --build build -j
```

### Run

```bash
./build/main.elf
```

## Usage

- Type a message per user and press **Send** to transmit one frame.
- Toggle **Real Time** to stream random frames continuously.
- Adjust the OFDM parameters (FFT size, pilot spacing, guard ratio, CP ratio) and the channel parameters (number of paths, noise PSD, bandwidth, carrier frequency) — the next frame will use the new settings.
- Watch the subcarrier map, constellations, spectrum and BER history.

## Notes

- The configuration sliders write shared state while the processing thread reads it; changing parameters mid-frame may produce a single inconsistent frame, but no crash.
- The channel generator uses a fixed seed (42) for reproducible noise; uncomment `rd()` in `chanel_model.cpp` to randomize it.