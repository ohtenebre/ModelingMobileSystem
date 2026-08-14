#include "backends/imgui_impl_opengl3.h"
#include "backends/imgui_impl_sdl2.h"
#include "header.hpp"
#include "imgui.h"
#include "implot.h"

#include <complex>
#include <fftw3.h>
#include <mutex>
#include <thread>
#include <vector>

int main(int argc, char *argv[])
{
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER);

    SDL_Window *window = SDL_CreateWindow(
        "OFDMA", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        1920, 1080, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE
    );

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    SDL_GL_SetSwapInterval(0);

    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO &io = ImGui::GetIO();
    ImGuiStyle &style = ImGui::GetStyle();

    style.WindowRounding = 10.f;
    style.FrameRounding = 8.f;
    style.ChildRounding = 8.f;
    style.ScrollbarRounding = 10.f;
    style.TabRounding = 8.f;
    style.WindowBorderSize = 0.0f;
    style.FrameBorderSize = 0.0f;
    style.PopupBorderSize = 0.0f;
    style.WindowPadding = ImVec2(10, 10);
    style.FramePadding = ImVec2(4, 4);
    style.ItemSpacing = ImVec2(8, 8);
    style.ItemInnerSpacing = ImVec2(4, 4);

    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init("#version 330");

    SharedData sd;
    sd.qpsk_gui.resize(4);

    sd.BER_vec.resize(1000, 0.0f);
    sd.ber_vec_offset = 0;
    sd.ber_vec_size = 1000;

    sd.in_ifft = fftwf_alloc_complex(sd.OfdmCfg.FFT_SIZE);
    sd.out_ifft = fftwf_alloc_complex(sd.OfdmCfg.FFT_SIZE);
    sd.in_fft = fftwf_alloc_complex(sd.OfdmCfg.FFT_SIZE);
    sd.out_fft = fftwf_alloc_complex(sd.OfdmCfg.FFT_SIZE);
    sd.in_spectrum = fftwf_alloc_complex(1024);
    sd.out_spectrum = fftwf_alloc_complex(1024);

    sd.plan_ifft = fftwf_plan_dft_1d(sd.OfdmCfg.FFT_SIZE, sd.in_ifft, sd.out_ifft, FFTW_BACKWARD, FFTW_ESTIMATE);
    sd.plan_fft = fftwf_plan_dft_1d(sd.OfdmCfg.FFT_SIZE, sd.in_fft, sd.out_fft, FFTW_FORWARD, FFTW_ESTIMATE);
    sd.plan_spectrum = fftwf_plan_dft_1d(1024, sd.in_spectrum, sd.out_spectrum, FFTW_FORWARD, FFTW_ESTIMATE);

    sd.fft_size_alloc = sd.OfdmCfg.FFT_SIZE;

    std::thread Back(back, std::ref(sd));

    ImVec4 user_colors[4] = {
        { 1.0f, 0.35f, 0.35f, 1.0f },
        { 0.35f, 1.0f, 0.35f, 1.0f },
        { 0.4f, 0.6f, 1.0f, 1.0f },
        { 1.0f, 0.85f, 0.2f, 1.0f },
    };

    ImU32 user_colors32[4] = {
        IM_COL32(200, 80, 80, 255),
        IM_COL32(80, 200, 80, 255),
        IM_COL32(80, 130, 220, 255),
        IM_COL32(220, 200, 60, 255),
    };

    static char tx_inputs[4][101] = {};

    bool running = true;
    while (running)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT)
            {
                running = false;
                sd.back_running.store(false);
            }
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        ImGui::DockSpaceOverViewport(0, nullptr, ImGuiDockNodeFlags_None);

        ImGui::Begin("Config");

        ImGui::SliderInt("FFT Size", &sd.OfdmCfg.FFT_SIZE, 64, 512);
        ImGui::SliderInt("Pilot Space", &sd.OfdmCfg.RS, 4, 30);
        ImGui::SliderFloat("Guard", &sd.OfdmCfg.C, 0, 1);
        ImGui::SliderInt("CP Ratio", &sd.OfdmCfg.CP_ratio, 1, sd.OfdmCfg.FFT_SIZE);

        ImGui::Separator();

        ImGui::SliderInt("Num of beams", &sd.ChannelCfg.N_b, 1, 12);
        ImGui::SliderFloat("PSD (N0 dB)", &sd.ChannelCfg.N_0, -100, 40);
        ImGui::DragInt("Bandwidth", &sd.ChannelCfg.B, 100000, 0, 20000000);
        ImGui::SliderFloat("Carrier Freq", &sd.ChannelCfg.carrier_freq, 1.0f, 3000e6f, "%.3e");

        ImGui::Separator();
        ImGui::Text("Avg BER: %f", sd.BER);

        bool isreal = sd.is_realtime.load();
        if (ImGui::Checkbox("Real Time", &isreal))
            sd.is_realtime.store(isreal);

        bool exp = sd.experiment.load();
        if (ImGui::Checkbox("Experiment", &exp))
            sd.experiment.store(exp);

        ImGui::End();

        ImGui::Begin("OFDMA");

        ImGui::SeparatorText("Message");

        ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);

        if (ImGui::BeginTable("users_table", 4, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoHostExtendX))
        {
            ImGui::TableSetupColumn("User", ImGuiTableColumnFlags_WidthFixed, 80.0f);
            ImGui::TableSetupColumn("TX", ImGuiTableColumnFlags_WidthStretch, 0.25f);
            ImGui::TableSetupColumn("RX", ImGuiTableColumnFlags_WidthStretch, 0.60f);
            ImGui::TableSetupColumn("BER", ImGuiTableColumnFlags_WidthFixed, 100.0f);

            for (int u = 0; u < 4; u++)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                ImGui::PushStyleColor(ImGuiCol_Text, user_colors[u]);
                ImGui::Text("User %d", u + 1);
                ImGui::PopStyleColor();

                ImGui::TableSetColumnIndex(1);
                ImGui::PushID(u);
                ImGui::SetNextItemWidth(-FLT_MIN);
                ImGui::InputText("##tx", tx_inputs[u], 101);
                ImGui::PopID();

                ImGui::TableSetColumnIndex(2);
                char buf[256];
                snprintf(buf, sizeof(buf), "%-60s", sd.users[u].rx_buf.empty() ? "-" : sd.users[u].rx_buf.c_str());

                ImGui::PushStyleColor(ImGuiCol_Text, user_colors[u]);
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();

                ImGui::TableSetColumnIndex(3);
                ImGui::Text("%.6f", sd.users[u].ber);
            }

            ImGui::EndTable();
        }

        ImGui::PopFont();

        if (ImGui::Button("Send"))
        {
            for (int u = 0; u < 4; u++)
                sd.users[u].tx_buf = tx_inputs[u];

            sd.input_flag.store(true);
        }

        ImGui::SeparatorText("SubCarrier map");

        {
            int NSYM, RS, CP;
            compute_OFDM_params(sd.OfdmCfg, NSYM, RS, CP);

            int N_fft = sd.OfdmCfg.FFT_SIZE;
            int guard = (int)floor(sd.OfdmCfg.C * (NSYM + RS));
            int usable = N_fft - 2 * guard;
            int step = (usable > 0 && 4 > 0) ? usable / 4 : 1;

            ImVec2 pos = ImGui::GetCursorScreenPos();
            float bar_w = ImGui::GetContentRegionAvail().x;
            float bar_h = 30.0f;

            ImDrawList *dl = ImGui::GetWindowDrawList();

            for (int i = 0; i < N_fft; i++)
            {
                float x0 = pos.x + (float)i / N_fft * bar_w;
                float x1 = pos.x + (float)(i + 1) / N_fft * bar_w;

                ImU32 col;

                if (i < guard || i >= N_fft - guard)
                {
                    col = IM_COL32(40, 40, 40, 255);
                }
                else
                {
                    int k = i - guard;

                    if ((i - guard) % sd.OfdmCfg.RS == 0)
                    {
                        col = IM_COL32(200, 140, 30, 255);
                    }
                    else
                    {
                        int u = k / step;
                        if (u >= 4)
                            u = 3;
                        col = user_colors32[u];
                    }
                }

                dl->AddRectFilled({ x0, pos.y }, { x1, pos.y + bar_h }, col);
            }

            dl->AddRect(pos, { pos.x + bar_w, pos.y + bar_h }, IM_COL32(80, 80, 80, 255));

            ImGui::Dummy({ bar_w, bar_h });

            ImGui::Spacing();
            ImGui::TextDisabled("  Guard");
            ImGui::SameLine();
            ImGui::TextColored({ 0.78f, 0.55f, 0.12f, 1.f }, "  Pilot");
            ImGui::SameLine();
            for (int u = 0; u < 4; u++)
            {
                ImGui::PushStyleColor(ImGuiCol_Text, user_colors[u]);
                ImGui::Text("  User %d", u + 1);
                ImGui::PopStyleColor();
                ImGui::SameLine();
            }
            ImGui::NewLine();

            ImGui::TextDisabled("FFT=%d  guard=%d  usable=%d  SC/user=%d  pilot_spacing=%d", N_fft, guard, usable, step, sd.OfdmCfg.RS);
        }

        ImGui::SeparatorText("Constelation");

        float csize = (ImGui::GetContentRegionAvail().x - 3 * 8.f) / 4.f;

        for (int u = 0; u < 4; u++)
        {
            if (u > 0)
                ImGui::SameLine();

            ImGui::PushID(100 + u);

            char title[32];
            snprintf(title, sizeof(title), "User %d", u + 1);

            if (ImPlot::BeginPlot(title, ImVec2(csize, csize)))
            {
                ImPlot::SetupAxesLimits(-1.5, 1.5, -1.5, 1.5, ImPlotCond_Always);
                ImPlot::SetupFinish();
                std::vector<std::complex<float>> syms_gui;

                {
                    std::lock_guard<std::mutex> lock(sd.mtx);
                    syms_gui = sd.qpsk_gui[u];
                }

                if (!syms_gui.empty())
                {
                    std::vector<float> I, Q;
                    I.reserve(syms_gui.size());
                    Q.reserve(syms_gui.size());

                    for (auto &s : syms_gui)
                    {
                        I.push_back(s.real());
                        Q.push_back(s.imag());
                    }

                    ImPlot::PlotScatter("##s", I.data(), Q.data(), (int)I.size());
                }

                ImPlot::EndPlot();
            }

            ImGui::PopID();
        }

        ImGui::End();

        ImGui::Begin("Spectrum");

        if (ImPlot::BeginPlot("Spectrum", ImVec2(-1, -1)))
        {
            if (!sd.rx_spectrum.empty())
                ImPlot::PlotLine("dB", sd.rx_spectrum.data(), (int)sd.rx_spectrum.size());
            ImPlot::EndPlot();
        }

        ImGui::End();

        ImGui::Begin("BER History");

        static std::vector<float> ber_buf(1000);

        int render_start = (sd.ber_vec_offset + 1) % sd.ber_vec_size;
        int samples = (int)sd.ber_frames_processed.load();
        if (samples > (int)sd.ber_vec_size)
            samples = (int)sd.ber_vec_size;

        {
            std::lock_guard<std::mutex> lock(sd.mtx);
            for (int i = 0; i < samples; i++)
                ber_buf[i] = sd.BER_vec[(render_start + i) % sd.ber_vec_size];
        }

        if (ImPlot::BeginPlot("BER", ImVec2(-1, -1)))
        {
            ImPlot::SetupAxis(ImAxis_X1, "Frame");
            ImPlot::SetupAxis(ImAxis_Y1, "BER");
            ImPlot::PlotLine("BER", ber_buf.data(), samples);
            ImPlot::EndPlot();
        }

        ImGui::End();

        ImGui::Render();
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(window);
    }

    if (Back.joinable())
        Back.join();

    fftwf_free(sd.in_ifft);
    fftwf_free(sd.out_ifft);
    fftwf_free(sd.in_fft);
    fftwf_free(sd.out_fft);
    fftwf_free(sd.in_spectrum);
    fftwf_free(sd.out_spectrum);

    fftwf_destroy_plan(sd.plan_fft);
    fftwf_destroy_plan(sd.plan_ifft);
    fftwf_destroy_plan(sd.plan_spectrum);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}